import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
import os
import argparse
import json
import re
# import wandb
import numpy as np

# Adjust path to import from local directories
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, MBlock, Config
from lit_gpt.utils import num_parameters
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
from mqar_data import MQARDataset, generate_mqar_data


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_extrapolation_factors(factors: str) -> list[int]:
    parsed = []
    for token in factors.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"Invalid extrapolation factor {value}. Factors must be >= 1.")
        parsed.append(value)

    if not parsed:
        parsed = [1]
    if 1 not in parsed:
        parsed.append(1)

    return sorted(set(parsed))


def infer_seq_and_key_len_from_data_dir(data_dir: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"seq(\d+)_key(\d+)", data_dir)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def build_extrapolation_val_loader(
    fabric,
    args,
    seq_len: int,
    key_len: int,
    num_examples: int,
    factor: int,
):
    cache_root = args.extrapol_cache_dir or os.path.join(args.data_dir, "extrapolation_cache")
    cache_path = os.path.join(
        cache_root,
        (
            f"val_seq{seq_len}_key{key_len}_pairs{args.num_pairs}"
            f"_seed{args.seed + 1000 + factor}_n{num_examples}.npz"
        ),
    )

    if fabric.global_rank == 0 and not os.path.exists(cache_path):
        os.makedirs(cache_root, exist_ok=True)
        extrapol_data = generate_mqar_data(
            num_examples=num_examples,
            seq_len=seq_len,
            key_len=key_len,
            num_pairs=args.num_pairs,
            seed=args.seed + 1000 + factor,
        )
        np.savez_compressed(cache_path, **extrapol_data)
        fabric.print(f"Generated extrapolation val data at {cache_path}")

    fabric.barrier()

    val_dataset = MQARDataset(data_path=cache_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    return val_dataloader

def main(args):
    # Setup Logger
    if args.debug:
        wandb_logger = WandbLogger(project=os.environ.get("WANDB_PROJECT", "mqar_linear_attn"), mode='disabled', name=args.exp_name)
    else:
        wandb_logger = WandbLogger(project=os.environ.get("WANDB_PROJECT", "mqar_linear_attn"), name=args.exp_name, save_dir=args.wandb_dir)

    # Setup Fabric
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        strategy = "auto" # Single GPU or DDP is fine for small models
        fabric = L.Fabric(devices=args.devices, strategy=strategy, precision="bf16-mixed", loggers=[wandb_logger])
    else:
        fabric = L.Fabric(devices=1, strategy="auto", precision="32-true", loggers=[wandb_logger])

    fabric.launch()
    fabric.seed_everything(args.seed)

    if fabric.global_rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        fabric.print(args)

    # Load Data
    train_dataset = MQARDataset(data_path=os.path.join(args.data_dir, "train.npz"))
    val_dataset = MQARDataset(data_path=os.path.join(args.data_dir, "val.npz"))
    test_dataset = MQARDataset(data_path=os.path.join(args.data_dir, "test.npz"))

    inferred_seq_len, inferred_key_len = infer_seq_and_key_len_from_data_dir(args.data_dir)
    train_seq_len = int(train_dataset.input_ids.shape[1])
    base_eval_seq_len = args.extrapol_base_seq_len or inferred_seq_len or train_seq_len
    eval_key_len = args.extrapol_key_len or inferred_key_len or 1
    extrapolation_factors = parse_extrapolation_factors(args.extrapolation_factors)
    extrapol_num_val = args.extrapol_num_val if args.extrapol_num_val > 0 else len(val_dataset)

    if fabric.global_rank == 0:
        fabric.print(
            (
                f"Final extrapolation validation setup: factors={extrapolation_factors}, "
                f"base_seq_len={base_eval_seq_len}, key_len={eval_key_len}, "
                f"num_val_examples={extrapol_num_val}"
            )
        )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    train_dataloader, val_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader, test_dataloader)

    # Load Model
    config = Config.from_name(args.model_name)

    # Override config params from args if provided
    if args.n_embd: config.n_embd = args.n_embd
    if args.n_head: config.n_head = args.n_head
    if args.n_layer: config.n_layer = args.n_layer
    if args.qk_norm is not None: config.qk_norm = args.qk_norm
    if args.seq_factor_mode is not None: config.seq_factor_mode = args.seq_factor_mode
    if args.gate_mode is not None: config.gate_mode = args.gate_mode
    if args.expand_k is not None: config.expand_k = args.expand_k
    if args.expand_v is not None: config.expand_v = args.expand_v
    if args.learned_norm_init is not None: config.learned_norm_init = args.learned_norm_init
    if args.use_mamba_gate is not None: config.use_mamba_gate = args.use_mamba_gate

    # Ensure head_size consistency
    if config.n_embd % config.n_head != 0:
        config.n_head = config.n_embd // 64 # Default head size 64

    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

    model = fabric.setup(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # Training Loop
    state = {
        "model": model,
        "optimizer": optimizer,
        "iter_num": 0,
        "best_val_acc": 0.0,
        "no_improve_count": 0
    }

    loss_func = FusedCrossEntropyLoss()

    best_model_path = train(fabric, state, train_dataloader, val_dataloader, loss_func, args)

    # Test with best model
    if best_model_path and os.path.exists(best_model_path):
        fabric.print(f"Loading best model from {best_model_path} for testing...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model"])

        final_val_acc = {}
        for factor in extrapolation_factors:
            if factor == 1:
                eval_loader = val_dataloader
            else:
                eval_loader = build_extrapolation_val_loader(
                    fabric=fabric,
                    args=args,
                    seq_len=base_eval_seq_len * factor,
                    key_len=eval_key_len,
                    num_examples=extrapol_num_val,
                    factor=factor,
                )

            acc = validate(fabric, model, eval_loader)
            final_val_acc[f"{factor}x"] = float(acc)
            fabric.log_dict({f"val/final_acc@{factor}x": float(acc)})
            fabric.print(f"Final Val Acc @{factor}x: {acc:.4f}")

        test_acc = validate(fabric, model, test_dataloader)
        fabric.print(f"Test Acc: {test_acc:.4f}")
        fabric.log_dict({"test/acc": test_acc})

        # Save results to JSON
        results = {
            "model_name": args.model_name,
            "test_acc": float(test_acc),
            "best_val_acc": float(state["best_val_acc"]),
            "final_val_acc": final_val_acc,
            "args": vars(args)
        }
        if fabric.global_rank == 0:
            with open(os.path.join(args.out_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
    else:
        fabric.print("No best model found or training failed.")

def train(fabric, state, train_dataloader, val_dataloader, loss_func, args):
    model = state["model"]
    optimizer = state["optimizer"]

    max_steps = args.max_steps
    val_interval = args.val_interval
    save_interval = args.save_interval

    step = 0
    train_iter = iter(train_dataloader)

    best_checkpoints = [] # Keep track of (acc, path)
    best_model_path = None
    consecutive_nonfinite = 0

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        input_ids = batch['input_ids']
        labels = batch['labels']

        logits = model(input_ids)

        # Flatten for loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = loss_func(logits_flat, labels_flat)

        update_succeeded = True
        if not torch.isfinite(loss):
            update_succeeded = False
            if fabric.global_rank == 0:
                print(f"Step {step + 1}: non-finite loss detected ({loss.item()}); skipping optimizer step.")
        else:
            fabric.backward(loss)
            try:
                fabric.clip_gradients(model, optimizer, max_norm=1.0)
            except RuntimeError as exc:
                if "non-finite" in str(exc).lower():
                    update_succeeded = False
                    if fabric.global_rank == 0:
                        print(f"Step {step + 1}: non-finite gradient norm detected; skipping optimizer step.")
                else:
                    raise

        if update_succeeded:
            optimizer.step()
            optimizer.zero_grad()
            consecutive_nonfinite = 0
        else:
            optimizer.zero_grad()
            consecutive_nonfinite += 1
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * args.nonfinite_lr_decay, args.min_learning_rate)
            fabric.log_dict(
                {
                    "train/nonfinite_events": float(consecutive_nonfinite),
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=step + 1,
            )
            if consecutive_nonfinite >= args.max_consecutive_nonfinite:
                if fabric.global_rank == 0:
                    print(
                        "Too many consecutive non-finite updates "
                        f"({consecutive_nonfinite}); stopping early."
                    )
                break

        # Compute accuracy on this batch (only on masked tokens)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            acc = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)

        state["iter_num"] += 1
        step += 1

        # Logging
        if step % args.log_interval == 0:
            fabric.log_dict({
                "train/loss": loss.item(),
                "train/acc": acc.item(),
                "train/lr": optimizer.param_groups[0]['lr']
            }, step=step)
            if fabric.global_rank == 0:
                print(f"Step {step}: Loss {loss.item():.4f}, Acc {acc.item():.4f}")

        # Validation
        if step % val_interval == 0:
            val_acc = validate(fabric, model, val_dataloader)
            fabric.log_dict({"val/acc": val_acc}, step=step)
            if fabric.global_rank == 0:
                print(f"Step {step}: Val Acc {val_acc:.4f}")

            # Early stopping check
            # Always keep the first validation checkpoint as best baseline.
            # This avoids missing best_model/results when val_acc stays at 0.0.
            if best_model_path is None or val_acc > state["best_val_acc"]:
                state["best_val_acc"] = val_acc
                state["no_improve_count"] = 0

                # Save best model
                save_path = os.path.join(args.out_dir, f"best_model_step_{step}.pth")
                fabric.save(save_path, state)
                best_model_path = save_path

                # Maintain top 5
                best_checkpoints.append((val_acc, save_path))
                best_checkpoints.sort(key=lambda x: x[0], reverse=True)
                if len(best_checkpoints) > 5:
                    to_remove = best_checkpoints.pop()
                    if os.path.exists(to_remove[1]):
                        os.remove(to_remove[1])
            else:
                state["no_improve_count"] += 1
                if state["no_improve_count"] >= args.early_stop_patience:
                    if fabric.global_rank == 0:
                        print(f"Early stopping at step {step}")
                    break

        # Checkpoint
        if step % save_interval == 0:
            save_path = os.path.join(args.out_dir, f"ckpt_step_{step}.pth")
            fabric.save(save_path, state)

    return best_model_path

@torch.no_grad()
def validate(fabric, model, dataloader):
    model.eval()
    total_correct = torch.tensor(0.0, device=fabric.device)
    total_masked = torch.tensor(0.0, device=fabric.device)

    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']

        logits = model(input_ids)
        preds = torch.argmax(logits, dim=-1)

        mask = labels != -100
        correct = (preds == labels) & mask

        total_correct += correct.sum()
        total_masked += mask.sum()

    model.train()

    # Sync across devices
    total_correct = fabric.all_reduce(total_correct, reduce_op="sum")
    total_masked = fabric.all_reduce(total_masked, reduce_op="sum")

    if total_masked.item() > 0:
        return (total_correct / total_masked).item()
    return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--wandb_dir", type=str, default="./wandb")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="mqar_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_consecutive_nonfinite", type=int, default=20)
    parser.add_argument("--nonfinite_lr_decay", type=float, default=0.5)
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)

    parser.add_argument("--extrapolation_factors", type=str, default="1,2,4,8")
    parser.add_argument("--extrapol_num_val", type=int, default=2000)
    parser.add_argument("--extrapol_base_seq_len", type=int, default=None)
    parser.add_argument("--extrapol_key_len", type=int, default=None)
    parser.add_argument("--extrapol_cache_dir", type=str, default=None)
    parser.add_argument("--num_pairs", type=int, default=32)

    parser.add_argument("--qk_norm", type=str, default=None)
    parser.add_argument("--seq_factor_mode", type=str, default=None)
    parser.add_argument("--gate_mode", type=str, default=None)
    parser.add_argument("--expand_k", type=float, default=None)
    parser.add_argument("--expand_v", type=float, default=None)
    parser.add_argument("--learned_norm_init", type=float, default=None)
    parser.add_argument("--use_mamba_gate", type=str2bool, default=None)

    # Model overrides
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)

    args = parser.parse_args()
    main(args)
