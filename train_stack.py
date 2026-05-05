import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import FusedCrossEntropyLoss
from lit_gpt.model import GPT, Config
from stack_data import StackDataset, generate_stack_data


def parse_int_csv(values: str) -> list[int]:
    parsed = []
    for token in values.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 3:
            raise ValueError(f"Invalid sequence length {value}. Must be >= 3.")
        parsed.append(value)
    return sorted(set(parsed))


def validate_dataset_token_range(dataset, dataset_name: str, model_vocab_size: int) -> None:
    input_min = int(dataset.input_ids.min().item())
    input_max = int(dataset.input_ids.max().item())
    if input_min < 0 or input_max >= model_vocab_size:
        raise ValueError(
            (
                f"{dataset_name}: input_ids token range [{input_min}, {input_max}] is incompatible with "
                f"model vocab size {model_vocab_size}. This usually means cached data was generated "
                "with a different vocab_size. Regenerate the dataset with matching vocab_size."
            )
        )

    labels = dataset.labels
    valid_label_mask = labels != -100
    if valid_label_mask.any():
        label_min = int(labels[valid_label_mask].min().item())
        label_max = int(labels[valid_label_mask].max().item())
        if label_min < 0 or label_max >= model_vocab_size:
            raise ValueError(
                (
                    f"{dataset_name}: labels token range [{label_min}, {label_max}] is incompatible with "
                    f"model vocab size {model_vocab_size}. Regenerate cached data with matching vocab_size."
                )
            )


def validate_dataset_has_pop_targets(dataset, dataset_name: str) -> None:
    target_count = int((dataset.labels != -100).sum().item())
    if target_count <= 0:
        raise ValueError(
            (
                f"{dataset_name}: no supervised pop targets found. "
                "Please regenerate data with valid stack settings."
            )
        )


def build_eval_loader(
    fabric,
    args,
    seq_len: int,
    num_examples: int,
):
    cache_root = args.eval_cache_dir or os.path.join(args.data_dir, "eval_cache")
    cache_path = os.path.join(
        cache_root,
        (
            f"val_seq{seq_len}_seed{args.seed + 1000 + seq_len}"
            f"_n{num_examples}_v{args.vocab_size}_s{args.num_stacks}_val{args.num_values}.npz"
        ),
    )

    if fabric.global_rank == 0 and not os.path.exists(cache_path):
        os.makedirs(cache_root, exist_ok=True)
        eval_data = generate_stack_data(
            num_examples=num_examples,
            seq_len=seq_len,
            vocab_size=args.vocab_size,
            num_stacks=args.num_stacks,
            num_values=args.num_values,
            push_prob=args.push_prob,
            seed=args.seed + 1000 + seq_len,
        )
        np.savez_compressed(cache_path, **eval_data)
        fabric.print(f"Generated stack eval data at {cache_path}")

    fabric.barrier()

    val_dataset = StackDataset(data_path=cache_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return fabric.setup_dataloaders(val_dataloader)


def main(args):
    if args.debug:
        wandb_logger = WandbLogger(
            project=os.environ.get("WANDB_PROJECT", "stack_linear_attn"),
            mode="disabled",
            name=args.exp_name,
        )
    else:
        wandb_logger = WandbLogger(
            project=os.environ.get("WANDB_PROJECT", "stack_linear_attn"),
            name=args.exp_name,
            save_dir=args.wandb_dir,
        )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        fabric = L.Fabric(devices=args.devices, strategy="auto", precision="bf16-mixed", loggers=[wandb_logger])
    else:
        fabric = L.Fabric(devices=1, strategy="auto", precision="32-true", loggers=[wandb_logger])

    fabric.launch()
    fabric.seed_everything(args.seed)

    if fabric.global_rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        fabric.print(args)

    train_dataset = StackDataset(data_path=os.path.join(args.data_dir, "train.npz"))
    val_dataset = StackDataset(data_path=os.path.join(args.data_dir, "val.npz"))
    test_dataset = StackDataset(data_path=os.path.join(args.data_dir, "test.npz"))

    train_seq_len = int(train_dataset.input_ids.shape[1])

    eval_seq_lens = parse_int_csv(args.eval_seq_lens) if args.eval_seq_lens.strip() else []
    eval_seq_lens.append(train_seq_len)
    eval_seq_lens = sorted(set(eval_seq_lens))

    eval_num_val = args.eval_num_val if args.eval_num_val > 0 else len(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_dataloader, val_dataloader, test_dataloader = fabric.setup_dataloaders(
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )

    config = Config.from_name(args.model_name)

    if args.n_embd:
        config.n_embd = args.n_embd
    if args.n_head:
        config.n_head = args.n_head
    if args.n_layer:
        config.n_layer = args.n_layer

    required_block_size = max(eval_seq_lens)
    if config.block_size < required_block_size:
        config.block_size = required_block_size

    if config.n_embd % config.n_head != 0:
        config.n_head = max(1, config.n_embd // 64)

    model_vocab_size = int(config.padded_vocab_size)
    if args.vocab_size > model_vocab_size:
        raise ValueError(
            (
                f"Requested data/eval vocab_size={args.vocab_size} exceeds model vocab size={model_vocab_size}. "
                "Please lower --vocab_size or use a model config with larger vocab."
            )
        )

    validate_dataset_token_range(train_dataset, "train.npz", model_vocab_size)
    validate_dataset_token_range(val_dataset, "val.npz", model_vocab_size)
    validate_dataset_token_range(test_dataset, "test.npz", model_vocab_size)

    validate_dataset_has_pop_targets(train_dataset, "train.npz")
    validate_dataset_has_pop_targets(val_dataset, "val.npz")
    validate_dataset_has_pop_targets(test_dataset, "test.npz")

    if fabric.global_rank == 0:
        fabric.print(
            (
                f"Stack eval setup: eval_seq_lens={eval_seq_lens}, "
                f"train_seq_len={train_seq_len}, eval_num_val={eval_num_val}, "
                f"block_size={config.block_size}, model_vocab_size={model_vocab_size}"
            )
        )

    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

    model = fabric.setup(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model,
        "optimizer": optimizer,
        "iter_num": 0,
        "best_val_acc": 0.0,
        "no_improve_count": 0,
        "train_history": [],
        "val_history": [],
    }

    loss_func = FusedCrossEntropyLoss()

    best_model_path = train(fabric, state, train_dataloader, val_dataloader, loss_func, args)

    if best_model_path and os.path.exists(best_model_path):
        fabric.print(f"Loading best model from {best_model_path} for testing...")
        checkpoint = torch.load(best_model_path, map_location=fabric.device)
        model.load_state_dict(checkpoint["model"])

        eval_acc_by_seq_len = {}
        for seq_len in eval_seq_lens:
            eval_loader = build_eval_loader(
                fabric=fabric,
                args=args,
                seq_len=seq_len,
                num_examples=eval_num_val,
            )

            acc = validate(fabric, model, eval_loader)
            eval_acc_by_seq_len[str(seq_len)] = float(acc)
            fabric.log_dict({f"val/final_acc_seq{seq_len}": float(acc)})
            fabric.print(f"Final Val Acc @seq{seq_len}: {acc:.4f}")

        test_acc = validate(fabric, model, test_dataloader)
        fabric.print(f"Test Acc: {test_acc:.4f}")
        fabric.log_dict({"test/acc": test_acc})

        results = {
            "model_name": args.model_name,
            "test_acc": float(test_acc),
            "best_val_acc": float(state["best_val_acc"]),
            "eval_acc_by_seq_len": eval_acc_by_seq_len,
            "train_history": state["train_history"],
            "val_history": state["val_history"],
            "args": vars(args),
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

    best_checkpoints = []
    best_model_path = None

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits = model(input_ids)

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = loss_func(logits_flat, labels_flat)

        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            acc = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)

        state["iter_num"] += 1
        step += 1

        if step % args.log_interval == 0:
            train_entry = {
                "step": int(step),
                "loss": float(loss.item()),
                "acc": float(acc.item()),
            }
            state["train_history"].append(train_entry)

            fabric.log_dict(
                {
                    "train/loss": train_entry["loss"],
                    "train/acc": train_entry["acc"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )
            if fabric.global_rank == 0:
                print(f"Step {step}: Loss {loss.item():.4f}, Acc {acc.item():.4f}")

        if step % val_interval == 0:
            val_acc = validate(fabric, model, val_dataloader)
            val_entry = {"step": int(step), "acc": float(val_acc)}
            state["val_history"].append(val_entry)

            fabric.log_dict({"val/acc": val_acc}, step=step)
            if fabric.global_rank == 0:
                print(f"Step {step}: Val Acc {val_acc:.4f}")

            if best_model_path is None or val_acc > state["best_val_acc"]:
                state["best_val_acc"] = val_acc
                state["no_improve_count"] = 0

                save_path = os.path.join(args.out_dir, f"best_model_step_{step}.pth")
                fabric.save(save_path, state)
                best_model_path = save_path

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
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits = model(input_ids)
        preds = torch.argmax(logits, dim=-1)

        mask = labels != -100
        correct = (preds == labels) & mask

        total_correct += correct.sum()
        total_masked += mask.sum()

    model.train()

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
    parser.add_argument("--exp_name", type=str, default="stack_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--eval_seq_lens", type=str, default="")
    parser.add_argument("--eval_num_val", type=int, default=2000)
    parser.add_argument("--eval_cache_dir", type=str, default=None)

    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--num_stacks", type=int, default=64)
    parser.add_argument("--num_values", type=int, default=26)
    parser.add_argument("--push_prob", type=float, default=0.6)

    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)

    main(parser.parse_args())
