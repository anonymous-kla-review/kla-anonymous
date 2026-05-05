import torch
import sys
import os
import argparse
import math
import time
import csv
import json
from pathlib import Path
from typing import List

# Support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from lit_gpt.model import GPT, Config, Block, MBlock
from lit_gpt.utils import chunked_cross_entropy
from lit_gpt.packed_dataset import PackedDataset, CombinedDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import glob
import random

def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate PPL on different context lengths')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--config_name', type=str, required=True, help='Name of the model config (e.g. GatedDeltaNet_0.4B)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to validation data directory')
    parser.add_argument('--lengths', type=str, default='1024,2048,4096,8192,16384', help='Comma separated list of context lengths to evaluate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--eval_iters', type=int, default=10, help='Number of iterations for evaluation')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='linear_attn_eval', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default='', help='Wandb run name')
    parser.add_argument('--wandb_dir', type=str, default='./wandb', help='Wandb save directory')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging for offline evaluation')

    # Optional artifact outputs for downstream aggregation/plotting.
    parser.add_argument('--model_alias', type=str, default='', help='Optional model alias used in output artifacts')
    parser.add_argument('--save_json', type=str, default='', help='Path to save evaluation metrics JSON')
    parser.add_argument('--save_csv', type=str, default='', help='Path to save evaluation metrics CSV')
    
    return parser

@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    lengths: List[int],
    eval_iters: int,
    wandb_logger=None,
):
    fabric.print("Validating ...")
    model.eval()

    # Track per-length sums/counts. Some long lengths may be skipped if a batch is too short.
    loss_sums = {l: torch.zeros(1, device=fabric.device) for l in lengths}
    loss_counts = {l: 0 for l in lengths}
    
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        
        for length in lengths:
            # Check if we have enough data
            if val_data.size(1) < length + 1:
                # fabric.print(f"Warning: Batch size {val_data.size(1)} is smaller than required length {length+1}. Skipping this length for this batch.")
                continue

            input_ids = val_data[:, 0 : length].contiguous()
            targets = val_data[:, 1 : length + 1].contiguous()
            
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            loss_sums[length] += loss.detach()
            loss_counts[length] += 1
    
    fabric.print("\n" + "="*50)
    fabric.print(f"{'Context Length':<15} | {'Loss':<10} | {'PPL':<10}")
    fabric.print("-" * 50)

    result_rows = []
    
    for length in sorted(lengths):
        count = loss_counts[length]
        if count == 0:
            avg_loss = float('nan')
            ppl = float('nan')
            fabric.print(f"{length:<15} | {'nan':<10} | {'nan':<10} (no valid batches)")
        else:
            avg_loss = (loss_sums[length] / count).item()
            try:
                ppl = math.exp(avg_loss)
            except OverflowError:
                ppl = float('inf')
            fabric.print(f"{length:<15} | {avg_loss:.4f}     | {ppl:.4f}")

            # Log specific metrics for each length (scalar comparison/table use).
            fabric.log_dict({
                f"eval/loss_{length}": avg_loss,
                f"eval/ppl_{length}": ppl
            })

            # Log shared metrics for curve plotting (PPL vs Context Length).
            if wandb_logger is not None and fabric.global_rank == 0:
                wandb_logger.experiment.log({
                    "curve/context_length": length,
                    "curve/ppl": ppl,
                    "curve/loss": avg_loss
                })

        result_rows.append(
            {
                "context_length": int(length),
                "loss": float(avg_loss),
                "ppl": float(ppl),
                "num_batches": int(count),
            }
        )
    
    fabric.print("="*50 + "\n")
    model.train()
    return result_rows


def _ensure_parent(path: str) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _baseline_info(rows, preferred_baseline: int = 4096):
    finite = [row for row in rows if math.isfinite(row["ppl"])]
    if not finite:
        return None, None

    by_length = {row["context_length"]: row for row in finite}
    if preferred_baseline in by_length:
        base = by_length[preferred_baseline]
        return preferred_baseline, base["ppl"]

    shortest = min(finite, key=lambda x: x["context_length"])
    return shortest["context_length"], shortest["ppl"]


def _enrich_rows_with_baseline(rows, baseline_length: int | None, baseline_ppl: float | None):
    enriched = []
    for row in sorted(rows, key=lambda x: x["context_length"]):
        new_row = dict(row)
        if baseline_length is None or baseline_ppl is None:
            new_row["baseline_length"] = None
            new_row["baseline_ppl"] = None
            new_row["ppl_ratio_vs_baseline"] = None
            new_row["ppl_increase_pct_vs_baseline"] = None
        elif math.isfinite(row["ppl"]):
            ratio = row["ppl"] / baseline_ppl if baseline_ppl > 0 else float('nan')
            new_row["baseline_length"] = int(baseline_length)
            new_row["baseline_ppl"] = float(baseline_ppl)
            new_row["ppl_ratio_vs_baseline"] = float(ratio)
            new_row["ppl_increase_pct_vs_baseline"] = float((ratio - 1.0) * 100.0)
        else:
            new_row["baseline_length"] = int(baseline_length)
            new_row["baseline_ppl"] = float(baseline_ppl)
            new_row["ppl_ratio_vs_baseline"] = None
            new_row["ppl_increase_pct_vs_baseline"] = None
        enriched.append(new_row)
    return enriched


def maybe_save_results(args, rows) -> None:
    if not args.save_json and not args.save_csv:
        return

    baseline_length, baseline_ppl = _baseline_info(rows)
    enriched = _enrich_rows_with_baseline(rows, baseline_length, baseline_ppl)

    payload = {
        "model_name": args.model_alias or args.config_name,
        "config_name": args.config_name,
        "ckpt_path": args.ckpt_path,
        "data_dir": args.data_dir,
        "lengths": [int(l) for l in sorted(set(int(v["context_length"]) for v in rows))],
        "eval_iters": int(args.eval_iters),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "baseline_length": baseline_length,
        "baseline_ppl": baseline_ppl,
        "results": enriched,
    }

    if args.save_json:
        _ensure_parent(args.save_json)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {args.save_json}")

    if args.save_csv:
        _ensure_parent(args.save_csv)
        fieldnames = [
            "context_length",
            "loss",
            "ppl",
            "num_batches",
            "baseline_length",
            "baseline_ppl",
            "ppl_ratio_vs_baseline",
            "ppl_increase_pct_vs_baseline",
        ]
        with open(args.save_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in enriched:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"Saved CSV: {args.save_csv}")

def create_val_dataloader(
    batch_size: int,
    max_seq_len: int,
    data_dir: Path,
    fabric: L.Fabric,
    seed: int = 12345,
) -> DataLoader:
    # Use the max length required + 1
    block_size = max_seq_len + 1
    
    # Simple logic to find validation files
    # Assuming standard structure like 'validation*'
    filenames = sorted(glob.glob(os.path.join(data_dir, "validation*")))
    if not filenames:
         # Fallback to look for .bin files if specific prefix not found
         filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    
    if not filenames:
        raise RuntimeError(f"No data found at {data_dir}")

    dataset = PackedDataset(
        filenames,
        n_chunks=1, # Simplify for eval
        block_size=block_size,
        shuffle=False,
        seed=seed,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    lengths = [int(l) for l in args.lengths.split(',')]
    max_len = max(lengths)
    
    # Initialize optional Wandb logger.
    wandb_logger = None
    if not args.disable_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name or f"eval_{args.config_name}_{int(time.time())}",
            save_dir=args.wandb_dir,
            version=args.wandb_name
        )

    # Initialize Fabric
    # Use bfloat16-mixed if cuda is available, else 32-true
    precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    strategy = "auto"
    
    fabric_kwargs = {
        "accelerator": args.device,
        "strategy": strategy,
        "precision": precision,
    }
    if wandb_logger is not None:
        fabric_kwargs["loggers"] = [wandb_logger]

    fabric = L.Fabric(**fabric_kwargs)
    fabric.launch()
    fabric.seed_everything(args.seed)
    
    # Log hyperparameters only when wandb is enabled.
    if wandb_logger is not None and fabric.global_rank == 0:
        wandb_logger.log_hyperparams(args)

    if fabric.global_rank == 0:
        fabric.print(f"Loading config: {args.config_name}")
        fabric.print(f"Checkpoint: {args.ckpt_path}")
        fabric.print(f"Lengths to evaluate: {lengths}")
    
    config = Config.from_name(args.config_name)
    
    # IMPORTANT: Update block_size in config to match the max length we want to evaluate
    # The model's forward pass usually relies on config.block_size for mask/rope generation
    if config.block_size < max_len:
        fabric.print(f"Updating config.block_size from {config.block_size} to {max_len} to support long context evaluation.")
        config.block_size = max_len
        # Also update local_window if necessary? Usually local_window is fixed, but let's keep it as is unless it causes issues.

    with fabric.init_module(empty_init=False):
        model = GPT(config)
    
    # Load checkpoint
    # Fabric load expects a dict with 'model' key if the saved state was a dict
    # Based on pretrain.py: fabric.save(checkpoint_path, state) where state has 'model', 'optimizer', etc.
    # So we should create a dummy state dict
    state = {"model": model}
    fabric.print(f"Loading model weights...")
    fabric.load(args.ckpt_path, state)
    
    # Setup dataloader
    val_dataloader = create_val_dataloader(
        batch_size=args.batch_size,
        max_seq_len=max_len,
        data_dir=args.data_dir,
        fabric=fabric,
        seed=args.seed
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    
    # Run validation
    rows = validate(
        fabric,
        model,
        val_dataloader,
        lengths,
        args.eval_iters,
        wandb_logger=wandb_logger,
    )
    if fabric.global_rank == 0:
        maybe_save_results(args, rows)

if __name__ == "__main__":
    main()
