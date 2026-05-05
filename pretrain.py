# Modified by Songlin Yang & Ali Hatamizadeh

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
import torch.nn.functional as F
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, MBlock, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import num_parameters
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from lit_gpt import FusedCrossEntropyLoss
import random
import os
import argparse
import time
import torch.multiprocessing as mp
import shutil
from distutils.dir_util import copy_tree
import pdb
os.environ["TRITON_CACHE_MANAGER"] = "cache:ParallelFileCacheManager"


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def find_resume_checkpoint(out_dir: str) -> str | None:
    """Return a checkpoint path to resume from, preferring latest then final."""
    candidates = [
        os.path.join(out_dir, "latest-model-ckpt.pth"),
        os.path.join(out_dir, "final-model-ckpt.pth"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    all_ckpts = sorted(
        glob.glob(os.path.join(out_dir, "*-model-ckpt.pth")),
        key=os.path.getmtime,
        reverse=True,
    )
    return all_ckpts[0] if all_ckpts else None

def main(args):
    if args.debug:
        wandb_logger = WandbLogger(project=os.environ.get("WANDB_PROJECT", "linear_attn"), mode='disabled', name=args.exp_name, id=args.exp_name, save_dir=args.wandb_dir, dir=args.wandb_dir, version=args.exp_name, group="debug")
    else:
        wandb_logger = WandbLogger(project=os.environ.get("WANDB_PROJECT", "linear_attn"), name=args.exp_name, id=args.exp_name, save_dir=args.wandb_dir, dir=args.wandb_dir, version=args.exp_name, group=args.exp_group)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if args.interactive_job:
            strategy = FSDPStrategy(auto_wrap_policy={Block,MBlock}, state_dict_type="full")
        else:
            strategy = FSDPStrategy(auto_wrap_policy={Block,MBlock}, state_dict_type="full", sharding_strategy='HYBRID_SHARD')
        fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed", loggers=[wandb_logger])
    else:
        strategy = "auto"
        fabric = L.Fabric(devices=1, strategy=strategy, precision="32-true", loggers=[wandb_logger])
    fabric.launch()
    fabric.seed_everything(args.seed)
    fabric.print("##### Infra Details #####")
    fabric.print(f"Number of Nodes: {args.nodes}")
    fabric.print(f"Number of GPUs: {fabric.world_size}")
    fabric.print("##### Training Details #####")
    fabric.print(f"Maximum number of training tokens: {args.max_tokens}")
    fabric.print(f"Micro batch size: {args.micro_batch_size}")
    fabric.print(f"Batch size: {args.batch_size}")

    if fabric.global_rank == 0:
        fabric.print(args)
    fabric.logger.log_hyperparams(args)
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=args.log_iter_interval)

    resume_path = find_resume_checkpoint(args.out_dir)
    if os.path.exists(args.out_dir):
        args.resume = resume_path is not None
        if args.resume:
            print('Resuming from {}'.format(args.out_dir))
            if fabric.global_rank == 0:
                fabric.print(f"Auto-selected checkpoint: {resume_path}")
        elif fabric.global_rank == 0:
            expected_ckpts = [
                os.path.join(args.out_dir, "latest-model-ckpt.pth"),
                os.path.join(args.out_dir, "final-model-ckpt.pth"),
            ]
            fabric.print(f"Found existing output directory but no checkpoint at {expected_ckpts}. Starting from scratch.")
    else:
        if fabric.global_rank == 0:
            os.makedirs(args.out_dir)
            target_litgpt_save_dir = os.path.join(args.out_dir, 'lit_gpt')
            target_bash_scripts_save_dir = os.path.join(args.out_dir, 'bash_scripts')
            os.makedirs(target_litgpt_save_dir)
            os.makedirs(target_bash_scripts_save_dir)

    config = Config.from_name(args.model_name)
    if args.qk_norm is not None:
        config.qk_norm = args.qk_norm
    if args.seq_factor_mode is not None:
        config.seq_factor_mode = args.seq_factor_mode
    if args.gate_mode is not None:
        config.gate_mode = args.gate_mode
    if args.expand_k is not None:
        config.expand_k = args.expand_k
    if args.expand_v is not None:
        config.expand_v = args.expand_v
    if args.learned_norm_init is not None:
        config.learned_norm_init = args.learned_norm_init
    if args.use_mamba_gate is not None:
        config.use_mamba_gate = args.use_mamba_gate

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        seed=args.seed,
        )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if fabric.global_rank == 0:
        fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        # Ensure all parameters are in float32 before FSDP wrapping to avoid mixed precision errors
        model = model.to(torch.float32)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))

    if fabric.global_rank == 0:
        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        fabric.print(f"Total parameters {num_parameters(model.transformer.h):,}")
        fabric.print(model)

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), fused=True
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": args.hparams, "iter_num": 0, "step_count": 0}

    if args.resume:
        try:
            if resume_path is None:
                raise RuntimeError(f"Resume requested but no checkpoint found in {args.out_dir}")
            if fabric.global_rank == 0:
                fabric.print(f"Resuming training from {resume_path}")
            fabric.load(resume_path, state)
            fabric.print(f"Successfully resumed from {resume_path}")
        except Exception as e:
            fabric.print(f"Failed to resume from {resume_path}: {e!r}")
            raise RuntimeError(f"Resume requested but loading checkpoint failed: {resume_path}") from e
    train_time = time.perf_counter()
    train(args, fabric, state, train_dataloader, val_dataloader, monitor, args.resume)
    if fabric.global_rank == 0:
        fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        if fabric.global_rank == 0:
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(args, fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]
    total_lengths = 0
    total_t0 = time.perf_counter()
    max_tokens_per_device = args.max_tokens // fabric.world_size
    tokens_per_iter = args.micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    warmup_iters = args.warmup_tokens // fabric.world_size // tokens_per_iter
    initial_iter = state["iter_num"]
    curr_iter = 0
    loss_func = FusedCrossEntropyLoss()
    tokens = 0
    train_t0 = time.perf_counter()

    if args.eval_before_training:
        fabric.print("Do validation before training:")
        val_loss = validate(args, fabric, model, val_dataloader, None)
        for i in range(args.num_extrapol):
            if fabric.global_rank == 0:
                fabric.print(f"step {state['iter_num']} {i+1} x: val loss {val_loss[i]:.4f}")

    def run_validation(trigger_message: str, current_tokens: int | None = None):
        if val_dataloader is None:
            return

        if fabric.global_rank == 0:
            if current_tokens is None:
                fabric.print(f"Triggering evaluation: {trigger_message}")
            else:
                fabric.print(f"Triggering evaluation at {current_tokens} tokens ({trigger_message})...")

        t0_eval = time.perf_counter()
        val_loss_all = validate(args, fabric, model, val_dataloader, args.eval_iters)
        t1_eval = time.perf_counter() - t0_eval
        monitor.eval_end(t1_eval)

        if fabric.global_rank == 0 and current_tokens is not None:
            val_loss_val = val_loss_all[0].item()
            val_ppl = math.exp(val_loss_val)
            fabric.print(f"Recorded metrics at {current_tokens} tokens: loss {val_loss_val:.4f}, ppl {val_ppl:.4f}")

        for i in range(args.num_extrapol):
            if fabric.global_rank == 0:
                fabric.print(f"step {state['iter_num']} {i+1} x: val loss {val_loss_all[i]:.4f}, val time: {t1_eval * 1000:.2f}ms")
                fabric.log_dict({"metric/val_loss@"+str(i+1)+"x": val_loss_all[i].item()}, state["step_count"])
                fabric.log_dict({"metric/val_ppl@"+str(i+1)+"x": math.exp(val_loss_all[i].item())}, state["step_count"])

        fabric.barrier()

    def save_checkpoint(final=False):
        name = 'latest' if not final else 'final'
        checkpoint_path = os.path.join(args.out_dir,f"{name}-model-ckpt.pth")
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        if not final:
            fabric.save(checkpoint_path, state)
        else:
            state['optimizer'] = None
            fabric.save(checkpoint_path, state)
            latest_checkpoint_path = os.path.join(args.out_dir, "latest-model-ckpt.pth")
            fabric.print(f"Updating latest checkpoint to {str(latest_checkpoint_path)!r}")
            fabric.save(latest_checkpoint_path, state)

    for train_data in train_dataloader:
        tokens += model.config.block_size * args.micro_batch_size
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                if fabric.global_rank == 0:
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))

        if state["iter_num"] >= max_iters:
            break

        iter_t0 = time.perf_counter()
        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

        lr = get_lr(args, state["iter_num"], warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        is_accumulating = (state["iter_num"] + 1) % args.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = loss_func(logits, targets)
            fabric.backward(loss / args.gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        if fabric.global_rank == 0 and state["iter_num"] % 10 == 0:
            total_tokens = model.config.block_size * state["iter_num"] * args.micro_batch_size * fabric.world_size / 1e9
            fabric.print(
                    f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                    f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. "
                    f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
                    f" total training throughput {tokens / (t1 - train_t0) / 1e3:.2f}K tokens/s per GPU."
                    f" total trained tokens: {total_tokens} B tokens"
                    f" peak memory allocate {torch.cuda.memory_stats(0)['allocated_bytes.all.peak'] / 1e9} GB"
                )

        estimated_flops = 1
        monitor.on_train_batch_end(
            state["iter_num"] * args.micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

        # Run validation at evenly spaced token milestones, e.g. 1%, 2%, ... of total tokens.
        tokens_per_iter_global = model.config.block_size * args.micro_batch_size * fabric.world_size
        current_tokens = state["iter_num"] * tokens_per_iter_global
        prev_tokens = (state["iter_num"] - 1) * tokens_per_iter_global
        if val_dataloader is not None and args.total_evals > 0:
            completed_eval_milestones = current_tokens * args.total_evals // args.max_tokens
            previous_eval_milestones = prev_tokens * args.total_evals // args.max_tokens
            if completed_eval_milestones > previous_eval_milestones:
                eval_progress = min(completed_eval_milestones, args.total_evals)
                progress_pct = 100 * eval_progress / args.total_evals
                run_validation(f"{progress_pct:.1f}% of training tokens", current_tokens)

        if not is_accumulating and state["step_count"] % args.save_step_interval == 0:
            save_checkpoint()

        if (
            val_dataloader is not None
            and args.total_evals <= 0
            and args.eval_step_interval > 0
            and not is_accumulating
            and state["step_count"] % args.eval_step_interval == 0
        ):
            run_validation(f"optimizer step {state['step_count']}", current_tokens)

    save_checkpoint(final=True)


@torch.no_grad()
def validate(args, fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters=10) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    loss_sums = torch.zeros(args.num_extrapol, device=fabric.device)
    token_counts = torch.zeros(args.num_extrapol, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        for i, length in enumerate([2048, 4096]):
            input_ids = val_data[:, 0 : length].contiguous()
            targets = val_data[:, 1 : length + 1].contiguous()
            logits = model(input_ids)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            valid_mask = targets != -1

            if valid_mask.any():
                per_token_loss = F.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
                loss_sums[i] += per_token_loss[valid_mask].sum()
                token_counts[i] += valid_mask.sum()

    # Aggregate validation numerators and denominators across all ranks, so logged
    # loss/ppl represent global validation statistics instead of rank-local values.
    loss_sums = fabric.all_reduce(loss_sums, reduce_op="sum")
    token_counts = fabric.all_reduce(token_counts, reduce_op="sum")
    out = loss_sums / token_counts.clamp(min=1)

    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        # pdb.set_trace()
        filenames = sorted(glob.glob(os.path.join(data_dir,f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)
        if split != "train":
            n_chunks = - (8 // -nodes)
        else:
            n_chunks = 8
        dataset = PackedDataset(
            filenames,
            n_chunks=n_chunks,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )
    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]
    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size= - (batch_size // -2), # ceil division
            block_size=  16384 + 1, #num_extrapol * block_size + 1, # val 4* extrapolation
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


def get_lr(args, it: int, warmup_iters: int, max_iters: int) -> float:
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > max_iters:
        return args.min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    devices = torch.cuda.device_count() or 1
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='LLM Training')
    group = parser.add_argument_group('hyperparameters')
    group.add_argument('--output_root', default='', type=str, help='output root directory')
    group.add_argument('--wandb_dir', default='', type=str, help='wandb directory')
    group.add_argument('--train_data_dir', default='', type=str, help='training data directory')
    group.add_argument('--train_data_dir_raw', default='', type=str, help='training data directory (raw file for stream tok)')
    group.add_argument('--val_data_dir', default='', type=str, help='validation data directory')
    group.add_argument('--val_data_dir_raw', default='', type=str, help='validation data directory (raw file for stream tok)')
    group.add_argument('--model_name', default='Samba_421M', type=str, help='model name')
    group.add_argument('--qk_norm', default=None, type=str, help='optional qk_norm override')
    group.add_argument('--seq_factor_mode', default=None, type=str, help='optional sequence factor mode override')
    group.add_argument('--gate_mode', default=None, type=str, help='optional gate mode override')
    group.add_argument('--expand_k', default=None, type=float, help='optional key expansion override')
    group.add_argument('--expand_v', default=None, type=float, help='optional value expansion override')
    group.add_argument('--learned_norm_init', default=None, type=float, help='optional learned normalization scalar init')
    group.add_argument('--use_mamba_gate', default=None, type=str2bool, help='optional mamba gate override')
    group.add_argument('--exp_name', default='', type=str, help='experiment name')
    group.add_argument('--exp_group', default='', type=str, help='experiment group name')
    group.add_argument('--train_config', default='tsz512x4k_20B', type=str, help='training config')
    group.add_argument('--resume', action='store_true', default=False, help='resume flag')
    group.add_argument('--debug', action='store_true', default=False, help='debug flag')
    group.add_argument('--interactive_job', action='store_true', default=False, help='debug flag')
    group.add_argument('--tokenizer_name', type=str, default='TinyLlama/TinyLlama_v1.1')
    group.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    group.add_argument('--total_evals', type=int, default=0, help='number of token-progress evals spaced evenly over training; set to 0 to disable')
    group.add_argument('--eval_iters', type=int, default=10, help='number of evaluation iterations')
    group.add_argument('--log_step_interval', type=int, default=10, help='log_step_interval')
    group.add_argument('--save_step_interval', type=int, default=1000, help='save_step_interval')
    group.add_argument('--eval_step_interval', type=int, default=1000, help='eval_step_interval')
    group.add_argument('--seed', type=int, default=3407, help='seed')
    group.add_argument('--num_extrapol', type=int, default=2, help='num_extrapol')
    group.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
    group.add_argument('--beta1', type=float, default=0.9, help='beta1')
    group.add_argument('--beta2', type=float, default=0.95, help='beta2')
    group.add_argument('--grad_clip', type=float, default=1.0, help='gradient clip')
    group.add_argument('--eval_before_training', action='store_true', default=False, help='do validation before the training starts')
    group.add_argument('--nnodes', type=int, default=None, help='number of nodes')
    group.add_argument('--train_num_workers', type=int, default=8)
    group.add_argument('--val_num_workers', type=int, default=1)
    group.add_argument('--micro_batch_size', type=int, default=8, help='micro batch size')
    group.add_argument('--max_tokens', type=int, default=None, help='max tokens')

    args = parser.parse_args()
    name = args.train_config +"_" + args.exp_name
    args.out_dir = args.output_root + '/outputs/' + name
    args.wandb_dir = args.output_root + '/wandb/' + name

    train_data_config = [("train_slim", 1.0)]
    val_data_config = [("validation", 1.0)]
    _nnodes_env = os.getenv("SLURM_NNODES")
    nodes = int(_nnodes_env) if _nnodes_env is not None else 1
    args.nodes = nodes

    recommended_micro_batch_size = 8

    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    elif "20B" in name:
        max_tokens = int(1e11) // 5
    elif "100B" in name:
        max_tokens = int(1e11)
    elif "50B" in name:
        max_tokens = int(1e11) // 2
    elif "30B" in name:
        max_tokens = int(3e10)
    elif "15B" in name:
        max_tokens = int(3e10) // 2
    else:
        raise ValueError("Unknown training token config")

    if "512x4k" in name:
        recommended_micro_batch_size = 8
        global_batch_size = 512 // nodes
    elif "1024x4k" in name:
        recommended_micro_batch_size = 8
        global_batch_size = 1024 // nodes

    elif "256x8k" in name:
        global_batch_size = 256 // nodes
        recommended_micro_batch_size = 8

    elif "128x16k" in name:
        global_batch_size = 128 // nodes
        recommended_micro_batch_size = 4

    elif "64x32k" in name:
        global_batch_size = 64 // nodes
        recommended_micro_batch_size = 2

    elif "1024x2k" in name:
        global_batch_size = 1024 // nodes
        recommended_micro_batch_size = 32

    if "1.3B" in name:
        recommended_micro_batch_size = 4

    # CLI override takes precedence; use recommended value only when caller passes <= 0.
    if args.micro_batch_size <= 0:
        args.micro_batch_size = recommended_micro_batch_size
    args.micro_batch_size = max(1, int(args.micro_batch_size))
    args.min_lr = args.learning_rate / 10
    args.batch_size = global_batch_size // devices

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    assert gradient_accumulation_steps > 0
    log_iter_interval = args.log_step_interval * gradient_accumulation_steps
    args.gradient_accumulation_steps = gradient_accumulation_steps

    args.warmup_tokens = int(max_tokens * 0.01)
    args.max_tokens = max_tokens
    if args.batch_size % args.micro_batch_size != 0:
        print(
            "Warning: per-device batch size is not divisible by micro batch size; "
            "effective global batch may differ slightly from target."
        )
    args.log_iter_interval= log_iter_interval
    hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    args.hparams = hparams
    main(args)
