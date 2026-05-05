import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

# Support running without installing as a package.
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import Config, GPT


def _parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return []
    return [int(item) for item in items]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def _ensure_parent(path: str) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _sync(device)


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message


def _set_model_sequence_limit(model: GPT, target_seq_len: int) -> None:
    # Keep the model cache sized to the currently tested length to avoid
    # pre-allocating very large rope/mask caches for the full sweep upfront.
    seq_len = max(1, int(target_seq_len))
    model.config.block_size = seq_len
    model.max_len = seq_len
    model.rope_cache = None
    model.mask_cache = None
    model.kv_caches.clear()


def _bytes_to_mib(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _timed_benchmark(
    fn,
    warmup_iters: int,
    benchmark_iters: int,
    device: torch.device,
) -> Dict[str, float]:
    for _ in range(max(0, warmup_iters)):
        fn()
    _sync(device)

    elapsed_ms: List[float] = []
    peak_alloc_mib: List[float] = []
    peak_reserved_mib: List[float] = []

    for _ in range(max(1, benchmark_iters)):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)

        t0 = time.perf_counter()
        fn()
        _sync(device)
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000.0
        elapsed_ms.append(ms)

        if device.type == "cuda":
            peak_alloc_mib.append(_bytes_to_mib(torch.cuda.max_memory_allocated(device)))
            peak_reserved_mib.append(_bytes_to_mib(torch.cuda.max_memory_reserved(device)))
        else:
            peak_alloc_mib.append(0.0)
            peak_reserved_mib.append(0.0)

    mean_ms = statistics.fmean(elapsed_ms)
    p50_ms = statistics.median(elapsed_ms)
    std_ms = statistics.pstdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0
    sorted_ms = sorted(elapsed_ms)
    p90_idx = min(len(sorted_ms) - 1, max(0, int(math.ceil(0.9 * len(sorted_ms)) - 1)))
    p90_ms = sorted_ms[p90_idx]

    return {
        "latency_ms_mean": float(mean_ms),
        "latency_ms_p50": float(p50_ms),
        "latency_ms_p90": float(p90_ms),
        "latency_ms_std": float(std_ms),
        "latency_ms_min": float(min(elapsed_ms)),
        "latency_ms_max": float(max(elapsed_ms)),
        "peak_allocated_mib": float(max(peak_alloc_mib) if peak_alloc_mib else 0.0),
        "peak_reserved_mib": float(max(peak_reserved_mib) if peak_reserved_mib else 0.0),
    }


def _load_model_state(model: GPT, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    stripped = {}
    any_module_prefix = False
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module.") :]] = value
            any_module_prefix = True
        else:
            stripped[key] = value

    if not any_module_prefix:
        raise

    model.load_state_dict(stripped, strict=True)


def _prefill_once(model: GPT, prompt: torch.Tensor) -> None:
    model.reset_cache()
    _ = model(prompt)


def _decode_full_context_once(
    model: GPT,
    prompt: torch.Tensor,
    new_tokens: int,
    max_seq_length: int,
) -> None:
    model.reset_cache()

    seq = prompt
    for _ in range(new_tokens):
        logits = model(seq[:, -max_seq_length:])
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)

    model.reset_cache()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark prefill/decode latency and throughput")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint (*.pth)")
    parser.add_argument("--config_name", type=str, required=True, help="Model config name")
    parser.add_argument("--model_alias", type=str, default="", help="Optional alias for output metadata")

    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--prefill_lengths", type=str, default="1024,2048,4096,8192,16384")
    parser.add_argument("--decode_lengths", type=str, default="4096,8192,16384")
    parser.add_argument("--prefill_batch_size", type=int, default=1)
    parser.add_argument("--decode_batch_sizes", type=str, default="1,2,4")
    parser.add_argument("--decode_new_tokens", type=int, default=128)

    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--benchmark_iters", type=int, default=10)

    parser.add_argument(
        "--max_prefill_latency_ms",
        type=float,
        default=0.0,
        help="If > 0, stop testing longer prefill lengths after mean latency exceeds this threshold",
    )
    parser.add_argument(
        "--max_decode_latency_ms",
        type=float,
        default=0.0,
        help="If > 0, stop testing longer decode lengths after mean latency exceeds this threshold",
    )

    parser.add_argument("--save_json", type=str, default="", help="Output JSON path")
    parser.add_argument("--save_csv", type=str, default="", help="Output flattened CSV path")
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    prefill_lengths = sorted(set(_parse_int_list(args.prefill_lengths)))
    decode_lengths = sorted(set(_parse_int_list(args.decode_lengths)))
    decode_batch_sizes = sorted(set(_parse_int_list(args.decode_batch_sizes)))

    if not prefill_lengths:
        raise ValueError("--prefill_lengths cannot be empty")
    if not decode_lengths:
        raise ValueError("--decode_lengths cannot be empty")
    if not decode_batch_sizes:
        raise ValueError("--decode_batch_sizes cannot be empty")
    if args.decode_new_tokens <= 0:
        raise ValueError("--decode_new_tokens must be > 0")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    _set_seed(args.seed)

    config = Config.from_name(args.config_name)
    max_required_len = max(max(prefill_lengths), max(decode_lengths))
    if max_required_len > config.block_size:
        print(
            f"Requested max length {max_required_len} exceeds config.block_size={config.block_size}. "
            "Will run with per-length dynamic cache sizing."
        )

    print("=" * 72)
    print(f"Model config   : {args.config_name}")
    print(f"Checkpoint     : {args.ckpt_path}")
    print(f"Device / dtype : {device} / {args.dtype}")
    print(f"Prefill lengths: {prefill_lengths}")
    print(f"Decode lengths : {decode_lengths}")
    print(f"Decode batches : {decode_batch_sizes}")
    print(f"Decode tokens  : {args.decode_new_tokens}")
    if args.max_prefill_latency_ms > 0:
        print(f"Prefill limit  : stop when mean latency > {args.max_prefill_latency_ms:.2f} ms")
    if args.max_decode_latency_ms > 0:
        print(f"Decode limit   : stop when mean latency > {args.max_decode_latency_ms:.2f} ms")
    print("=" * 72)

    model = GPT(config)
    _load_model_state(model, args.ckpt_path)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    vocab_size = int(config.padded_vocab_size or config.vocab_size)
    prefill_results: List[Dict[str, Any]] = []
    decode_results: List[Dict[str, Any]] = []
    interrupted = False

    try:
        with torch.inference_mode():
            for context_len in prefill_lengths:
                _set_model_sequence_limit(model, context_len)
                prompt = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=(args.prefill_batch_size, context_len),
                    dtype=torch.long,
                    device=device,
                )

                try:
                    stats = _timed_benchmark(
                        lambda: _prefill_once(model, prompt),
                        warmup_iters=args.warmup_iters,
                        benchmark_iters=args.benchmark_iters,
                        device=device,
                    )
                except RuntimeError as exc:
                    if _is_oom_error(exc):
                        print(
                            f"[Prefill] L={context_len:>6}, B={args.prefill_batch_size:>2}, "
                            "OOM encountered; stop remaining longer prefill lengths"
                        )
                        _clear_cuda_cache(device)
                        break
                    raise

                total_tokens = args.prefill_batch_size * context_len
                tokens_per_sec = total_tokens / (stats["latency_ms_mean"] / 1000.0)

                row = {
                    "context_length": int(context_len),
                    "batch_size": int(args.prefill_batch_size),
                    "num_tokens": int(total_tokens),
                    "tokens_per_sec": float(tokens_per_sec),
                    **stats,
                }
                prefill_results.append(row)
                print(
                    f"[Prefill] L={context_len:>6}, B={args.prefill_batch_size:>2}, "
                    f"lat={stats['latency_ms_mean']:.2f} ms, tok/s={tokens_per_sec:.2f}"
                )
                if args.max_prefill_latency_ms > 0 and stats["latency_ms_mean"] > args.max_prefill_latency_ms:
                    print(
                        f"[Prefill] Reached max_prefill_latency_ms={args.max_prefill_latency_ms:.2f}; "
                        "stop remaining longer prefill lengths"
                    )
                    break

            for context_len in decode_lengths:
                _set_model_sequence_limit(model, context_len)
                should_stop_decode = False
                for batch_size in decode_batch_sizes:
                    prompt = torch.randint(
                        low=0,
                        high=vocab_size,
                        size=(batch_size, context_len),
                        dtype=torch.long,
                        device=device,
                    )
                    # Benchmark decode at a fixed context window size.
                    max_seq_length = context_len
                    decode_fn = lambda: _decode_full_context_once(
                        model,
                        prompt,
                        new_tokens=args.decode_new_tokens,
                        max_seq_length=max_seq_length,
                    )

                    try:
                        stats = _timed_benchmark(
                            decode_fn,
                            warmup_iters=args.warmup_iters,
                            benchmark_iters=args.benchmark_iters,
                            device=device,
                        )
                    except RuntimeError as exc:
                        if _is_oom_error(exc):
                            print(
                                f"[Decode] L={context_len:>6}, B={batch_size:>2}, "
                                "OOM encountered; stop remaining longer decode lengths"
                            )
                            _clear_cuda_cache(device)
                            should_stop_decode = True
                            break
                        raise

                    total_decode_tokens = batch_size * args.decode_new_tokens
                    tokens_per_sec = total_decode_tokens / (stats["latency_ms_mean"] / 1000.0)
                    latency_ms_per_token = stats["latency_ms_mean"] / float(args.decode_new_tokens)

                    row = {
                        "context_length": int(context_len),
                        "batch_size": int(batch_size),
                        "new_tokens": int(args.decode_new_tokens),
                        "total_decode_tokens": int(total_decode_tokens),
                        "tokens_per_sec": float(tokens_per_sec),
                        "latency_ms_per_token": float(latency_ms_per_token),
                        **stats,
                    }
                    decode_results.append(row)
                    print(
                        f"[Decode] L={context_len:>6}, B={batch_size:>2}, "
                        f"lat={stats['latency_ms_mean']:.2f} ms, "
                        f"tpot={latency_ms_per_token:.2f} ms, tok/s={tokens_per_sec:.2f}"
                    )
                    if args.max_decode_latency_ms > 0 and stats["latency_ms_mean"] > args.max_decode_latency_ms:
                        print(
                            f"[Decode] Reached max_decode_latency_ms={args.max_decode_latency_ms:.2f} at "
                            f"L={context_len}, B={batch_size}; stop remaining longer decode lengths"
                        )
                        should_stop_decode = True
                        break
                if should_stop_decode:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("Benchmark interrupted by user; saving partial results.")

    payload = {
        "model_name": args.model_alias or args.config_name,
        "config_name": args.config_name,
        "ckpt_path": args.ckpt_path,
        "device": str(device),
        "dtype": args.dtype,
        "seed": int(args.seed),
        "warmup_iters": int(args.warmup_iters),
        "benchmark_iters": int(args.benchmark_iters),
        "max_prefill_latency_ms": float(args.max_prefill_latency_ms),
        "max_decode_latency_ms": float(args.max_decode_latency_ms),
        "prefill_lengths": prefill_lengths,
        "decode_lengths": decode_lengths,
        "prefill_batch_size": int(args.prefill_batch_size),
        "decode_batch_sizes": decode_batch_sizes,
        "decode_new_tokens": int(args.decode_new_tokens),
        "interrupted": bool(interrupted),
        "prefill_results": prefill_results,
        "decode_results": decode_results,
    }

    if args.save_json:
        _ensure_parent(args.save_json)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {args.save_json}")

    if args.save_csv:
        _ensure_parent(args.save_csv)
        fieldnames = [
            "phase",
            "model_name",
            "config_name",
            "context_length",
            "batch_size",
            "num_tokens",
            "new_tokens",
            "total_decode_tokens",
            "tokens_per_sec",
            "latency_ms_per_token",
            "latency_ms_mean",
            "latency_ms_p50",
            "latency_ms_p90",
            "latency_ms_std",
            "latency_ms_min",
            "latency_ms_max",
            "peak_allocated_mib",
            "peak_reserved_mib",
        ]

        with open(args.save_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in prefill_results:
                writer.writerow(
                    {
                        "phase": "prefill",
                        "model_name": payload["model_name"],
                        "config_name": payload["config_name"],
                        "context_length": row["context_length"],
                        "batch_size": row["batch_size"],
                        "num_tokens": row["num_tokens"],
                        "new_tokens": "",
                        "total_decode_tokens": "",
                        "tokens_per_sec": row["tokens_per_sec"],
                        "latency_ms_per_token": "",
                        "latency_ms_mean": row["latency_ms_mean"],
                        "latency_ms_p50": row["latency_ms_p50"],
                        "latency_ms_p90": row["latency_ms_p90"],
                        "latency_ms_std": row["latency_ms_std"],
                        "latency_ms_min": row["latency_ms_min"],
                        "latency_ms_max": row["latency_ms_max"],
                        "peak_allocated_mib": row["peak_allocated_mib"],
                        "peak_reserved_mib": row["peak_reserved_mib"],
                    }
                )

            for row in decode_results:
                writer.writerow(
                    {
                        "phase": "decode",
                        "model_name": payload["model_name"],
                        "config_name": payload["config_name"],
                        "context_length": row["context_length"],
                        "batch_size": row["batch_size"],
                        "num_tokens": "",
                        "new_tokens": row["new_tokens"],
                        "total_decode_tokens": row["total_decode_tokens"],
                        "tokens_per_sec": row["tokens_per_sec"],
                        "latency_ms_per_token": row["latency_ms_per_token"],
                        "latency_ms_mean": row["latency_ms_mean"],
                        "latency_ms_p50": row["latency_ms_p50"],
                        "latency_ms_p90": row["latency_ms_p90"],
                        "latency_ms_std": row["latency_ms_std"],
                        "latency_ms_min": row["latency_ms_min"],
                        "latency_ms_max": row["latency_ms_max"],
                        "peak_allocated_mib": row["peak_allocated_mib"],
                        "peak_reserved_mib": row["peak_reserved_mib"],
                    }
                )

        print(f"Saved CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
