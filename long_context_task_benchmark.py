import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Support running without installing as a package.
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import Config, GPT
from lit_gpt.tokenizer import Tokenizer as LitTokenizer

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def _parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def _parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_parent(path: str) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _set_model_sequence_limit(model: GPT, target_seq_len: int) -> None:
    seq_len = max(1, int(target_seq_len))
    model.config.block_size = seq_len
    model.max_len = seq_len
    model.rope_cache = None
    model.mask_cache = None
    model.kv_caches.clear()


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


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _token_f1(prediction: str, target: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    tgt_tokens = _normalize_text(target).split()

    if not pred_tokens and not tgt_tokens:
        return 1.0
    if not pred_tokens or not tgt_tokens:
        return 0.0

    pred_count: Dict[str, int] = {}
    tgt_count: Dict[str, int] = {}
    for tok in pred_tokens:
        pred_count[tok] = pred_count.get(tok, 0) + 1
    for tok in tgt_tokens:
        tgt_count[tok] = tgt_count.get(tok, 0) + 1

    common = 0
    for tok, cnt in pred_count.items():
        if tok in tgt_count:
            common += min(cnt, tgt_count[tok])

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(tgt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _truncate_at_stop_strings(text: str, stop_strings: Sequence[str]) -> str:
    if not stop_strings:
        return text

    cut_idx: Optional[int] = None
    for stop in stop_strings:
        pos = text.find(stop)
        if pos >= 0:
            if cut_idx is None or pos < cut_idx:
                cut_idx = pos

    if cut_idx is None:
        return text
    return text[:cut_idx]


def _resolve_answers(sample: Dict[str, Any]) -> List[str]:
    if "answers" in sample and isinstance(sample["answers"], list):
        return [str(item) for item in sample["answers"]]
    if "answer" in sample:
        return [str(sample["answer"])]
    if "target" in sample:
        return [str(sample["target"])]
    return []


def _prepare_prompt_ids(
    prompt_ids: torch.Tensor,
    max_prompt_tokens: int,
    prompt_truncation: str,
) -> Tuple[torch.Tensor, bool, bool]:
    if max_prompt_tokens <= 0:
        return prompt_ids, False, False

    if prompt_ids.numel() <= max_prompt_tokens:
        return prompt_ids, False, False

    if prompt_truncation == "left":
        return prompt_ids[-max_prompt_tokens:], True, False
    if prompt_truncation == "right":
        return prompt_ids[:max_prompt_tokens], True, False

    # prompt_truncation == "none"
    return prompt_ids, False, True


def _build_bucket_edges(value: str) -> List[int]:
    edges = sorted(set(_parse_int_list(value)))
    return [edge for edge in edges if edge > 0]


def _bucket_label(length: int, edges: Sequence[int]) -> str:
    if not edges:
        return "all"

    lower = 1
    for edge in edges:
        if length <= edge:
            return f"{lower}-{edge}"
        lower = edge + 1
    return f">{edges[-1]}"


class TextTokenizer:
    def __init__(self, tokenizer_name: str, tokenizer_dir: str = "") -> None:
        self.backend = ""
        self.hf = None
        self.lit = None
        self.eos_id: Optional[int] = None

        if tokenizer_dir:
            candidate = Path(tokenizer_dir)
            if (candidate / "tokenizer.model").is_file() or (candidate / "tokenizer.json").is_file():
                self.lit = LitTokenizer(candidate)
                self.backend = "lit"
                self.eos_id = self.lit.eos_id
                return

        if AutoTokenizer is None:
            raise ImportError(
                "transformers is required when tokenizer_dir does not contain tokenizer.model/tokenizer.json"
            )

        source = tokenizer_dir or tokenizer_name
        self.hf = AutoTokenizer.from_pretrained(source, use_fast=True)
        self.backend = "hf"
        self.eos_id = self.hf.eos_token_id

    def encode(self, text: str) -> torch.Tensor:
        if self.backend == "lit":
            return self.lit.encode(text, eos=False).to(dtype=torch.long)

        encoded = self.hf(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )
        return encoded["input_ids"][0].to(dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        if len(token_ids) == 0:
            return ""

        if self.backend == "lit":
            tensor = torch.tensor(token_ids, dtype=torch.long)
            return self.lit.decode(tensor)

        return self.hf.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )


def load_samples(input_jsonl: str, max_samples: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_idx}: {exc}") from exc

            if "prompt" not in sample:
                raise ValueError(f"Missing 'prompt' at line {line_idx}")

            answers = _resolve_answers(sample)
            if not answers:
                raise ValueError(
                    f"Missing answers at line {line_idx}. Provide 'answers' list or 'answer'/'target'."
                )

            sample_id = sample.get("id", f"line_{line_idx}")
            sample["id"] = str(sample_id)
            sample["answers"] = answers
            samples.append(sample)

            if max_samples > 0 and len(samples) >= max_samples:
                break

    return samples


@torch.no_grad()
def greedy_generate(
    model: GPT,
    tokenizer: TextTokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    stop_strings: Sequence[str],
) -> Tuple[str, List[int]]:
    device = next(model.parameters()).device
    seq = prompt_ids.unsqueeze(0).to(device)
    generated_ids: List[int] = []

    for _ in range(max_new_tokens):
        logits = model(seq)
        next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated_ids.append(next_token_id)

        next_token = torch.tensor([[next_token_id]], device=device, dtype=torch.long)
        seq = torch.cat([seq, next_token], dim=1)

        if tokenizer.eos_id is not None and next_token_id == tokenizer.eos_id:
            break

        if stop_strings:
            text_now = tokenizer.decode(generated_ids)
            if any(stop in text_now for stop in stop_strings):
                break

    pred_text = tokenizer.decode(generated_ids)
    pred_text = _truncate_at_stop_strings(pred_text, stop_strings).strip()
    return pred_text, generated_ids


def _best_metrics(prediction: str, answers: Sequence[str]) -> Tuple[float, float, float]:
    norm_pred = _normalize_text(prediction)

    exact = 0.0
    contains = 0.0
    best_f1 = 0.0

    for answer in answers:
        norm_ans = _normalize_text(answer)
        if norm_pred == norm_ans:
            exact = 1.0
        if norm_ans and norm_ans in norm_pred:
            contains = 1.0
        best_f1 = max(best_f1, _token_f1(prediction, answer))

    return exact, contains, best_f1


def _aggregate_rows(rows: Sequence[Dict[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_key, "unknown"))
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups.keys()):
        group_rows = groups[key]
        n = len(group_rows)
        em = sum(float(r["exact_match"]) for r in group_rows) / n if n > 0 else float("nan")
        contains = sum(float(r["contains_match"]) for r in group_rows) / n if n > 0 else float("nan")
        f1 = sum(float(r["f1"]) for r in group_rows) / n if n > 0 else float("nan")
        out.append(
            {
                group_key: key,
                "num_samples": int(n),
                "exact_match": float(em),
                "contains_match": float(contains),
                "f1": float(f1),
            }
        )
    return out


def maybe_save_csv(save_csv: str, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    if not save_csv:
        return

    _ensure_parent(save_csv)

    sample_fields = [
        "sample_id",
        "benchmark",
        "task",
        "subtask",
        "context_bucket",
        "prompt_tokens",
        "max_new_tokens",
        "exact_match",
        "contains_match",
        "f1",
        "prediction",
        "answers",
        "truncated_prompt",
        "skip_reason",
    ]

    with open(save_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.get("sample_id", ""),
                    "benchmark": row.get("benchmark", ""),
                    "task": row.get("task", ""),
                    "subtask": row.get("subtask", ""),
                    "context_bucket": row.get("context_bucket", ""),
                    "prompt_tokens": row.get("prompt_tokens", 0),
                    "max_new_tokens": row.get("max_new_tokens", 0),
                    "exact_match": row.get("exact_match", ""),
                    "contains_match": row.get("contains_match", ""),
                    "f1": row.get("f1", ""),
                    "prediction": row.get("prediction", ""),
                    "answers": " || ".join(row.get("answers", [])),
                    "truncated_prompt": row.get("truncated_prompt", False),
                    "skip_reason": row.get("skip_reason", ""),
                }
            )

    stem, ext = os.path.splitext(save_csv)
    summary_csv = f"{stem}_summary{ext or '.csv'}"

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "name", "num_samples", "exact_match", "contains_match", "f1"])

        overall = summary["overall"]
        writer.writerow(
            [
                "overall",
                "all",
                overall["num_evaluated"],
                overall["exact_match"],
                overall["contains_match"],
                overall["f1"],
            ]
        )

        for item in summary["by_task"]:
            writer.writerow(
                [
                    "task",
                    item["task"],
                    item["num_samples"],
                    item["exact_match"],
                    item["contains_match"],
                    item["f1"],
                ]
            )

        for item in summary["by_context_bucket"]:
            writer.writerow(
                [
                    "context_bucket",
                    item["context_bucket"],
                    item["num_samples"],
                    item["exact_match"],
                    item["contains_match"],
                    item["f1"],
                ]
            )

    print(f"Saved CSV: {save_csv}")
    print(f"Saved Summary CSV: {summary_csv}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-context task benchmark (RULER-style zero-shot)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint (*.pth)")
    parser.add_argument("--config_name", type=str, required=True, help="Model config name")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Benchmark JSONL file")

    parser.add_argument("--benchmark_name", type=str, default="RULER", help="Benchmark name")
    parser.add_argument("--task_name", type=str, default="", help="Optional task name override")
    parser.add_argument("--model_alias", type=str, default="", help="Optional model alias for outputs")

    parser.add_argument("--tokenizer_name", type=str, default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="",
        help="Local tokenizer directory. If it contains tokenizer.model/tokenizer.json, lit_gpt tokenizer is used.",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--max_samples", type=int, default=0, help="If > 0, evaluate only first N samples")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--default_stop_strings",
        type=str,
        default="",
        help="Comma-separated stop strings applied to all samples",
    )

    parser.add_argument(
        "--length_buckets",
        type=str,
        default="4096,8192,16384,32768,65536,131072",
        help="Comma-separated context bucket upper bounds",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=0,
        help="If > 0, apply prompt truncation rules when prompt exceeds this value",
    )
    parser.add_argument(
        "--prompt_truncation",
        type=str,
        default="none",
        choices=["none", "left", "right"],
        help="Prompt truncation strategy when max_prompt_tokens > 0",
    )

    parser.add_argument("--allow_oom_skip", action="store_true", help="Skip samples that OOM instead of failing")
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--save_json", type=str, default="", help="Output JSON path")
    parser.add_argument("--save_csv", type=str, default="", help="Output sample CSV path")
    parser.add_argument("--save_predictions", action="store_true", help="Save prediction text in JSON results")

    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input_jsonl):
        raise FileNotFoundError(f"input_jsonl does not exist: {args.input_jsonl}")

    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("CPU device detected; overriding dtype to float32")
        dtype = torch.float32

    _set_seed(args.seed)

    samples = load_samples(args.input_jsonl, args.max_samples)
    if not samples:
        raise RuntimeError("No samples loaded from input_jsonl")

    stop_strings = _parse_str_list(args.default_stop_strings)
    bucket_edges = _build_bucket_edges(args.length_buckets)

    tokenizer = TextTokenizer(tokenizer_name=args.tokenizer_name, tokenizer_dir=args.tokenizer_dir)

    config = Config.from_name(args.config_name)
    model = GPT(config)
    _load_model_state(model, args.ckpt_path)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    print("=" * 72)
    print(f"Benchmark        : {args.benchmark_name}")
    print(f"Task override    : {args.task_name or '(from dataset)'}")
    print(f"Model config     : {args.config_name}")
    print(f"Checkpoint       : {args.ckpt_path}")
    print(f"Device / dtype   : {device} / {dtype}")
    print(f"Samples          : {len(samples)}")
    print(f"Max new tokens   : {args.max_new_tokens}")
    print(f"Length buckets   : {bucket_edges or ['all']}")
    print("=" * 72)

    started_at = time.time()

    evaluated_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for idx, sample in enumerate(samples, start=1):
            prompt = str(sample["prompt"])
            answers: List[str] = sample["answers"]
            task = str(args.task_name or sample.get("task", "unknown"))
            subtask = str(sample.get("subtask", ""))

            per_sample_stop = list(stop_strings)
            extra_stop = sample.get("stop_strings")
            if isinstance(extra_stop, list):
                per_sample_stop.extend(str(item) for item in extra_stop)

            sample_max_new_tokens = int(sample.get("max_new_tokens", args.max_new_tokens))
            sample_max_new_tokens = max(1, sample_max_new_tokens)

            prompt_ids = tokenizer.encode(prompt)
            prompt_ids, truncated_prompt, should_skip = _prepare_prompt_ids(
                prompt_ids=prompt_ids,
                max_prompt_tokens=args.max_prompt_tokens,
                prompt_truncation=args.prompt_truncation,
            )

            if should_skip:
                skipped_rows.append(
                    {
                        "sample_id": sample["id"],
                        "benchmark": args.benchmark_name,
                        "task": task,
                        "subtask": subtask,
                        "context_bucket": _bucket_label(int(prompt_ids.numel()), bucket_edges),
                        "prompt_tokens": int(prompt_ids.numel()),
                        "max_new_tokens": int(sample_max_new_tokens),
                        "exact_match": float("nan"),
                        "contains_match": float("nan"),
                        "f1": float("nan"),
                        "prediction": "",
                        "answers": answers,
                        "truncated_prompt": bool(truncated_prompt),
                        "skip_reason": "prompt_exceeds_max_prompt_tokens",
                    }
                )
                continue

            prompt_len = int(prompt_ids.numel())
            if prompt_len <= 0:
                skipped_rows.append(
                    {
                        "sample_id": sample["id"],
                        "benchmark": args.benchmark_name,
                        "task": task,
                        "subtask": subtask,
                        "context_bucket": "all",
                        "prompt_tokens": 0,
                        "max_new_tokens": int(sample_max_new_tokens),
                        "exact_match": float("nan"),
                        "contains_match": float("nan"),
                        "f1": float("nan"),
                        "prediction": "",
                        "answers": answers,
                        "truncated_prompt": bool(truncated_prompt),
                        "skip_reason": "empty_prompt_after_tokenization",
                    }
                )
                continue

            bucket = _bucket_label(prompt_len, bucket_edges)

            try:
                _set_model_sequence_limit(model, prompt_len + sample_max_new_tokens)
                prediction, generated_ids = greedy_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    max_new_tokens=sample_max_new_tokens,
                    stop_strings=per_sample_stop,
                )
                model.reset_cache()
            except RuntimeError as exc:
                if args.allow_oom_skip and _is_oom_error(exc):
                    _clear_cuda_cache(device)
                    skipped_rows.append(
                        {
                            "sample_id": sample["id"],
                            "benchmark": args.benchmark_name,
                            "task": task,
                            "subtask": subtask,
                            "context_bucket": bucket,
                            "prompt_tokens": prompt_len,
                            "max_new_tokens": int(sample_max_new_tokens),
                            "exact_match": float("nan"),
                            "contains_match": float("nan"),
                            "f1": float("nan"),
                            "prediction": "",
                            "answers": answers,
                            "truncated_prompt": bool(truncated_prompt),
                            "skip_reason": "oom",
                        }
                    )
                    continue
                raise

            exact, contains, f1 = _best_metrics(prediction, answers)

            row = {
                "sample_id": sample["id"],
                "benchmark": args.benchmark_name,
                "task": task,
                "subtask": subtask,
                "context_bucket": bucket,
                "prompt_tokens": prompt_len,
                "max_new_tokens": int(sample_max_new_tokens),
                "exact_match": float(exact),
                "contains_match": float(contains),
                "f1": float(f1),
                "answers": answers,
                "truncated_prompt": bool(truncated_prompt),
                "skip_reason": "",
            }

            if args.save_predictions:
                row["prediction"] = prediction
                row["generated_token_count"] = len(generated_ids)

            evaluated_rows.append(row)

            if args.log_every > 0 and idx % args.log_every == 0:
                print(
                    f"[{idx}/{len(samples)}] "
                    f"evaluated={len(evaluated_rows)} skipped={len(skipped_rows)}"
                )

    total_rows = evaluated_rows + skipped_rows

    if evaluated_rows:
        overall_em = sum(float(r["exact_match"]) for r in evaluated_rows) / len(evaluated_rows)
        overall_contains = sum(float(r["contains_match"]) for r in evaluated_rows) / len(evaluated_rows)
        overall_f1 = sum(float(r["f1"]) for r in evaluated_rows) / len(evaluated_rows)
    else:
        overall_em = float("nan")
        overall_contains = float("nan")
        overall_f1 = float("nan")

    by_task = _aggregate_rows(evaluated_rows, "task")
    by_bucket = _aggregate_rows(evaluated_rows, "context_bucket")

    elapsed_sec = time.time() - started_at

    summary = {
        "overall": {
            "num_total": int(len(total_rows)),
            "num_evaluated": int(len(evaluated_rows)),
            "num_skipped": int(len(skipped_rows)),
            "exact_match": float(overall_em),
            "contains_match": float(overall_contains),
            "f1": float(overall_f1),
            "elapsed_sec": float(elapsed_sec),
            "samples_per_sec": float(len(evaluated_rows) / elapsed_sec) if elapsed_sec > 0 else float("nan"),
        },
        "by_task": [{"task": item["task"], **{k: v for k, v in item.items() if k != "task"}} for item in by_task],
        "by_context_bucket": [
            {"context_bucket": item["context_bucket"], **{k: v for k, v in item.items() if k != "context_bucket"}}
            for item in by_bucket
        ],
    }

    payload = {
        "benchmark_name": args.benchmark_name,
        "task_name_override": args.task_name,
        "model_name": args.model_alias or args.config_name,
        "config_name": args.config_name,
        "ckpt_path": args.ckpt_path,
        "input_jsonl": args.input_jsonl,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_dir": args.tokenizer_dir,
        "device": str(device),
        "dtype": str(dtype),
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "max_new_tokens": int(args.max_new_tokens),
        "default_stop_strings": stop_strings,
        "length_buckets": bucket_edges,
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "prompt_truncation": args.prompt_truncation,
        "allow_oom_skip": bool(args.allow_oom_skip),
        "save_predictions": bool(args.save_predictions),
        "summary": summary,
        "results": total_rows,
    }

    if args.save_json:
        _ensure_parent(args.save_json)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {args.save_json}")

    if args.save_csv:
        maybe_save_csv(args.save_csv, total_rows, summary)

    print("=" * 72)
    print(
        "Overall: "
        f"evaluated={summary['overall']['num_evaluated']}, "
        f"skipped={summary['overall']['num_skipped']}, "
        f"EM={summary['overall']['exact_match']:.4f}, "
        f"Contains={summary['overall']['contains_match']:.4f}, "
        f"F1={summary['overall']['f1']:.4f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
