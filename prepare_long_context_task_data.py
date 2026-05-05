import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: datasets. Install with `pip install datasets`."
    ) from exc


DEFAULT_RULER_CONFIGS = "niah_single_1_4k,niah_multikey_1_4k,niah_multiquery_4k,niah_multivalue_4k,cwe_4k,fwe_4k"


def _parse_csv(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _to_answers(raw: Any) -> List[str]:
    if isinstance(raw, list):
        out = [str(item).strip() for item in raw if str(item).strip()]
        return out
    if raw is None:
        return []
    text = str(raw).strip()
    return [text] if text else []


def _build_longbench_prompt(sample: Dict[str, Any]) -> str:
    context = str(sample.get("context", "")).strip()
    question = str(sample.get("question", "")).strip()
    option_a = str(sample.get("choice_A", "")).strip()
    option_b = str(sample.get("choice_B", "")).strip()
    option_c = str(sample.get("choice_C", "")).strip()
    option_d = str(sample.get("choice_D", "")).strip()

    return (
        "You are given a long context and a multiple-choice question.\n"
        "Read the context carefully and answer with only one capital letter (A, B, C, or D).\n\n"
        "[Context]\n"
        f"{context}\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Options]\n"
        f"A. {option_a}\n"
        f"B. {option_b}\n"
        f"C. {option_c}\n"
        f"D. {option_d}\n\n"
        "Answer:"
    )


def _prepare_ruler_selflong(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset_id = args.hf_dataset or "self-long/RULER-llama3-1M"
    split = args.hf_split or "validation"
    configs = _parse_csv(args.ruler_configs)
    if not configs:
        raise ValueError("ruler_selflong requires --ruler_configs")

    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        print(f"[RULER] loading {dataset_id} config={cfg} split={split}")
        dataset = load_dataset(dataset_id, cfg, split=split, cache_dir=args.hf_cache_dir or None)

        for idx, sample in enumerate(dataset):
            prompt = str(sample.get("input", "")).strip()
            answers = _to_answers(sample.get("answers"))
            if not prompt or not answers:
                continue

            task_prefix = cfg.split("_", maxsplit=1)[0].upper()
            sample_index = sample.get("index", idx)

            row: Dict[str, Any] = {
                "id": f"{cfg}_{split}_{sample_index}",
                "benchmark": "RULER",
                "task": f"RULER-{task_prefix}",
                "subtask": cfg,
                "prompt": prompt,
                "answers": answers,
                "source_dataset": dataset_id,
                "source_config": cfg,
                "source_split": split,
            }

            if "length" in sample:
                try:
                    row["context_length"] = int(sample["length"])
                except (TypeError, ValueError):
                    pass

            rows.append(row)

    return rows


def _prepare_longbench_v2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset_id = args.hf_dataset or "zai-org/LongBench-v2"
    config = args.hf_config or "default"
    split = args.hf_split or "train"

    keep_lengths = set(_parse_csv(args.longbench_lengths))
    keep_domains = set(_parse_csv(args.longbench_domains))

    print(f"[LongBench-v2] loading {dataset_id} config={config} split={split}")
    dataset = load_dataset(dataset_id, config, split=split, cache_dir=args.hf_cache_dir or None)

    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(dataset):
        length_tag = str(sample.get("length", "")).strip()
        domain = str(sample.get("domain", "")).strip()
        sub_domain = str(sample.get("sub_domain", "")).strip()

        if keep_lengths and length_tag not in keep_lengths:
            continue
        if keep_domains and domain not in keep_domains:
            continue

        answer_letter = str(sample.get("answer", "")).strip().upper()
        options = {
            "A": str(sample.get("choice_A", "")).strip(),
            "B": str(sample.get("choice_B", "")).strip(),
            "C": str(sample.get("choice_C", "")).strip(),
            "D": str(sample.get("choice_D", "")).strip(),
        }
        if answer_letter not in options:
            continue

        prompt = _build_longbench_prompt(sample)
        answers = [answer_letter]
        if args.longbench_include_choice_text and options[answer_letter]:
            answers.append(options[answer_letter])

        row = {
            "id": str(sample.get("_id", f"longbench_{idx}")),
            "benchmark": "LongBench-v2",
            "task": f"LongBench-v2/{domain}" if domain else "LongBench-v2",
            "subtask": sub_domain,
            "prompt": prompt,
            "answers": answers,
            "stop_strings": ["\n"],
            "source_dataset": dataset_id,
            "source_config": config,
            "source_split": split,
            "length_tag": length_tag,
        }

        rows.append(row)

    return rows


def _prepare_mrcr_openai(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset_id = args.hf_dataset or "openai/mrcr"
    config = args.hf_config or "default"
    split = args.hf_split or "train"

    print(f"[MRCR] loading {dataset_id} config={config} split={split}")
    dataset = load_dataset(dataset_id, config, split=split, cache_dir=args.hf_cache_dir or None)

    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(dataset):
        n_needles = int(sample.get("n_needles", 0))
        if args.mrcr_needles > 0 and n_needles != args.mrcr_needles:
            continue

        prompt = str(sample.get("prompt", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        if not prompt or not answer:
            continue

        row = {
            "id": f"mrcr_{split}_{idx}",
            "benchmark": "MRCR",
            "task": f"MRCR-{n_needles}needle",
            "subtask": f"messages_{sample.get('total_messages', 'unknown')}",
            "prompt": prompt,
            "answers": [answer],
            "max_new_tokens": int(args.mrcr_max_new_tokens),
            "source_dataset": dataset_id,
            "source_config": config,
            "source_split": split,
            "n_needles": n_needles,
            "n_chars": int(sample.get("n_chars", 0)),
        }

        rows.append(row)

    return rows


def _select_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.preset == "ruler_selflong":
        rows = _prepare_ruler_selflong(args)
    elif args.preset == "longbench_v2":
        rows = _prepare_longbench_v2(args)
    elif args.preset == "mrcr_openai":
        rows = _prepare_mrcr_openai(args)
    else:
        raise ValueError(f"Unsupported preset: {args.preset}")

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_meta(path: str, args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    task_counter = Counter(str(row.get("task", "unknown")) for row in rows)
    subtask_counter = Counter(str(row.get("subtask", "unknown")) for row in rows)
    avg_prompt_chars = sum(len(str(row.get("prompt", ""))) for row in rows) / len(rows)

    meta = {
        "preset": args.preset,
        "output_jsonl": path,
        "num_rows": len(rows),
        "seed": int(args.seed),
        "shuffle": bool(args.shuffle),
        "max_samples": int(args.max_samples),
        "task_counts": dict(sorted(task_counter.items())),
        "subtask_counts": dict(sorted(subtask_counter.items())),
        "avg_prompt_chars": float(avg_prompt_chars),
        "source_dataset_override": args.hf_dataset,
        "source_config_override": args.hf_config,
        "source_split_override": args.hf_split,
        "ruler_configs": _parse_csv(args.ruler_configs),
        "longbench_lengths": _parse_csv(args.longbench_lengths),
        "longbench_domains": _parse_csv(args.longbench_domains),
        "mrcr_needles": int(args.mrcr_needles),
        "mrcr_max_new_tokens": int(args.mrcr_max_new_tokens),
    }

    meta_path = f"{path}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata: {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare real long-context benchmark datasets into benchmark JSONL format"
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        choices=["ruler_selflong", "longbench_v2", "mrcr_openai"],
    )
    parser.add_argument("--output_jsonl", type=str, required=True)

    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--hf_dataset", type=str, default="")
    parser.add_argument("--hf_config", type=str, default="")
    parser.add_argument("--hf_split", type=str, default="")
    parser.add_argument("--hf_cache_dir", type=str, default="")

    parser.add_argument("--ruler_configs", type=str, default=DEFAULT_RULER_CONFIGS)

    parser.add_argument(
        "--longbench_lengths",
        type=str,
        default="long",
        help="Comma-separated values from short,medium,long. Empty means keep all.",
    )
    parser.add_argument(
        "--longbench_domains",
        type=str,
        default="",
        help="Optional comma-separated domain filter.",
    )
    parser.add_argument(
        "--longbench_include_choice_text",
        action="store_true",
        help="Also accept the choice text as a valid answer (in addition to A/B/C/D).",
    )

    parser.add_argument(
        "--mrcr_needles",
        type=int,
        default=2,
        help="Keep only this number of needles. 0 keeps all.",
    )
    parser.add_argument("--mrcr_max_new_tokens", type=int, default=256)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rows = _select_rows(args)
    if not rows:
        raise RuntimeError("No rows generated. Relax filters or check dataset availability.")

    _write_jsonl(args.output_jsonl, rows)
    _write_meta(args.output_jsonl, args, rows)

    print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
