import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import pandas as pd


METRIC_KEY = "metric/val_ppl@1x"


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _find_latest_summary(model_wandb_root: str):
    pattern = os.path.join(model_wandb_root, "wandb", "run-*", "files", "wandb-summary.json")
    summary_paths = glob.glob(pattern)
    if not summary_paths:
        return None

    best_path = None
    best_score = None
    for path in summary_paths:
        data = _load_json(path)
        if not isinstance(data, dict) or METRIC_KEY not in data:
            continue
        timestamp = data.get("_timestamp", 0.0)
        mtime = os.path.getmtime(path)
        score = (float(timestamp), float(mtime))
        if best_score is None or score > best_score:
            best_score = score
            best_path = path
    return best_path


def _parse_manifest(path: str):
    if not path or not os.path.isfile(path):
        return {}

    manifest = {}
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            variant = row.get("variant")
            if not variant:
                continue
            manifest[variant] = {
                "group": row.get("group", "unknown"),
                "label": row.get("label", variant),
            }
    return manifest


def _infer_variant(name: str, known_variants):
    if not name:
        return None
    for variant in sorted(known_variants, key=len, reverse=True):
        if f"ab47_{variant}" in name or name.endswith(variant):
            return variant
    return None


def _parse_factor_key(value):
    match = re.search(r"(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1))


def load_mqar_results(mqar_out_root: str, known_variants):
    rows = []
    files = glob.glob(os.path.join(mqar_out_root, "**", "results.json"), recursive=True)

    for path in files:
        data = _load_json(path)
        if not isinstance(data, dict):
            continue

        args = data.get("args", {})
        exp_name = args.get("exp_name") or os.path.basename(os.path.dirname(path))
        variant = _infer_variant(exp_name, known_variants)
        if not variant:
            continue

        test_acc = _safe_float(data.get("test_acc"))
        test_acc = test_acc * 100.0 if test_acc is not None else None

        final_val_acc = data.get("final_val_acc", {})
        val_8x = None
        max_factor = None
        max_factor_acc = None
        if isinstance(final_val_acc, dict):
            for key, acc in final_val_acc.items():
                factor = _parse_factor_key(key)
                acc_value = _safe_float(acc)
                if factor is None or acc_value is None:
                    continue
                acc_value *= 100.0
                if factor == 8:
                    val_8x = acc_value
                if max_factor is None or factor > max_factor:
                    max_factor = factor
                    max_factor_acc = acc_value
        if val_8x is None:
            val_8x = max_factor_acc

        rows.append(
            {
                "variant": variant,
                "mqar_test_acc": test_acc,
                "mqar_val_acc_extrap": val_8x,
                "mqar_result_path": path,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("variant", as_index=False)
        .agg(
            mqar_test_acc=("mqar_test_acc", "mean"),
            mqar_val_acc_extrap=("mqar_val_acc_extrap", "mean"),
            mqar_runs=("mqar_result_path", "count"),
        )
    )
    return agg


def load_pretrain_results(pretrain_wandb_root: str, known_variants):
    rows = []
    model_roots = glob.glob(os.path.join(pretrain_wandb_root, "tsz512x4k_ab47_*"))

    for model_root in model_roots:
        run_name = os.path.basename(model_root)
        variant = _infer_variant(run_name, known_variants)
        if not variant:
            continue

        summary_path = _find_latest_summary(model_root)
        if summary_path is None:
            continue

        summary = _load_json(summary_path)
        if not isinstance(summary, dict):
            continue

        val_ppl = _safe_float(summary.get(METRIC_KEY))
        if val_ppl is None:
            continue

        rows.append(
            {
                "variant": variant,
                "pretrain_val_ppl": val_ppl,
                "pretrain_summary_path": summary_path,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("variant", as_index=False)
        .agg(
            pretrain_val_ppl=("pretrain_val_ppl", "mean"),
            pretrain_runs=("pretrain_summary_path", "count"),
        )
    )
    return agg


def format_metric(value, precision=4):
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{precision}f}"


def format_count(value):
    if value is None or pd.isna(value):
        return "0"
    try:
        return str(int(value))
    except Exception:
        return "0"


def markdown_table(rows, headers):
    if not rows:
        return "No results found."

    sep = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(row) + " |")
    return "\n".join([sep, divider] + body)


def build_report(merged_df: pd.DataFrame, save_path: str):
    group_titles = {
        "ab1_norm": "Ablation 1: Normalization Strategy",
        "ab2_seq_factor": "Ablation 2: Sequence-Length Factor",
        "ab3_gate": "Ablation 3: Gating Mechanism",
        "ab4_state": "Ablation 4: State Expansion",
    }

    lines = []
    lines.append("# 4.7 Ablation Experiment Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for group_key in ["ab1_norm", "ab2_seq_factor", "ab3_gate", "ab4_state"]:
        lines.append(f"## {group_titles[group_key]}")
        lines.append("")

        subset = merged_df[merged_df["group"] == group_key].copy()
        subset = subset.sort_values("variant")

        rows = []
        for _, row in subset.iterrows():
            rows.append(
                [
                    str(row.get("label", row["variant"])),
                    format_metric(row.get("mqar_test_acc"), precision=4),
                    format_metric(row.get("mqar_val_acc_extrap"), precision=4),
                    format_metric(row.get("pretrain_val_ppl"), precision=4),
                    format_count(row.get("mqar_runs", 0)),
                    format_count(row.get("pretrain_runs", 0)),
                ]
            )

        lines.append(
            markdown_table(
                rows,
                [
                    "Variant",
                    "MQAR Test Acc (%)",
                    "MQAR Extrap Acc (prefer 8x, %) ",
                    "Pretrain Val PPL@1x",
                    "MQAR Runs",
                    "Pretrain Runs",
                ],
            )
        )
        lines.append("")

    missing = merged_df[
        merged_df["mqar_test_acc"].isna() | merged_df["pretrain_val_ppl"].isna()
    ][["variant", "label", "group", "mqar_test_acc", "pretrain_val_ppl"]]

    lines.append("## Missing Results")
    lines.append("")
    if missing.empty:
        lines.append("All variants have both MQAR and pretrain metrics.")
    else:
        for _, row in missing.iterrows():
            missing_parts = []
            if pd.isna(row["mqar_test_acc"]):
                missing_parts.append("MQAR")
            if pd.isna(row["pretrain_val_ppl"]):
                missing_parts.append("Pretrain")
            lines.append(
                f"- {row['label']} ({row['variant']}, {row['group']}): missing {', '.join(missing_parts)}"
            )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- MQAR extrapolation列优先取8x；若8x缺失则回退到最大可用外推因子。")
    lines.append("- Pretrain 指标来自 wandb-summary.json 的 metric/val_ppl@1x。")

    with open(save_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mqar_out_root", type=str, default="./out/ablation_47/mqar")
    parser.add_argument("--pretrain_wandb_root", type=str, default="../save_dir/wandb")
    parser.add_argument("--save_dir", type=str, default="./analysis_results/ablation_47")
    parser.add_argument("--manifest", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tables_dir = os.path.join(args.save_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    manifest = _parse_manifest(args.manifest) if args.manifest else {}

    if manifest:
        known_variants = sorted(manifest.keys())
    else:
        # Fallback variant registry.
        known_variants = [
            "a1_kla",
            "a1_no_norm",
            "a1_k_norm_only",
            "a1_seq_only",
            "a1_learned_norm",
            "a2_no_seq_factor",
            "a2_inv_t",
            "a2_inv_sqrt_t",
            "a2_inv_log_t",
            "a3_kla_single_gate",
            "a3_gdn_dual_gate",
            "a3_gla_independent_gate",
            "a4_expand_v2",
            "a4_expand_v4",
            "a4_expand_v8",
            "a4_expand_v16",
        ]
        manifest = {
            variant: {
                "group": "unknown",
                "label": variant,
            }
            for variant in known_variants
        }

    mqar_df = load_mqar_results(args.mqar_out_root, known_variants)
    pretrain_df = load_pretrain_results(args.pretrain_wandb_root, known_variants)

    base_rows = []
    for variant in known_variants:
        base_rows.append(
            {
                "variant": variant,
                "group": manifest.get(variant, {}).get("group", "unknown"),
                "label": manifest.get(variant, {}).get("label", variant),
            }
        )
    merged = pd.DataFrame(base_rows)

    if not mqar_df.empty:
        merged = merged.merge(mqar_df, on="variant", how="left")
    else:
        merged["mqar_test_acc"] = None
        merged["mqar_val_acc_extrap"] = None
        merged["mqar_runs"] = 0

    if not pretrain_df.empty:
        merged = merged.merge(pretrain_df, on="variant", how="left")
    else:
        merged["pretrain_val_ppl"] = None
        merged["pretrain_runs"] = 0

    merged_csv = os.path.join(tables_dir, "ablation_47_merged.csv")
    merged.to_csv(merged_csv, index=False)

    if not mqar_df.empty:
        mqar_df.to_csv(os.path.join(tables_dir, "ablation_47_mqar.csv"), index=False)
    if not pretrain_df.empty:
        pretrain_df.to_csv(os.path.join(tables_dir, "ablation_47_pretrain.csv"), index=False)

    report_path = os.path.join(args.save_dir, "ablation_47_report.md")
    build_report(merged, report_path)

    print(f"Saved merged CSV: {merged_csv}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
