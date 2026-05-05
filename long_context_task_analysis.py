import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from analysis_plot_style import (
    MODEL_ORDER,
    apply_publication_style,
    get_tables_dir,
    get_model_style,
    normalize_model_name,
    save_publication_figure,
)


def _parse_csv_keywords(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bucket_sort_key(label: str) -> Tuple[int, int]:
    if not label:
        return (3, 10**9)

    label = str(label)
    if label == "all":
        return (0, -1)
    if label.startswith(">"):
        try:
            return (2, int(label[1:]))
        except ValueError:
            return (2, 10**9)
    if "-" in label:
        parts = label.split("-", maxsplit=1)
        try:
            return (1, int(parts[1]))
        except ValueError:
            return (1, 10**9)
    return (3, 10**9)


def _flatten_result(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Skip invalid JSON {path}: {exc}")
        return [], [], []

    model_raw = data.get("model_name") or data.get("config_name") or "Unknown"
    model = normalize_model_name(str(model_raw))
    benchmark = str(data.get("benchmark_name", "Unknown"))
    summary = data.get("summary", {})
    overall = summary.get("overall", {})

    overall_rows = [
        {
            "Model": model,
            "Model Raw": str(model_raw),
            "Benchmark": benchmark,
            "Task Override": str(data.get("task_name_override", "")),
            "Exact Match": _safe_float(overall.get("exact_match")),
            "Contains Match": _safe_float(overall.get("contains_match")),
            "F1": _safe_float(overall.get("f1")),
            "Num Evaluated": _safe_int(overall.get("num_evaluated")),
            "Num Skipped": _safe_int(overall.get("num_skipped")),
            "Num Total": _safe_int(overall.get("num_total")),
            "Elapsed Sec": _safe_float(overall.get("elapsed_sec")),
            "Path": path,
            "Config Name": str(data.get("config_name", "")),
            "Checkpoint": str(data.get("ckpt_path", "")),
            "Input JSONL": str(data.get("input_jsonl", "")),
        }
    ]

    task_rows = []
    for item in summary.get("by_task", []):
        task_rows.append(
            {
                "Model": model,
                "Benchmark": benchmark,
                "Task": str(item.get("task", "unknown")),
                "Exact Match": _safe_float(item.get("exact_match")),
                "Contains Match": _safe_float(item.get("contains_match")),
                "F1": _safe_float(item.get("f1")),
                "Num Samples": _safe_int(item.get("num_samples")),
                "Path": path,
            }
        )

    bucket_rows = []
    for item in summary.get("by_context_bucket", []):
        bucket_rows.append(
            {
                "Model": model,
                "Benchmark": benchmark,
                "Context Bucket": str(item.get("context_bucket", "unknown")),
                "Exact Match": _safe_float(item.get("exact_match")),
                "Contains Match": _safe_float(item.get("contains_match")),
                "F1": _safe_float(item.get("f1")),
                "Num Samples": _safe_int(item.get("num_samples")),
                "Path": path,
            }
        )

    return overall_rows, task_rows, bucket_rows


def load_results(out_root: str):
    files = glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True)
    overall_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []

    for path in files:
        o_rows, t_rows, b_rows = _flatten_result(path)
        overall_rows.extend(o_rows)
        task_rows.extend(t_rows)
        bucket_rows.extend(b_rows)

    return pd.DataFrame(overall_rows), pd.DataFrame(task_rows), pd.DataFrame(bucket_rows), files


def _aggregate_mean(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    agg_dict = {col: "mean" for col in value_cols if col in df.columns}
    has_path = "Path" in df.columns
    if has_path:
        agg_dict["Path"] = "nunique"

    if not agg_dict:
        return pd.DataFrame(columns=group_cols)

    out = df.groupby(group_cols, as_index=False).agg(agg_dict)
    if has_path and "Path" in out.columns:
        out = out.rename(columns={"Path": "Num Runs"})
    return out


def _resolve_plot_order(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []

    present_models = set(df["Model"].unique())
    order = [model for model in MODEL_ORDER if model in present_models]
    remaining = sorted(present_models - set(order))
    return order + remaining


def _plot_overall_bar(ax, df: pd.DataFrame, metric: str, plot_order: List[str]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
        ax.set_axis_off()
        return

    values = []
    colors = []
    labels = []

    for model in plot_order:
        row = df[df["Model"] == model]
        if row.empty:
            continue
        values.append(float(row.iloc[0][metric]))
        colors.append(get_model_style(model).get("color", "#666666"))
        labels.append(model)

    if not values:
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
        ax.set_axis_off()
        return

    x = list(range(len(values)))
    ax.bar(x, values, color=colors, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.12))
    ax.grid(True, axis="y")


def _plot_bucket_curve(ax, df: pd.DataFrame, metric: str, plot_order: List[str]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No bucket-level metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    bucket_order = sorted(df["Context Bucket"].unique(), key=_bucket_sort_key)

    for model in plot_order:
        model_df = df[df["Model"] == model].copy()
        if model_df.empty:
            continue

        model_df["_bucket_order"] = model_df["Context Bucket"].map(lambda x: bucket_order.index(x))
        model_df = model_df.sort_values("_bucket_order")

        style = get_model_style(model)
        ax.plot(
            model_df["_bucket_order"],
            model_df[metric],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    ax.set_xticks(list(range(len(bucket_order))))
    ax.set_xticklabels(bucket_order, rotation=20, ha="right")
    ax.set_xlabel("Context bucket (prompt tokens)")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.05)
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, handlelength=2.8)


def main() -> None:
    apply_publication_style()

    parser = argparse.ArgumentParser(description="Aggregate long-context task benchmark results")
    parser.add_argument("--out_root", type=str, default="./out/long_context_tasks")
    parser.add_argument("--save_dir", type=str, default="./analysis_results/long_context_tasks")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated model aliases to keep (e.g. KLA,GDN,Mamba2)",
    )
    parser.add_argument("--figure_stem", type=str, default="long_context_tasks")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tables_dir = get_tables_dir(args.save_dir)

    overall_df, task_df, bucket_df, files = load_results(args.out_root)
    if not files:
        print(f"No results.json found under {args.out_root}")
        return

    model_filter = set(_parse_csv_keywords(args.models))
    if model_filter:
        overall_df = overall_df[overall_df["Model"].isin(model_filter)].copy()
        task_df = task_df[task_df["Model"].isin(model_filter)].copy()
        bucket_df = bucket_df[bucket_df["Model"].isin(model_filter)].copy()

    if overall_df.empty:
        print("No rows remain after model filtering.")
        return

    overall_mean = _aggregate_mean(
        overall_df,
        group_cols=["Model", "Benchmark"],
        value_cols=[
            "Exact Match",
            "Contains Match",
            "F1",
            "Num Evaluated",
            "Num Skipped",
            "Num Total",
            "Elapsed Sec",
        ],
    )
    task_mean = _aggregate_mean(
        task_df,
        group_cols=["Model", "Benchmark", "Task"],
        value_cols=["Exact Match", "Contains Match", "F1", "Num Samples"],
    )
    bucket_mean = _aggregate_mean(
        bucket_df,
        group_cols=["Model", "Benchmark", "Context Bucket"],
        value_cols=["Exact Match", "Contains Match", "F1", "Num Samples"],
    )

    overall_raw_csv = os.path.join(tables_dir, "long_context_task_overall_raw.csv")
    task_raw_csv = os.path.join(tables_dir, "long_context_task_task_raw.csv")
    bucket_raw_csv = os.path.join(tables_dir, "long_context_task_bucket_raw.csv")

    overall_mean_csv = os.path.join(tables_dir, "long_context_task_overall_mean.csv")
    task_mean_csv = os.path.join(tables_dir, "long_context_task_task_mean.csv")
    bucket_mean_csv = os.path.join(tables_dir, "long_context_task_bucket_mean.csv")

    overall_df.to_csv(overall_raw_csv, index=False)
    task_df.to_csv(task_raw_csv, index=False)
    bucket_df.to_csv(bucket_raw_csv, index=False)

    overall_mean.to_csv(overall_mean_csv, index=False)
    task_mean.to_csv(task_mean_csv, index=False)
    bucket_mean.to_csv(bucket_mean_csv, index=False)

    plot_order = _resolve_plot_order(overall_mean)

    # If multiple benchmarks are present, aggregate once more by model for high-level figure.
    overall_by_model = _aggregate_mean(
        overall_mean,
        group_cols=["Model"],
        value_cols=["Exact Match", "Contains Match", "F1", "Num Evaluated", "Num Total"],
    )

    fig_overall, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    _plot_overall_bar(axes[0], overall_by_model, "Exact Match", plot_order)
    _plot_overall_bar(axes[1], overall_by_model, "F1", plot_order)
    axes[0].set_title("Overall Exact Match")
    axes[1].set_title("Overall F1")
    overall_png, overall_pdf = save_publication_figure(fig_overall, args.save_dir, f"{args.figure_stem}_overall")
    plt.close(fig_overall)

    bucket_by_model = _aggregate_mean(
        bucket_mean,
        group_cols=["Model", "Context Bucket"],
        value_cols=["Exact Match", "Contains Match", "F1", "Num Samples"],
    )

    fig_bucket, axes_bucket = plt.subplots(1, 2, figsize=(12.0, 4.6))
    _plot_bucket_curve(axes_bucket[0], bucket_by_model, "Exact Match", plot_order)
    _plot_bucket_curve(axes_bucket[1], bucket_by_model, "F1", plot_order)
    axes_bucket[0].set_title("Exact Match vs Context")
    axes_bucket[1].set_title("F1 vs Context")
    bucket_png, bucket_pdf = save_publication_figure(fig_bucket, args.save_dir, f"{args.figure_stem}_bucket")
    plt.close(fig_bucket)

    print(f"Loaded {len(files)} result files")
    print(f"Saved CSV: {overall_raw_csv}")
    print(f"Saved CSV: {task_raw_csv}")
    print(f"Saved CSV: {bucket_raw_csv}")
    print(f"Saved CSV: {overall_mean_csv}")
    print(f"Saved CSV: {task_mean_csv}")
    print(f"Saved CSV: {bucket_mean_csv}")
    print(f"Figure: {overall_png}")
    print(f"Figure: {overall_pdf}")
    print(f"Figure: {bucket_png}")
    print(f"Figure: {bucket_pdf}")

    if not overall_by_model.empty:
        ranked = overall_by_model.sort_values("Exact Match", ascending=False)
        print("\nOverall ranking by Exact Match:")
        cols = ["Model", "Exact Match", "F1", "Num Evaluated", "Num Total"]
        print(ranked[cols].to_string(index=False))


if __name__ == "__main__":
    main()
