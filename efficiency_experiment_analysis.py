import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List

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


def _parse_csv_keywords(value: str):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_float(value):
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value):
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _flatten_results(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skip invalid JSON {path}: {e}")
        return []

    model_raw = data.get("model_name") or data.get("config_name") or "Unknown"
    model = normalize_model_name(str(model_raw))

    rows = []
    for item in data.get("prefill_results", []):
        rows.append(
            {
                "Model": model,
                "Model Raw": str(model_raw),
                "Phase": "prefill",
                "Context Length": _safe_int(item.get("context_length")),
                "Batch Size": _safe_int(item.get("batch_size")),
                "New Tokens": 0,
                "Latency Mean (ms)": _safe_float(item.get("latency_ms_mean")),
                "Latency P50 (ms)": _safe_float(item.get("latency_ms_p50")),
                "Latency P90 (ms)": _safe_float(item.get("latency_ms_p90")),
                "Latency Std (ms)": _safe_float(item.get("latency_ms_std")),
                "Latency/Token (ms)": float("nan"),
                "Tokens/s": _safe_float(item.get("tokens_per_sec")),
                "Peak Allocated MiB": _safe_float(item.get("peak_allocated_mib")),
                "Peak Reserved MiB": _safe_float(item.get("peak_reserved_mib")),
                "Path": path,
                "Config Name": data.get("config_name", ""),
                "Checkpoint": data.get("ckpt_path", ""),
            }
        )

    for item in data.get("decode_results", []):
        rows.append(
            {
                "Model": model,
                "Model Raw": str(model_raw),
                "Phase": "decode",
                "Context Length": _safe_int(item.get("context_length")),
                "Batch Size": _safe_int(item.get("batch_size")),
                "New Tokens": _safe_int(item.get("new_tokens")),
                "Latency Mean (ms)": _safe_float(item.get("latency_ms_mean")),
                "Latency P50 (ms)": _safe_float(item.get("latency_ms_p50")),
                "Latency P90 (ms)": _safe_float(item.get("latency_ms_p90")),
                "Latency Std (ms)": _safe_float(item.get("latency_ms_std")),
                "Latency/Token (ms)": _safe_float(item.get("latency_ms_per_token")),
                "Tokens/s": _safe_float(item.get("tokens_per_sec")),
                "Peak Allocated MiB": _safe_float(item.get("peak_allocated_mib")),
                "Peak Reserved MiB": _safe_float(item.get("peak_reserved_mib")),
                "Path": path,
                "Config Name": data.get("config_name", ""),
                "Checkpoint": data.get("ckpt_path", ""),
            }
        )

    return rows


def load_results(out_root: str):
    files = glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True)
    rows: List[Dict[str, Any]] = []
    for path in files:
        rows.extend(_flatten_results(path))

    return pd.DataFrame(rows), files


def aggregate_mean(df: pd.DataFrame):
    if df.empty:
        return df

    finite_df = df[df["Latency Mean (ms)"].map(math.isfinite)].copy()
    if finite_df.empty:
        return finite_df

    group_cols = [
        "Model",
        "Phase",
        "Context Length",
        "Batch Size",
        "New Tokens",
    ]

    agg = (
        finite_df.groupby(group_cols, as_index=False)
        .agg(
            {
                "Latency Mean (ms)": "mean",
                "Latency P50 (ms)": "mean",
                "Latency P90 (ms)": "mean",
                "Latency Std (ms)": "mean",
                "Latency/Token (ms)": "mean",
                "Tokens/s": "mean",
                "Peak Allocated MiB": "mean",
                "Peak Reserved MiB": "mean",
                "Path": "nunique",
            }
        )
        .rename(columns={"Path": "Num Runs"})
    )

    return agg


def _resolve_plot_order(df: pd.DataFrame):
    if df.empty:
        return []
    present_models = set(df["Model"].unique())
    order = [model for model in MODEL_ORDER if model in present_models]
    remaining = sorted(present_models - set(order))
    return order + remaining


def _setup_length_axis(ax, lengths):
    sorted_lengths = sorted(set(int(v) for v in lengths))
    if not sorted_lengths:
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted_lengths)
    ax.set_xticklabels([str(v) for v in sorted_lengths])


def _plot_prefill_latency(ax, prefill_df: pd.DataFrame, plot_order):
    if prefill_df.empty:
        ax.text(0.5, 0.5, "No prefill metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    for model in plot_order:
        model_df = prefill_df[prefill_df["Model"] == model].sort_values("Context Length")
        if model_df.empty:
            continue
        style = get_model_style(model)
        ax.plot(
            model_df["Context Length"],
            model_df["Latency Mean (ms)"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    ax.set_xlabel("Context length")
    ax.set_ylabel("Prefill latency (ms)")
    _setup_length_axis(ax, prefill_df["Context Length"].tolist())
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, handlelength=2.8)


def _plot_decode_throughput(ax, decode_df: pd.DataFrame, plot_order):
    if decode_df.empty:
        ax.text(0.5, 0.5, "No decode metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    for model in plot_order:
        model_df = decode_df[decode_df["Model"] == model].sort_values("Context Length")
        if model_df.empty:
            continue
        style = get_model_style(model)
        ax.plot(
            model_df["Context Length"],
            model_df["Tokens/s"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    ax.set_xlabel("Context length")
    ax.set_ylabel("Decode throughput (tokens/s)")
    _setup_length_axis(ax, decode_df["Context Length"].tolist())
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, handlelength=2.8)


def _plot_decode_tpot(ax, decode_df: pd.DataFrame, plot_order):
    if decode_df.empty:
        ax.text(0.5, 0.5, "No decode metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    for model in plot_order:
        model_df = decode_df[decode_df["Model"] == model].sort_values("Context Length")
        if model_df.empty:
            continue
        style = get_model_style(model)
        ax.plot(
            model_df["Context Length"],
            model_df["Latency/Token (ms)"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    ax.set_xlabel("Context length")
    ax.set_ylabel("Decode TPOT (ms)")
    _setup_length_axis(ax, decode_df["Context Length"].tolist())
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, handlelength=2.8)


def _pick_batch(df: pd.DataFrame, requested: int) -> int:
    if df.empty:
        return requested
    available = sorted(set(int(v) for v in df["Batch Size"].tolist()))
    if requested in available:
        return requested
    return available[0]


def _pick_new_tokens(df: pd.DataFrame, requested: int) -> int:
    if df.empty:
        return requested
    available = sorted(set(int(v) for v in df["New Tokens"].tolist()))
    if requested in available:
        return requested
    return available[0]


def main():
    apply_publication_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="./out/efficiency")
    parser.add_argument("--save_dir", type=str, default="./analysis_results/efficiency")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated model aliases to keep (e.g. KLA,GDN,Mamba2)",
    )
    parser.add_argument("--plot_prefill_batch", type=int, default=1)
    parser.add_argument("--plot_decode_batch", type=int, default=1)
    parser.add_argument("--plot_new_tokens", type=int, default=128)
    parser.add_argument("--figure_stem", type=str, default="efficiency")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tables_dir = get_tables_dir(args.save_dir)

    raw_df, files = load_results(args.out_root)
    if not files:
        print(f"No results.json found under {args.out_root}")
        return

    model_filter = set(_parse_csv_keywords(args.models))
    if model_filter:
        raw_df = raw_df[raw_df["Model"].isin(model_filter)].copy()

    if raw_df.empty:
        print("No rows remain after filtering.")
        return

    mean_df = aggregate_mean(raw_df)

    prefill_mean = mean_df[mean_df["Phase"] == "prefill"].copy()
    decode_mean = mean_df[mean_df["Phase"] == "decode"].copy()

    selected_prefill_batch = _pick_batch(prefill_mean, args.plot_prefill_batch)
    selected_decode_batch = _pick_batch(decode_mean, args.plot_decode_batch)
    selected_new_tokens = _pick_new_tokens(decode_mean, args.plot_new_tokens)

    prefill_plot_df = prefill_mean[prefill_mean["Batch Size"] == selected_prefill_batch].copy()
    decode_plot_df = decode_mean[
        (decode_mean["Batch Size"] == selected_decode_batch)
        & (decode_mean["New Tokens"] == selected_new_tokens)
    ].copy()

    raw_csv = os.path.join(tables_dir, "efficiency_raw.csv")
    mean_csv = os.path.join(tables_dir, "efficiency_mean.csv")
    prefill_csv = os.path.join(tables_dir, "efficiency_prefill_summary.csv")
    decode_csv = os.path.join(tables_dir, "efficiency_decode_summary.csv")

    raw_df.to_csv(raw_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    prefill_mean.to_csv(prefill_csv, index=False)
    decode_mean.to_csv(decode_csv, index=False)

    stale_kv_artifacts = [
        os.path.join(tables_dir, "efficiency_kv_cache_summary.csv"),
        os.path.join(args.save_dir, "figures", f"{args.figure_stem}_kv_cache.png"),
        os.path.join(args.save_dir, "figures", f"{args.figure_stem}_kv_cache.pdf"),
        os.path.join(args.save_dir, "efficiency_kv_cache_summary.csv"),
        os.path.join(args.save_dir, f"{args.figure_stem}_kv_cache.png"),
        os.path.join(args.save_dir, f"{args.figure_stem}_kv_cache.pdf"),
    ]
    for stale_path in stale_kv_artifacts:
        if os.path.exists(stale_path):
            os.remove(stale_path)
            print(f"Removed stale KV artifact: {stale_path}")

    plot_order = _resolve_plot_order(mean_df)

    fig_prefill, ax_prefill = plt.subplots(figsize=(6.4, 4.8))
    _plot_prefill_latency(ax_prefill, prefill_plot_df, plot_order)
    prefill_png, prefill_pdf = save_publication_figure(fig_prefill, args.save_dir, f"{args.figure_stem}_prefill_latency")
    plt.close(fig_prefill)

    fig_decode, ax_decode = plt.subplots(figsize=(6.4, 4.8))
    _plot_decode_throughput(ax_decode, decode_plot_df, plot_order)
    decode_png, decode_pdf = save_publication_figure(fig_decode, args.save_dir, f"{args.figure_stem}_decode_throughput")
    plt.close(fig_decode)

    fig_decode_tpot, ax_decode_tpot = plt.subplots(figsize=(6.4, 4.8))
    _plot_decode_tpot(ax_decode_tpot, decode_plot_df, plot_order)
    decode_tpot_png, decode_tpot_pdf = save_publication_figure(fig_decode_tpot, args.save_dir, f"{args.figure_stem}_decode_tpot")
    plt.close(fig_decode_tpot)

    print(f"Loaded {len(files)} efficiency result files")
    print(f"Saved raw CSV: {raw_csv}")
    print(f"Saved mean CSV: {mean_csv}")
    print(f"Saved prefill CSV: {prefill_csv}")
    print(f"Saved decode CSV: {decode_csv}")
    print(f"Selected prefill batch for plots: {selected_prefill_batch}")
    print(f"Selected decode batch for plots : {selected_decode_batch}")
    print(f"Selected decode new_tokens      : {selected_new_tokens}")
    print(f"Figure (Prefill): {prefill_png}")
    print(f"Figure (Prefill): {prefill_pdf}")
    print(f"Figure (Decode): {decode_png}")
    print(f"Figure (Decode): {decode_pdf}")
    print(f"Figure (Decode TPOT): {decode_tpot_png}")
    print(f"Figure (Decode TPOT): {decode_tpot_pdf}")


if __name__ == "__main__":
    main()
