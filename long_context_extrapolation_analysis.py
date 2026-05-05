import argparse
import glob
import json
import math
import os

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
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_single_result(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skip invalid JSON {path}: {e}")
        return []

    model_raw = data.get("model_name") or data.get("config_name") or "Unknown"
    model = normalize_model_name(str(model_raw))

    rows = []
    for item in data.get("results", []):
        length = item.get("context_length")
        if length is None:
            continue

        ppl = _safe_float(item.get("ppl"))
        loss = _safe_float(item.get("loss"))
        num_batches = item.get("num_batches")
        if num_batches is None:
            num_batches = 0

        rows.append(
            {
                "Model": model,
                "Model Raw": str(model_raw),
                "Context Length": int(length),
                "Loss": loss,
                "PPL": ppl,
                "Seed": data.get("seed"),
                "Path": path,
                "Num Batches": int(num_batches),
                "Config Name": data.get("config_name", ""),
                "Checkpoint": data.get("ckpt_path", ""),
            }
        )

    return rows


def load_results(out_root: str):
    files = glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True)

    rows = []
    for path in files:
        rows.extend(_load_single_result(path))

    return pd.DataFrame(rows), files


def aggregate_mean(df: pd.DataFrame):
    if df.empty:
        return df

    finite_df = df[df["PPL"].map(math.isfinite)].copy()
    if finite_df.empty:
        return finite_df

    agg = (
        finite_df.groupby(["Model", "Context Length"], as_index=False)
        .agg(
            {
                "Loss": "mean",
                "PPL": "mean",
                "Num Batches": "mean",
                "Path": "nunique",
            }
        )
        .rename(columns={"Path": "Num Runs", "Num Batches": "Mean Num Batches"})
    )

    return agg


def _fit_linear_slope(x_values, y_values):
    n = len(x_values)
    if n < 2:
        return float("nan")

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    var_x = sum((x - x_mean) ** 2 for x in x_values)
    if var_x <= 0:
        return float("nan")

    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    return cov_xy / var_x


def build_relative_and_slope(mean_df: pd.DataFrame, baseline_length: int):
    if mean_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    relative_rows = []
    slope_rows = []

    for model in sorted(mean_df["Model"].unique()):
        model_df = mean_df[mean_df["Model"] == model].sort_values("Context Length").copy()
        if model_df.empty:
            continue

        lengths = model_df["Context Length"].tolist()
        baseline_for_model = baseline_length if baseline_length in lengths else min(lengths)
        base_row = model_df[model_df["Context Length"] == baseline_for_model].iloc[0]
        baseline_ppl = float(base_row["PPL"])

        slope_x = []
        slope_y = []

        for _, row in model_df.iterrows():
            ppl = float(row["PPL"])
            length = int(row["Context Length"])
            ratio = ppl / baseline_ppl if baseline_ppl > 0 else float("nan")
            inc_pct = (ratio - 1.0) * 100.0 if math.isfinite(ratio) else float("nan")

            relative_rows.append(
                {
                    "Model": model,
                    "Context Length": length,
                    "PPL": ppl,
                    "Baseline Length": int(baseline_for_model),
                    "Baseline PPL": baseline_ppl,
                    "PPL Ratio vs Baseline": ratio,
                    "PPL Increase (%) vs Baseline": inc_pct,
                    "Num Runs": int(row["Num Runs"]),
                }
            )

            if length >= baseline_for_model and baseline_ppl > 0 and ppl > 0:
                x = math.log2(length / float(baseline_for_model))
                y = math.log(ppl / baseline_ppl)
                slope_x.append(x)
                slope_y.append(y)

        slope = _fit_linear_slope(slope_x, slope_y)
        ratio_per_double = math.exp(slope) if math.isfinite(slope) else float("nan")

        max_row = model_df.iloc[model_df["Context Length"].argmax()]
        max_len = int(max_row["Context Length"])
        ppl_at_max = float(max_row["PPL"])
        ratio_at_max = ppl_at_max / baseline_ppl if baseline_ppl > 0 else float("nan")

        slope_rows.append(
            {
                "Model": model,
                "Baseline Length": int(baseline_for_model),
                "Baseline PPL": baseline_ppl,
                "Slope ln(PPL_ratio) per doubling": slope,
                "PPL Ratio per Doubling": ratio_per_double,
                "PPL Increase (%) per Doubling": (ratio_per_double - 1.0) * 100.0
                if math.isfinite(ratio_per_double)
                else float("nan"),
                "Max Length": max_len,
                "PPL@Max Length": ppl_at_max,
                "PPL Ratio at Max Length": ratio_at_max,
                "Num Length Points": len(slope_x),
            }
        )

    relative_df = pd.DataFrame(relative_rows)
    slope_df = pd.DataFrame(slope_rows)
    if not slope_df.empty:
        slope_df = slope_df.sort_values("Slope ln(PPL_ratio) per doubling", ascending=True)

    return relative_df, slope_df


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


def plot_ppl_curve(ax, mean_df: pd.DataFrame, plot_order):
    if mean_df.empty:
        ax.text(0.5, 0.5, "No long-context metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    for model in plot_order:
        model_df = mean_df[mean_df["Model"] == model].sort_values("Context Length")
        if model_df.empty:
            continue
        style = get_model_style(model)
        ax.plot(
            model_df["Context Length"],
            model_df["PPL"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    ax.set_xlabel("Context length")
    ax.set_ylabel("Validation perplexity")
    _setup_length_axis(ax, mean_df["Context Length"].tolist())
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, handlelength=2.8)


def plot_relative_curve(ax, relative_df: pd.DataFrame, plot_order):
    if relative_df.empty:
        ax.text(0.5, 0.5, "No relative metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    for model in plot_order:
        model_df = relative_df[relative_df["Model"] == model].sort_values("Context Length")
        if model_df.empty:
            continue
        style = get_model_style(model)
        ax.plot(
            model_df["Context Length"],
            model_df["PPL Ratio vs Baseline"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    ax.axhline(1.0, color="#666666", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Context length")
    ax.set_ylabel("PPL / PPL@baseline")
    _setup_length_axis(ax, relative_df["Context Length"].tolist())
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2, linestyle=":")


def main():
    apply_publication_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="./out/long_context")
    parser.add_argument("--save_dir", type=str, default="./analysis_results/pretrain")
    parser.add_argument("--baseline_length", type=int, default=4096)
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated model aliases to keep (e.g. KLA,GDN,Mamba2)",
    )
    parser.add_argument("--figure_stem", type=str, default="long_context_extrapolation")
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
    relative_df, slope_df = build_relative_and_slope(mean_df, baseline_length=args.baseline_length)

    raw_csv = os.path.join(tables_dir, "long_context_ppl_raw.csv")
    mean_csv = os.path.join(tables_dir, "long_context_ppl_mean.csv")
    rel_csv = os.path.join(tables_dir, "long_context_relative_to_baseline.csv")
    slope_csv = os.path.join(tables_dir, "long_context_slope_summary.csv")

    raw_df.to_csv(raw_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    relative_df.to_csv(rel_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)

    plot_order = _resolve_plot_order(mean_df)

    fig_ppl, ax_ppl = plt.subplots(figsize=(6.4, 4.8))
    plot_ppl_curve(ax_ppl, mean_df, plot_order)
    ppl_png_path, ppl_pdf_path = save_publication_figure(
        fig_ppl, args.save_dir, f"{args.figure_stem}_ppl"
    )
    plt.close(fig_ppl)

    fig_relative, ax_relative = plt.subplots(figsize=(6.4, 4.8))
    plot_relative_curve(ax_relative, relative_df, plot_order)
    rel_png_path, rel_pdf_path = save_publication_figure(
        fig_relative, args.save_dir, f"{args.figure_stem}_relative"
    )
    plt.close(fig_relative)

    print(f"Loaded {len(files)} long-context result files")
    print(f"Saved raw CSV: {raw_csv}")
    print(f"Saved mean CSV: {mean_csv}")
    print(f"Saved relative CSV: {rel_csv}")
    print(f"Saved slope CSV: {slope_csv}")
    print(f"Figure (PPL): {ppl_png_path}")
    print(f"Figure (PPL): {ppl_pdf_path}")
    print(f"Figure (Relative): {rel_png_path}")
    print(f"Figure (Relative): {rel_pdf_path}")

    if not slope_df.empty:
        print("\nSlope summary (smaller is better):")
        print(
            slope_df[
                [
                    "Model",
                    "Slope ln(PPL_ratio) per doubling",
                    "PPL Ratio per Doubling",
                    "PPL Ratio at Max Length",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
