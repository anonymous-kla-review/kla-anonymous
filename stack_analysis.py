import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from analysis_plot_style import (
    MODEL_ORDER,
    apply_publication_style,
    format_axis,
    get_tables_dir,
    get_model_style,
    normalize_model_name,
    save_publication_figure,
)


def _format_k_step(value, _pos):
    if abs(value) < 1000:
        return str(int(value))
    k = value / 1000.0
    if abs(k - round(k)) < 1e-8:
        return f"{int(round(k))}k"
    return f"{k:.1f}k"


def load_results(out_root: str):
    result_files = glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True)

    seq_rows = []
    step_rows = []

    for path in result_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skip invalid result file {path}: {e}")
            continue

        model_raw = data.get("model_name", "Unknown")
        model = normalize_model_name(model_raw)
        args = data.get("args", {})
        seed = args.get("seed")

        eval_acc_by_seq_len = data.get("eval_acc_by_seq_len", {})
        for seq_len_str, acc in eval_acc_by_seq_len.items():
            seq_rows.append(
                {
                    "Model": model,
                    "Model Raw": model_raw,
                    "Seq Len": int(seq_len_str),
                    "Accuracy": float(acc) * 100.0,
                    "Seed": seed,
                    "Path": path,
                }
            )

        val_history = data.get("val_history", [])
        for item in val_history:
            if "step" not in item or "acc" not in item:
                continue
            step_rows.append(
                {
                    "Model": model,
                    "Model Raw": model_raw,
                    "Step": int(item["step"]),
                    "Accuracy": float(item["acc"]) * 100.0,
                    "Seed": seed,
                    "Path": path,
                }
            )

    return pd.DataFrame(seq_rows), pd.DataFrame(step_rows), result_files


def aggregate_for_plot(df: pd.DataFrame, group_cols):
    if df.empty:
        return df
    return df.groupby(group_cols, as_index=False)["Accuracy"].mean()


def plot_seq_len(ax, df: pd.DataFrame):
    if df.empty:
        ax.text(0.5, 0.5, "No sequence-length metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(df, ["Model", "Seq Len"])

    plotted = False
    for model in MODEL_ORDER:
        model_df = agg[agg["Model"] == model].sort_values("Seq Len")
        if model_df.empty:
            continue
        style = get_model_style(model)
        ax.plot(
            model_df["Seq Len"],
            model_df["Accuracy"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No sequence-length metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    format_axis(ax, xlabel="Sequence length", ylabel="Accuracy (%)", ylim=(80, 102))
    seq_ticks = sorted(agg["Seq Len"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_ticks)
    ax.set_xticklabels([str(int(v)) for v in seq_ticks])


def plot_steps(ax, df: pd.DataFrame):
    if df.empty:
        ax.text(0.5, 0.5, "No training-step metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(df, ["Model", "Step"])

    plotted = False
    for model in MODEL_ORDER:
        model_df = agg[agg["Model"] == model].sort_values("Step")
        if model_df.empty:
            continue
        style = get_model_style(model)
        ax.plot(
            model_df["Step"],
            model_df["Accuracy"],
            label=model,
            linewidth=2.0,
            markersize=5,
            markeredgewidth=0.7,
            **style,
        )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No training-step metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    format_axis(ax, xlabel="Training steps", ylabel="Accuracy (%)", ylim=(0, 102))
    max_step = int(agg["Step"].max())
    if max_step >= 2000:
        ticks = list(range(2000, max_step + 1, 2000))
        if ticks[-1] != max_step and max_step not in ticks:
            ticks.append(max_step)
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_k_step))


def main():
    apply_publication_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="./out/stack")
    parser.add_argument("--save_dir", type=str, default="./analysis_results/stack")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tables_dir = get_tables_dir(args.save_dir)

    seq_df, step_df, files = load_results(args.out_root)
    if not files:
        print("No results.json found.")
        return

    seq_plot_df = aggregate_for_plot(seq_df, ["Model", "Seq Len"]) if not seq_df.empty else seq_df
    step_plot_df = aggregate_for_plot(step_df, ["Model", "Step"]) if not step_df.empty else step_df

    if not seq_df.empty:
        seq_df.to_csv(os.path.join(tables_dir, "seq_len_metrics_raw.csv"), index=False)
        seq_plot_df.to_csv(os.path.join(tables_dir, "seq_len_metrics_mean.csv"), index=False)
    if not step_df.empty:
        step_df.to_csv(os.path.join(tables_dir, "step_metrics_raw.csv"), index=False)
        step_plot_df.to_csv(os.path.join(tables_dir, "step_metrics_mean.csv"), index=False)

    fig_seq, ax_seq = plt.subplots(figsize=(7.2, 4.2))
    plot_seq_len(ax_seq, seq_df)
    seq_png_path, seq_pdf_path = save_publication_figure(fig_seq, args.save_dir, "stack_seq_len_metrics")
    plt.close(fig_seq)

    fig_step, ax_step = plt.subplots(figsize=(7.2, 4.2))
    plot_steps(ax_step, step_df)
    step_png_path, step_pdf_path = save_publication_figure(fig_step, args.save_dir, "stack_step_metrics")
    plt.close(fig_step)

    print(f"Loaded {len(files)} result files")
    print(f"Saved plots and tables to {args.save_dir}")
    print(f"Figure: {seq_png_path}")
    print(f"Figure: {seq_pdf_path}")
    print(f"Figure: {step_png_path}")
    print(f"Figure: {step_pdf_path}")


if __name__ == "__main__":
    main()
