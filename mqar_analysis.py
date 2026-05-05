import argparse
import glob
import json
import os
import re

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


def _extract_int_tag(text: str, tag: str):
    if not text:
        return None
    match = re.search(rf"{tag}(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _parse_factor_key(value) -> int | None:
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1))


def _format_k_step(value, _pos):
    if abs(value) < 1000:
        return str(int(value))
    k = value / 1000.0
    if abs(k - round(k)) < 1e-8:
        return f"{int(round(k))}k"
    return f"{k:.1f}k"


def _set_legend_lower_right(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, handlelength=2.8)


def _extract_cli_arg_from_wandb_config(config_text: str, arg_name: str) -> str | None:
    pattern = rf"-\s+--{re.escape(arg_name)}\s*\n\s*-\s+([^\n]+)"
    match = re.search(pattern, config_text)
    if not match:
        return None
    return match.group(1).strip()


def _build_wandb_run_index(wandb_root: str):
    by_exp_name = {}
    by_out_dir = {}

    run_dirs = glob.glob(os.path.join(wandb_root, "wandb", "run-*"))
    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, "files", "config.yaml")
        output_log_path = os.path.join(run_dir, "files", "output.log")
        if not os.path.isfile(config_path) or not os.path.isfile(output_log_path):
            continue

        try:
            with open(config_path, "r", encoding="utf-8", errors="ignore") as fp:
                config_text = fp.read()
        except Exception:
            continue

        exp_name = _extract_cli_arg_from_wandb_config(config_text, "exp_name")
        out_dir = _extract_cli_arg_from_wandb_config(config_text, "out_dir")
        mtime = os.path.getmtime(output_log_path)

        if exp_name:
            prev = by_exp_name.get(exp_name)
            if prev is None or mtime > prev[0]:
                by_exp_name[exp_name] = (mtime, output_log_path)

        if out_dir:
            prev = by_out_dir.get(out_dir)
            if prev is None or mtime > prev[0]:
                by_out_dir[out_dir] = (mtime, output_log_path)

    return {
        "by_exp_name": {k: v[1] for k, v in by_exp_name.items()},
        "by_out_dir": {k: v[1] for k, v in by_out_dir.items()},
    }


def _select_wandb_output_log(index, exp_name: str | None, out_dir: str | None) -> str | None:
    if exp_name and exp_name in index["by_exp_name"]:
        return index["by_exp_name"][exp_name]
    if out_dir and out_dir in index["by_out_dir"]:
        return index["by_out_dir"][out_dir]
    return None


def _parse_val_curve_from_output_log(path: str):
    pattern = re.compile(r"Step\s+(\d+):\s+Val\s+Acc\s+([0-9]*\.?[0-9]+)")
    points_by_step = {}

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                match = pattern.search(line)
                if not match:
                    continue
                step = int(match.group(1))
                acc = float(match.group(2))
                # Training logs are usually in [0, 1], but keep robustness for [%] values.
                if acc <= 1.0:
                    acc *= 100.0
                points_by_step[step] = acc
    except Exception:
        return []

    return sorted(points_by_step.items(), key=lambda x: x[0])


def load_training_val_curves(test_df: pd.DataFrame):
    if test_df.empty:
        return pd.DataFrame()

    index_cache = {}
    rows = []

    for _, row in test_df.iterrows():
        wandb_root = row.get("Wandb Dir")
        exp_name = row.get("Exp Name")
        out_dir = row.get("Out Dir")

        if not isinstance(wandb_root, str) or not wandb_root:
            continue
        if not os.path.isdir(wandb_root):
            continue

        if wandb_root not in index_cache:
            index_cache[wandb_root] = _build_wandb_run_index(wandb_root)

        log_path = _select_wandb_output_log(index_cache[wandb_root], exp_name, out_dir)
        if not log_path:
            continue

        points = _parse_val_curve_from_output_log(log_path)
        for step, acc in points:
            rows.append(
                {
                    "Model": row["Model"],
                    "Model Raw": row["Model Raw"],
                    "Step": int(step),
                    "Val Acc": float(acc),
                    "Seed": row.get("Seed"),
                    "Exp Name": exp_name,
                    "Path": row.get("Path"),
                }
            )

    return pd.DataFrame(rows)


def load_results(out_root: str):
    test_rows = []
    val_rows = []
    files = glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True)

    for path in files:
        try:
            with open(path, "r") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"Skip invalid result file {path}: {e}")
            continue

        args = data.get("args", {})
        model_raw = data.get("model_name", "Unknown")
        model = normalize_model_name(model_raw)

        data_dir = args.get("data_dir", "")
        exp_name = args.get("exp_name", "")

        seq_len = _extract_int_tag(data_dir, "seq")
        key_len = _extract_int_tag(data_dir, "key")

        if seq_len is None:
            seq_len = _extract_int_tag(exp_name, "seq")
        if key_len is None:
            key_len = _extract_int_tag(exp_name, "key")

        test_acc = data.get("test_acc")
        best_val_acc = data.get("best_val_acc")

        # Per-length extrapolation validation metrics logged at the end of training.
        final_val_acc = data.get("final_val_acc", {})
        if isinstance(final_val_acc, dict):
            for factor_key, acc in final_val_acc.items():
                factor = _parse_factor_key(factor_key)
                if factor is None:
                    continue
                if acc is None:
                    continue
                if seq_len is None:
                    # If we cannot infer train seq len, we cannot map factor to actual eval length.
                    continue
                val_rows.append(
                    {
                        "Model": model,
                        "Model Raw": model_raw,
                        "Val Acc": float(acc) * 100.0,
                        "Factor": int(factor),
                        "Train Seq Len": int(seq_len),
                        "Eval Seq Len": int(seq_len) * int(factor),
                        "Key Len": int(key_len) if key_len is not None else None,
                        "Seed": args.get("seed"),
                        "Path": path,
                    }
                )

        if test_acc is None or seq_len is None or key_len is None:
            print(f"Skip incomplete MQAR test metric file: {path}")
            continue

        test_rows.append(
            {
                "Model": model,
                "Model Raw": model_raw,
                "Test Acc": float(test_acc) * 100.0,
                "Best Val Acc": float(best_val_acc) * 100.0 if best_val_acc is not None else None,
                "Seq Len": int(seq_len),
                "Key Len": int(key_len),
                "Seed": args.get("seed"),
                "Exp Name": args.get("exp_name"),
                "Out Dir": args.get("out_dir"),
                "Wandb Dir": args.get("wandb_dir"),
                "Path": path,
            }
        )

    return pd.DataFrame(test_rows), pd.DataFrame(val_rows), files


def aggregate_for_plot(df: pd.DataFrame, group_cols, metric_col: str):
    if df.empty:
        return df
    return df.groupby(group_cols, as_index=False)[metric_col].mean()


def _ordered_models(df: pd.DataFrame):
    present = set(df["Model"].tolist())
    ordered = [model for model in MODEL_ORDER if model in present]
    extras = sorted(present.difference(set(ordered)))
    return ordered + extras


def plot_final_val_extrapolation(ax, val_df: pd.DataFrame):
    if val_df.empty:
        ax.text(0.5, 0.5, "No final_val_acc metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(val_df, ["Model", "Eval Seq Len"], "Val Acc")
    for model in _ordered_models(agg):
        model_df = agg[agg["Model"] == model].sort_values("Eval Seq Len")
        style = get_model_style(model)
        ax.plot(
            model_df["Eval Seq Len"],
            model_df["Val Acc"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    format_axis(ax, xlabel="Evaluation sequence length", ylabel="Final val accuracy (%)", ylim=(0, 102))
    seq_ticks = sorted(agg["Eval Seq Len"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_ticks)
    ax.set_xticklabels([str(int(v)) for v in seq_ticks])
    _set_legend_lower_right(ax)


def plot_training_val_curve(ax, curve_df: pd.DataFrame):
    if curve_df.empty:
        ax.text(0.5, 0.5, "No val/acc step metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(curve_df, ["Model", "Step"], "Val Acc")
    for model in _ordered_models(agg):
        model_df = agg[agg["Model"] == model].sort_values("Step")
        style = get_model_style(model)
        ax.plot(
            model_df["Step"],
            model_df["Val Acc"],
            label=model,
            linewidth=2.0,
            markersize=5,
            markeredgewidth=0.7,
            **style,
        )

    format_axis(ax, xlabel="Training steps", ylabel="Validation accuracy (%)", ylim=(0, 102))
    max_step = int(agg["Step"].max())
    if max_step >= 2000:
        ticks = list(range(2000, max_step + 1, 2000))
        if ticks[-1] != max_step and max_step not in ticks:
            ticks.append(max_step)
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_k_step))
    _set_legend_lower_right(ax)


def plot_test_accuracy_vs_seq_len(ax, test_df: pd.DataFrame, key_len: int):
    subset = test_df[test_df["Key Len"] == key_len]
    if subset.empty:
        ax.text(0.5, 0.5, f"No metrics for key length={key_len}", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(subset, ["Model", "Seq Len"], "Test Acc")
    for model in _ordered_models(agg):
        model_df = agg[agg["Model"] == model].sort_values("Seq Len")
        style = get_model_style(model)
        ax.plot(
            model_df["Seq Len"],
            model_df["Test Acc"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    format_axis(ax, xlabel="Sequence length", ylabel="Accuracy (%)", ylim=(0, 102))
    seq_ticks = sorted(agg["Seq Len"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_ticks)
    ax.set_xticklabels([str(int(v)) for v in seq_ticks])
    _set_legend_lower_right(ax)


def plot_test_accuracy_vs_key_len(ax, test_df: pd.DataFrame, seq_len: int):
    subset = test_df[test_df["Seq Len"] == seq_len]
    if subset.empty:
        ax.text(0.5, 0.5, f"No metrics for seq length={seq_len}", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(subset, ["Model", "Key Len"], "Test Acc")
    for model in _ordered_models(agg):
        model_df = agg[agg["Model"] == model].sort_values("Key Len")
        style = get_model_style(model)
        ax.plot(
            model_df["Key Len"],
            model_df["Test Acc"],
            label=model,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=0.7,
            **style,
        )

    format_axis(ax, xlabel="Association order (key length)", ylabel="Accuracy (%)", ylim=(0, 102))
    ax.set_xticks(sorted(agg["Key Len"].unique()))
    _set_legend_lower_right(ax)


def main():
    apply_publication_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="./out/mqar")
    parser.add_argument("--save_dir", type=str, default="./analysis_results/mqar")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tables_dir = get_tables_dir(args.save_dir)

    test_df, val_df, files = load_results(args.out_root)
    if not files:
        print("No results.json found.")
        return
    if test_df.empty and val_df.empty:
        print("No valid MQAR metrics parsed from results.json files.")
        return

    if not test_df.empty:
        print("Loaded test results (mean acc %):")
        print(test_df.groupby(["Model", "Seq Len", "Key Len"])["Test Acc"].mean())
    if not val_df.empty:
        print("Loaded final validation extrapolation results (mean acc %):")
        print(val_df.groupby(["Model", "Eval Seq Len"])["Val Acc"].mean())

    curve_df = load_training_val_curves(test_df)
    if not curve_df.empty:
        print("Loaded training validation curves (mean acc %):")
        print(curve_df.groupby(["Model", "Step"])["Val Acc"].mean().head(20))

    if not test_df.empty:
        test_df.to_csv(os.path.join(tables_dir, "summary_raw.csv"), index=False)
        test_df_mean = aggregate_for_plot(test_df, ["Model", "Seq Len", "Key Len"], "Test Acc")
        test_df_mean.to_csv(os.path.join(tables_dir, "summary_mean.csv"), index=False)

    if not val_df.empty:
        val_df.to_csv(os.path.join(tables_dir, "final_val_extrapolation_raw.csv"), index=False)
        val_df_mean = aggregate_for_plot(val_df, ["Model", "Train Seq Len", "Eval Seq Len"], "Val Acc")
        val_df_mean.to_csv(os.path.join(tables_dir, "final_val_extrapolation_mean.csv"), index=False)

    if not curve_df.empty:
        curve_df.to_csv(os.path.join(tables_dir, "val_curve_raw.csv"), index=False)
        curve_df_mean = aggregate_for_plot(curve_df, ["Model", "Step"], "Val Acc")
        curve_df_mean.to_csv(os.path.join(tables_dir, "val_curve_mean.csv"), index=False)

    fig_top, ax1 = plt.subplots(figsize=(7.2, 4.2))
    if not val_df.empty:
        plot_final_val_extrapolation(ax1, val_df)
        ax1.set_ylim(50, 102)
    elif not test_df.empty:
        default_key_len = 1 if (test_df["Key Len"] == 1).any() else int(test_df["Key Len"].mode().iloc[0])
        plot_test_accuracy_vs_seq_len(ax1, test_df, key_len=default_key_len)
        ax1.set_ylim(50, 102)
    else:
        ax1.text(0.5, 0.5, "No plot data available", ha="center", va="center")
        ax1.set_axis_off()
    top_png_path, top_pdf_path = save_publication_figure(fig_top, args.save_dir, "mqar_metrics_panel1")
    plt.close(fig_top)

    fig_bottom, ax2 = plt.subplots(figsize=(7.2, 4.2))
    if not curve_df.empty:
        plot_training_val_curve(ax2, curve_df)
    elif not test_df.empty:
        common_seq_len = int(test_df["Seq Len"].mode().iloc[0])
        plot_test_accuracy_vs_key_len(ax2, test_df, seq_len=common_seq_len)
    else:
        ax2.text(0.5, 0.5, "No step curve metrics found", ha="center", va="center")
        ax2.set_axis_off()
    bottom_png_path, bottom_pdf_path = save_publication_figure(fig_bottom, args.save_dir, "mqar_metrics_panel2")
    plt.close(fig_bottom)

    print(f"Loaded {len(files)} result files")
    print(f"Saved plots and tables to {args.save_dir}")
    print(f"Figure: {top_png_path}")
    print(f"Figure: {top_pdf_path}")
    print(f"Figure: {bottom_png_path}")
    print(f"Figure: {bottom_pdf_path}")


if __name__ == "__main__":
    main()
