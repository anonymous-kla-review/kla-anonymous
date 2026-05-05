import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from analysis_plot_style import (
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


def _parse_factor_key(value):
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


def _extract_cli_arg_from_wandb_config(config_text: str, arg_name: str):
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


def _select_wandb_output_log(index, exp_name, out_dir):
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
                if acc <= 1.0:
                    acc *= 100.0
                points_by_step[step] = acc
    except Exception:
        return []

    return sorted(points_by_step.items(), key=lambda x: x[0])


def _variant_model_name(model_name: str, args: dict, out_root: str) -> str:
    return model_name


def _is_preferred_mamba_run(args: dict, out_root: str) -> bool:
    early_stop_patience = args.get("early_stop_patience")
    out_dir = str(args.get("out_dir", ""))
    out_root_str = str(out_root)

    return (
        (isinstance(early_stop_patience, (int, float)) and early_stop_patience >= 100000)
        or ("no_early_stop" in out_dir.lower())
        or ("no_early_stop" in out_root_str.lower())
    )


def _ordered_models(df: pd.DataFrame):
    preferred = ["KLA", "GDN", "Mamba2"]
    present = set(df["Model"].tolist())
    ordered = [model for model in preferred if model in present]
    extras = sorted(present.difference(set(ordered)))
    return ordered + extras


def _get_plot_style(model_name: str):
    return get_model_style(model_name)


def _filter_only_no_es(df: pd.DataFrame):
    if df.empty or "Model" not in df.columns or "Run Tag" not in df.columns:
        return df
    mamba_mask = df["Model"] == "Mamba2"
    if not mamba_mask.any():
        return df
    preferred_mask = mamba_mask & (df["Run Tag"] == "Preferred")
    if preferred_mask.any():
        return df[(~mamba_mask) | preferred_mask].copy()
    return df


def _filter_single_mamba(df: pd.DataFrame):
    if df.empty or "Model" not in df.columns or "Run Tag" not in df.columns:
        return df
    # Prefer the designated Mamba2 run when both preferred/regular Mamba2 runs exist.
    mamba_df = df[df["Model"] == "Mamba2"]
    if mamba_df.empty:
        return df
    has_preferred = (mamba_df["Run Tag"] == "Preferred").any()
    has_regular = (mamba_df["Run Tag"] == "Regular").any()
    if has_preferred and has_regular:
        return df[(df["Model"] != "Mamba2") | (df["Run Tag"] == "Preferred")].copy()
    return df


def aggregate_for_plot(df: pd.DataFrame, group_cols, metric_col: str):
    if df.empty:
        return df
    return df.groupby(group_cols, as_index=False)[metric_col].mean()


def load_results(out_roots, min_max_steps: int):
    test_rows = []
    val_rows = []
    parsed_files = []

    for out_root in out_roots:
        result_files = glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True)
        for path in result_files:
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception as e:
                print(f"Skip invalid result file {path}: {e}")
                continue

            args = data.get("args", {})
            max_steps = args.get("max_steps")
            if min_max_steps is not None:
                if max_steps is None or int(max_steps) < int(min_max_steps):
                    continue

            model_raw = data.get("model_name", "Unknown")
            model_base = normalize_model_name(model_raw)
            model = _variant_model_name(model_base, args, out_root)
            is_preferred_mamba = model == "Mamba2" and _is_preferred_mamba_run(args, out_root)
            run_tag = "Main"
            if model == "Mamba2":
                run_tag = "Preferred" if is_preferred_mamba else "Regular"

            data_dir = args.get("data_dir", "")
            exp_name = args.get("exp_name", "")

            seq_len = args.get("extrapol_base_seq_len")
            if seq_len is None:
                seq_len = _extract_int_tag(data_dir, "seq")
            if seq_len is None:
                seq_len = _extract_int_tag(exp_name, "seq")

            test_acc = data.get("test_acc")
            best_val_acc = data.get("best_val_acc")

            final_val_acc = data.get("final_val_acc", {})
            if isinstance(final_val_acc, dict):
                for factor_key, acc in final_val_acc.items():
                    factor = _parse_factor_key(factor_key)
                    if factor is None or acc is None or seq_len is None:
                        continue
                    val_rows.append(
                        {
                            "Model": model,
                            "Model Raw": model_raw,
                            "Run Tag": run_tag,
                            "Val Acc": float(acc) * 100.0,
                            "Factor": int(factor),
                            "Train Seq Len": int(seq_len),
                            "Eval Seq Len": int(seq_len) * int(factor),
                            "Seed": args.get("seed"),
                            "Exp Name": exp_name,
                            "Out Dir": args.get("out_dir"),
                            "Wandb Dir": args.get("wandb_dir"),
                            "Path": path,
                        }
                    )

            if test_acc is None:
                continue

            test_rows.append(
                {
                    "Model": model,
                    "Model Raw": model_raw,
                    "Run Tag": run_tag,
                    "Test Acc": float(test_acc) * 100.0,
                    "Best Val Acc": float(best_val_acc) * 100.0 if best_val_acc is not None else None,
                    "Seed": args.get("seed"),
                    "Max Steps": args.get("max_steps"),
                    "Exp Name": exp_name,
                    "Out Dir": args.get("out_dir"),
                    "Wandb Dir": args.get("wandb_dir"),
                    "Path": path,
                }
            )
            parsed_files.append(path)

    return pd.DataFrame(test_rows), pd.DataFrame(val_rows), sorted(set(parsed_files))


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
                    "Step": int(step),
                    "Val Acc": float(acc),
                    "Seed": row.get("Seed"),
                    "Exp Name": exp_name,
                    "Path": row.get("Path"),
                }
            )

    return pd.DataFrame(rows)


def plot_final_val_extrapolation(ax, val_df: pd.DataFrame):
    if val_df.empty:
        ax.text(0.5, 0.5, "No final_val_acc metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(val_df, ["Model", "Eval Seq Len"], "Val Acc")
    for model in _ordered_models(agg):
        model_df = agg[agg["Model"] == model].sort_values("Eval Seq Len")
        style = _get_plot_style(model)
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


def plot_training_val_curve(ax, curve_df: pd.DataFrame, max_curve_step=None):
    if max_curve_step is not None and not curve_df.empty:
        curve_df = curve_df[curve_df["Step"] <= int(max_curve_step)].copy()

    if curve_df.empty:
        ax.text(0.5, 0.5, "No val/acc step metrics found", ha="center", va="center")
        ax.set_axis_off()
        return

    agg = aggregate_for_plot(curve_df, ["Model", "Step"], "Val Acc")
    for model in _ordered_models(agg):
        model_df = agg[agg["Model"] == model].sort_values("Step")
        style = _get_plot_style(model)
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
    if max_curve_step is not None:
        ax.set_xlim(0, int(max_curve_step))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_k_step))


def main():
    apply_publication_style()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_roots",
        type=str,
        default="./out/sniah_synth,./out/sniah_synth_no_early_stop",
        help="Comma-separated SNIAH output roots to aggregate.",
    )
    parser.add_argument("--save_dir", type=str, default="./analysis_results/sniah")
    parser.add_argument(
        "--figure_stem",
        type=str,
        default="sniah_metrics",
        help="Output figure filename stem.",
    )
    parser.add_argument(
        "--only_no_es",
        action="store_true",
        help="Keep only the preferred Mamba2 run when multiple Mamba2 runs are available.",
    )
    parser.add_argument(
        "--keep_both_mamba",
        action="store_true",
        help="Keep both preferred and regular Mamba2 runs when both are available.",
    )
    parser.add_argument(
        "--max_curve_step",
        type=int,
        default=6000,
        help="Maximum step shown in the training-val curve panel.",
    )
    parser.add_argument(
        "--min_max_steps",
        type=int,
        default=10000,
        help="Filter out short debug runs whose configured max_steps is below this threshold.",
    )
    args = parser.parse_args()

    out_roots = [item.strip() for item in args.out_roots.split(",") if item.strip()]
    os.makedirs(args.save_dir, exist_ok=True)
    tables_dir = get_tables_dir(args.save_dir)

    test_df, val_df, files = load_results(out_roots, min_max_steps=args.min_max_steps)
    if not files:
        print("No valid SNIAH results.json found after filtering.")
        return

    if args.only_no_es:
        test_df = _filter_only_no_es(test_df)
        val_df = _filter_only_no_es(val_df)
    elif not args.keep_both_mamba:
        test_df = _filter_single_mamba(test_df)
        val_df = _filter_single_mamba(val_df)

    curve_df = load_training_val_curves(test_df)
    if args.max_curve_step is not None and not curve_df.empty:
        curve_df = curve_df[curve_df["Step"] <= int(args.max_curve_step)].copy()

    test_df.to_csv(os.path.join(tables_dir, "summary_raw.csv"), index=False)
    test_mean = aggregate_for_plot(test_df, ["Model"], "Test Acc")
    test_mean.to_csv(os.path.join(tables_dir, "summary_mean.csv"), index=False)

    val_df.to_csv(os.path.join(tables_dir, "final_val_extrapolation_raw.csv"), index=False)
    val_mean = aggregate_for_plot(val_df, ["Model", "Eval Seq Len"], "Val Acc")
    val_mean.to_csv(os.path.join(tables_dir, "final_val_extrapolation_mean.csv"), index=False)

    if not curve_df.empty:
        curve_df.to_csv(os.path.join(tables_dir, "val_curve_raw.csv"), index=False)
        curve_mean = aggregate_for_plot(curve_df, ["Model", "Step"], "Val Acc")
        curve_mean.to_csv(os.path.join(tables_dir, "val_curve_mean.csv"), index=False)

    fig_extrap, ax_extrap = plt.subplots(figsize=(7.2, 4.2))
    plot_final_val_extrapolation(ax_extrap, val_df)
    extrap_png_path, extrap_pdf_path = save_publication_figure(
        fig_extrap,
        args.save_dir,
        f"{args.figure_stem}_final_val_extrapolation",
    )
    plt.close(fig_extrap)

    fig_curve, ax_curve = plt.subplots(figsize=(7.2, 4.2))
    plot_training_val_curve(ax_curve, curve_df, max_curve_step=args.max_curve_step)
    curve_png_path, curve_pdf_path = save_publication_figure(
        fig_curve,
        args.save_dir,
        f"{args.figure_stem}_training_val_curve",
    )
    plt.close(fig_curve)

    print("Loaded SNIAH test results (mean acc %):")
    print(test_mean.sort_values("Test Acc", ascending=False))
    print("Loaded SNIAH final extrapolation validation results (mean acc %):")
    print(val_mean.sort_values(["Eval Seq Len", "Val Acc"], ascending=[True, False]))
    print(f"Loaded {len(files)} result files")
    print(f"Saved plots and tables to {args.save_dir}")
    print(f"Figure: {extrap_png_path}")
    print(f"Figure: {extrap_pdf_path}")
    print(f"Figure: {curve_png_path}")
    print(f"Figure: {curve_pdf_path}")


if __name__ == "__main__":
    main()