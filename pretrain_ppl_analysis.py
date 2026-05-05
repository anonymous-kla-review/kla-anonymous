import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from analysis_plot_style import apply_publication_style, get_tables_dir, save_publication_figure


MODEL_DIR_MAP = {
    "KLA": "tsz512x4k_512x4k_1B_RelaxedKaczmarzQNorm_0.4B",
    "GDN": "tsz512x4k_512x4k_1B_GatedDeltaNet_0.4B",
    "GLA": "tsz512x4k_512x4k_1B_GLA_0.4B",
    "DeltaNet": "tsz512x4k_512x4k_1B_DeltaNet_0.4B",
    "Longhorn": "tsz512x4k_512x4k_1B_Longhorn_0.4B",
    "Mamba2": "tsz512x4k_512x4k_1B_Mamba2_0.4B",
}

MODEL_STYLE = {
    "KLA": {"color": "#4E79A7", "linestyle": "-", "marker": "o"},
    "GDN": {"color": "#59A14F", "linestyle": "--", "marker": "s"},
    "GLA": {"color": "#F28E2B", "linestyle": "-", "marker": "D"},
    "DeltaNet": {"color": "#B07AA1", "linestyle": "-.", "marker": "v"},
    "Longhorn": {"color": "#9C755F", "linestyle": "--", "marker": "P"},
    "Mamba2": {"color": "#E15759", "linestyle": "-.", "marker": "^"},
}

METRIC_KEY = "metric/val_ppl@1x"
CURVE_PATTERN = re.compile(
    r"Recorded metrics at\s+([0-9]+)\s+tokens:\s+loss\s+([0-9eE+\-.]+),\s+ppl\s+([0-9eE+\-.]+)"
)


def _load_json(path: str):
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
        if not isinstance(data, dict):
            continue
        if METRIC_KEY not in data:
            continue

        timestamp = data.get("_timestamp", 0.0)
        mtime = os.path.getmtime(path)
        score = (float(timestamp), float(mtime))

        if best_score is None or score > best_score:
            best_score = score
            best_path = path

    return best_path


def _parse_curve(output_log_path: str):
    if not os.path.isfile(output_log_path):
        return []

    points = {}
    try:
        with open(output_log_path, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                match = CURVE_PATTERN.search(line)
                if not match:
                    continue
                tokens = int(match.group(1))
                ppl = float(match.group(3))
                points[tokens] = ppl
    except Exception:
        return []

    rows = []
    for tokens in sorted(points.keys()):
        rows.append(
            {
                "Tokens": int(tokens),
                "Tokens (B)": float(tokens) / 1e9,
                "Val PPL @1x": float(points[tokens]),
            }
        )
    return rows


def collect_model_metrics(wandb_root: str):
    curve_rows = []
    final_rows = []

    missing_models = []
    no_curve_models = []

    for model_name, model_dir in MODEL_DIR_MAP.items():
        model_wandb_root = os.path.join(wandb_root, model_dir)
        summary_path = _find_latest_summary(model_wandb_root)
        if summary_path is None:
            missing_models.append(model_name)
            continue

        summary = _load_json(summary_path)
        final_val = summary.get(METRIC_KEY)
        if final_val is None:
            missing_models.append(model_name)
            continue

        output_log_path = os.path.join(os.path.dirname(summary_path), "output.log")
        curve = _parse_curve(output_log_path)

        final_rows.append(
            {
                "Model": model_name,
                "Val PPL @1x": float(final_val),
                "Summary Path": summary_path,
                "Output Log": output_log_path,
            }
        )

        if curve:
            for item in curve:
                curve_rows.append(
                    {
                        "Model": model_name,
                        "Tokens": item["Tokens"],
                        "Tokens (B)": item["Tokens (B)"],
                        "Val PPL @1x": item["Val PPL @1x"],
                        "Summary Path": summary_path,
                        "Output Log": output_log_path,
                    }
                )
        else:
            no_curve_models.append(model_name)

    return pd.DataFrame(curve_rows), pd.DataFrame(final_rows), missing_models, no_curve_models


def _suffix_from_min_tokens(min_tokens: int) -> str:
    if min_tokens <= 0:
        return ""
    return f"_from_{min_tokens}"


def _downsample_for_plot(model_df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if model_df.empty:
        return model_df
    if max_points <= 1:
        return model_df.tail(1).copy()

    n_points = len(model_df)
    if n_points <= max_points:
        return model_df.copy()

    last_idx = n_points - 1
    sampled_indices = {
        int(round(i * last_idx / float(max_points - 1))) for i in range(max_points)
    }
    sampled_indices.add(last_idx)
    return model_df.iloc[sorted(sampled_indices)].copy()


def plot_curve(curve_df: pd.DataFrame, save_dir: str, stem: str, max_plot_points: int = 12):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    # Keep legend/plot order tied to the actual last-point values instead of hard-coded order.
    last_point = curve_df.sort_values("Tokens").groupby("Model", as_index=False).tail(1)
    plot_order = last_point.sort_values("Val PPL @1x", ascending=False)["Model"].tolist()

    for model_name in plot_order:
        model_df = curve_df[curve_df["Model"] == model_name].sort_values("Tokens")
        if model_df.empty:
            continue
        model_plot_df = _downsample_for_plot(model_df, max_plot_points)
        style = MODEL_STYLE.get(model_name, {"color": "#444444", "linestyle": "-", "marker": "o"})
        ax.plot(
            model_plot_df["Tokens (B)"],
            model_plot_df["Val PPL @1x"],
            label=model_name,
            linewidth=2.0,
            markersize=4.8,
            markeredgewidth=0.6,
            **style,
        )

    ax.set_xlabel("Training tokens (B)")
    ax.set_ylabel("Validation perplexity @1x")
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2, linestyle=":")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, handlelength=2.8)

    return save_publication_figure(fig, save_dir, stem)


def main():
    apply_publication_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_root", type=str, default="../save_dir/wandb")
    parser.add_argument("--save_dir", type=str, default="./analysis_results/pretrain")
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=0,
        help="Only keep curve points with tokens >= min_tokens.",
    )
    parser.add_argument(
        "--max_plot_points",
        type=int,
        default=12,
        help="Maximum sampled points per model used to draw each curve; always keeps the final point.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tables_dir = get_tables_dir(args.save_dir)

    curve_df, final_df, missing_models, no_curve_models = collect_model_metrics(args.wandb_root)
    suffix = _suffix_from_min_tokens(args.min_tokens)

    if final_df.empty:
        print("No 1B model val_ppl@1x metrics found.")
        return

    final_df.to_csv(os.path.join(tables_dir, "pretrain_val_ppl_1x_final_1B.csv"), index=False)

    if not curve_df.empty and args.min_tokens > 0:
        curve_df = curve_df[curve_df["Tokens"] >= int(args.min_tokens)].copy()

    if curve_df.empty:
        print("No val ppl @1x curve found in output.log for the selected 1B models.")
        if args.min_tokens > 0:
            print(f"No curve points satisfy tokens >= {args.min_tokens}.")
    else:
        curve_csv = os.path.join(tables_dir, f"pretrain_val_ppl_1x_curve_1B{suffix}.csv")
        curve_df.to_csv(curve_csv, index=False)
        stem = f"pretrain_val_ppl_1x_curve_1B{suffix}"
        png_path, pdf_path = plot_curve(curve_df, args.save_dir, stem, max_plot_points=args.max_plot_points)
        plt.close("all")
        print(f"Figure: {png_path}")
        print(f"Figure: {pdf_path}")
        print(f"Curve CSV: {curve_csv}")

    if missing_models:
        print("Missing metrics models:", ", ".join(missing_models))
    if no_curve_models:
        print("No-curve models:", ", ".join(no_curve_models))
    else:
        print("All selected 1B models have curve data.")

    print(f"Saved tables to {args.save_dir}")


if __name__ == "__main__":
    main()
