import os

import matplotlib as mpl


MODEL_ALIAS = {
    "RelaxedKaczmarzQNorm": "KLA",
    "GatedDeltaNet": "GDN",
    "Mamba2": "Mamba2",
}

MODEL_ORDER = ["KLA", "GDN", "Mamba2"]

# Colorblind-safe palette with distinct line/marker identities.
MODEL_STYLE = {
    "KLA": {"color": "#4E79A7", "linestyle": "-", "marker": "o"},
    "GDN": {"color": "#59A14F", "linestyle": "--", "marker": "s"},
    "Mamba2": {"color": "#E15759", "linestyle": "-.", "marker": "^"},
}


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        return "Unknown"
    for prefix, alias in MODEL_ALIAS.items():
        if model_name.startswith(prefix):
            return alias
    return model_name


def get_model_style(model_name: str) -> dict:
    return MODEL_STYLE.get(model_name, {"color": "#444444", "linestyle": "-", "marker": "o"})


def apply_publication_style() -> None:
    """Apply a consistent NeurIPS-like figure style across tasks."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "legend.frameon": False,
            "axes.grid": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_axis(ax, xlabel: str, ylabel: str, ylim=(0, 102)) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2, linestyle=":")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, handlelength=2.8)


def get_figures_dir(save_dir: str) -> str:
    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def get_tables_dir(save_dir: str) -> str:
    tables_dir = os.path.join(save_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    return tables_dir


def save_publication_figure(fig, save_dir: str, stem: str):
    figures_dir = get_figures_dir(save_dir)
    png_path = os.path.join(figures_dir, f"{stem}.png")
    pdf_path = os.path.join(figures_dir, f"{stem}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    return png_path, pdf_path