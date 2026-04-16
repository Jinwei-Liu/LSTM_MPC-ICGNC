"""Plot training-curve comparisons in a Nature-like publication style.

Examples
--------
Compare step-level training loss against number of seen samples:

    python plot_training_comparison.py ^
        --csvs exp1/vehicle_lstm_step_history.csv exp2/vehicle_lstm_step_history.csv ^
        --labels 5k_samples 20k_samples ^
        --x-col samples_seen ^
        --y-col train_loss ^
        --smoothing 50 ^
        --output figures/lstm_train_loss_comparison

Compare epoch-level validation loss:

    python plot_training_comparison.py ^
        --csvs exp1/vehicle_lstm_epoch_history.csv exp2/vehicle_lstm_epoch_history.csv ^
        --labels baseline more_data ^
        --x-col samples_seen ^
        --y-col val_loss ^
        --marker o ^
        --output figures/lstm_val_loss_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NATURE_COLORS = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9D755D",  # brown
]


def apply_nature_style(font_size: int = 10) -> None:
    """Approximate a clean Nature-like figure style."""
    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 4.6),
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": font_size - 1,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
        }
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1, center=False).mean().to_numpy()


def derive_default_label(path: Path) -> str:
    name = path.stem
    name = name.replace("vehicle_lstm_", "")
    name = name.replace("vehicle_lstm_mpc_", "")
    name = name.replace("_history", "")
    return name


def validate_columns(df: pd.DataFrame, x_col: str, y_col: str, path: Path) -> None:
    missing = [col for col in (x_col, y_col) if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def plot_curves(
    csv_paths: list[Path],
    labels: list[str],
    x_col: str,
    y_col: str,
    smoothing: int,
    output_stem: Path,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    marker: str | None,
    annotate_last: bool,
    logy: bool,
    show_raw: bool,
) -> None:
    apply_nature_style()

    fig, ax = plt.subplots()

    for idx, (path, label) in enumerate(zip(csv_paths, labels)):
        df = pd.read_csv(path)
        validate_columns(df, x_col, y_col, path)

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy(dtype=float)
        y_smooth = moving_average(y, smoothing)
        color = NATURE_COLORS[idx % len(NATURE_COLORS)]

        if show_raw and smoothing > 1:
            ax.plot(
                x,
                y,
                color=color,
                alpha=0.18,
                linewidth=1.0,
                label=f"{label} (raw)",
            )

        ax.plot(
            x,
            y_smooth,
            color=color,
            label=label,
            marker=marker if marker else None,
            markevery=max(len(x) // 12, 1) if marker else None,
        )

        if annotate_last and len(x) > 0:
            ax.text(
                x[-1],
                y_smooth[-1],
                f" {label}",
                color=color,
                va="center",
                ha="left",
            )

    ax.set_xlabel(xlabel or x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel or y_col.replace("_", " ").title())
    ax.set_title(title or f"{(y_col.replace('_', ' ')).title()} vs {(x_col.replace('_', ' ')).title()}")

    if logy:
        ax.set_yscale("log")

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.02)

    if not annotate_last:
        ax.legend(loc="best")

    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare training histories with a Nature-like plotting style"
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="One or more history CSV files to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for the curves; must match number of csvs if provided",
    )
    parser.add_argument(
        "--x-col",
        default="samples_seen",
        help="CSV column used for the x-axis (default: samples_seen)",
    )
    parser.add_argument(
        "--y-col",
        default="train_loss",
        help="CSV column used for the y-axis (default: train_loss)",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=1,
        help="Moving-average window size for smoothing (default: 1 = no smoothing)",
    )
    parser.add_argument(
        "--output",
        default="training_comparison",
        help="Output file stem; .png and .pdf are both saved",
    )
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--xlabel", default=None, help="Optional x-axis label")
    parser.add_argument("--ylabel", default=None, help="Optional y-axis label")
    parser.add_argument("--marker", default=None, help="Optional line marker, e.g. o or s")
    parser.add_argument(
        "--annotate-last",
        action="store_true",
        help="Write labels next to the last point instead of using a legend",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale on the y-axis",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="If smoothing is enabled, also draw the raw curve with low alpha",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

    if args.labels is None or len(args.labels) == 0:
        labels = [derive_default_label(path) for path in csv_paths]
    else:
        if len(args.labels) != len(csv_paths):
            raise ValueError("Number of labels must match number of csvs")
        labels = args.labels

    plot_curves(
        csv_paths=csv_paths,
        labels=labels,
        x_col=args.x_col,
        y_col=args.y_col,
        smoothing=args.smoothing,
        output_stem=Path(args.output),
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        marker=args.marker,
        annotate_last=args.annotate_last,
        logy=args.logy,
        show_raw=args.show_raw,
    )


if __name__ == "__main__":
    main()
