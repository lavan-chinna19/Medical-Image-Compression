"""Plot rate-distortion curves and compile summary statistics for baseline codecs.

Plots generated in results/plots/:
1. rd_curve_psnr.png: bpp (x) vs PSNR full (y)
2. rd_curve_ssim.png: bpp (x) vs SSIM full (y)
3. rd_curve_roi_psnr.png: bpp (x) vs ROI-only PSNR (y)

Outputs:
- results/baseline_summary.csv (Aggregated mean metrics by codec and setting)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medcomp.config import (
    BASELINES_CSV_PATH,
    BASELINE_SUMMARY_CSV_PATH,
    PLOTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("plot_baselines")

# Curated palette and markers for publication-quality RD curves
CODEC_STYLES = {
    "JPEG": {"color": "#1f77b4", "marker": "o", "label": "JPEG (DCT)"},
    "JPEG2000": {"color": "#ff7f0e", "marker": "s", "label": "JPEG 2000 (DWT)"},
    "PNG": {"color": "#2ca02c", "marker": "^", "label": "PNG (Lossless)"},
}


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate means grouped by codec and setting."""
    # Convert inf to nan for arithmetic means of numeric lossy values if needed
    df_calc = df.copy()

    # Aggregate means
    agg_dict = {
        "bpp": "mean",
        "compression_ratio": "mean",
        "psnr_full": lambda x: float("inf") if (x == float("inf")).all() else x[~np.isinf(x)].mean(),
        "ssim_full": "mean",
        "psnr_roi": lambda x: float("inf") if (x == float("inf")).all() else x[~np.isinf(x)].mean(),
        "psnr_background": lambda x: float("inf") if (x == float("inf")).all() else x[~np.isinf(x)].mean(),
        "roi_lossless": "mean",
        "encode_time_s": "mean",
        "decode_time_s": "mean",
    }

    summary = df_calc.groupby(["codec", "setting"], as_index=False).agg(agg_dict)

    # Sort logically
    def sort_key(row):
        codec = row["codec"]
        setting = row["setting"]
        if codec == "JPEG":
            return (0, int(setting))
        if codec == "JPEG2000":
            return (1, -int(setting))  # Lower ratio = higher bpp
        return (2, 0)

    summary["_sort"] = summary.apply(sort_key, axis=1)
    summary = summary.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    # Round columns for clean reporting
    for col in ["bpp", "compression_ratio", "ssim_full"]:
        summary[col] = summary[col].round(4)
    for col in ["psnr_full", "psnr_roi", "psnr_background"]:
        summary[col] = summary[col].apply(
            lambda v: "inf" if np.isinf(v) else f"{v:.2f}"
        )
    summary["roi_lossless"] = (summary["roi_lossless"] * 100).round(1).astype(str) + "%"
    for col in ["encode_time_s", "decode_time_s"]:
        summary[col] = summary[col].round(5)

    return summary


def plot_rate_distortion(
    df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
) -> None:
    """Generate and save publication-quality rate-distortion curves."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Compute group averages for plotting
    grouped = df.groupby(["codec", "setting"]).agg({
        "bpp": "mean",
        "psnr_full": lambda s: np.nan if (s == float("inf")).all() else s[~np.isinf(s)].mean(),
        "ssim_full": "mean",
        "psnr_roi": lambda s: np.nan if (s == float("inf")).all() else s[~np.isinf(s)].mean(),
    }).reset_index()

    # 1. Rate-Distortion: bpp vs PSNR full
    plt.figure(figsize=(8, 5.5), dpi=300)
    for codec, group in grouped.groupby("codec"):
        sorted_group = group.sort_values("bpp")
        style = CODEC_STYLES.get(codec, {"color": "black", "marker": "o", "label": codec})

        if codec == "PNG":
            # PNG is lossless (PSNR is inf); plot vertical reference line
            png_bpp = sorted_group["bpp"].values[0]
            plt.axvline(
                x=png_bpp,
                color=style["color"],
                linestyle="--",
                linewidth=1.8,
                label=f"{style['label']} (bpp={png_bpp:.2f}, PSNR=∞)",
            )
        else:
            plt.plot(
                sorted_group["bpp"],
                sorted_group["psnr_full"],
                marker=style["marker"],
                color=style["color"],
                linewidth=2.0,
                markersize=7,
                label=style["label"],
            )

    plt.xlabel("Bitrate (bits per pixel - bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("PSNR (dB)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion Curve: Full Image PSNR vs bpp", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()
    psnr_plot_path = plots_dir / "rd_curve_psnr.png"
    plt.savefig(psnr_plot_path)
    plt.close()
    logger.info("Saved %s", psnr_plot_path)

    # 2. Rate-Distortion: bpp vs SSIM full
    plt.figure(figsize=(8, 5.5), dpi=300)
    for codec, group in grouped.groupby("codec"):
        sorted_group = group.sort_values("bpp")
        style = CODEC_STYLES.get(codec, {"color": "black", "marker": "o", "label": codec})

        plt.plot(
            sorted_group["bpp"],
            sorted_group["ssim_full"],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            markersize=7,
            label=style["label"],
        )

    plt.xlabel("Bitrate (bits per pixel - bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Structural Similarity Index (SSIM)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion Curve: Full Image SSIM vs bpp", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()
    ssim_plot_path = plots_dir / "rd_curve_ssim.png"
    plt.savefig(ssim_plot_path)
    plt.close()
    logger.info("Saved %s", ssim_plot_path)

    # 3. Rate-Distortion: bpp vs ROI-only PSNR
    plt.figure(figsize=(8, 5.5), dpi=300)
    for codec, group in grouped.groupby("codec"):
        sorted_group = group.sort_values("bpp")
        style = CODEC_STYLES.get(codec, {"color": "black", "marker": "o", "label": codec})

        if codec == "PNG":
            png_bpp = sorted_group["bpp"].values[0]
            plt.axvline(
                x=png_bpp,
                color=style["color"],
                linestyle="--",
                linewidth=1.8,
                label=f"{style['label']} (bpp={png_bpp:.2f}, ROI=Lossless)",
            )
        else:
            plt.plot(
                sorted_group["bpp"],
                sorted_group["psnr_roi"],
                marker=style["marker"],
                color=style["color"],
                linewidth=2.0,
                markersize=7,
                label=f"{style['label']} (ROI)",
            )

    plt.xlabel("Bitrate (bits per pixel - bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("ROI PSNR (dB)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion Curve: Lung ROI PSNR vs bpp", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()
    roi_psnr_plot_path = plots_dir / "rd_curve_roi_psnr.png"
    plt.savefig(roi_psnr_plot_path)
    plt.close()
    logger.info("Saved %s", roi_psnr_plot_path)


def generate_plots_and_summary(
    csv_path: Path = BASELINES_CSV_PATH,
    summary_path: Path = BASELINE_SUMMARY_CSV_PATH,
    plots_dir: Path = PLOTS_DIR,
) -> pd.DataFrame:
    """Read baseline results, compute summary table, and generate RD plots."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline results not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Parse string inf to float inf
    df["psnr_full"] = pd.to_numeric(df["psnr_full"], errors="coerce").fillna(float("inf"))
    df["psnr_roi"] = pd.to_numeric(df["psnr_roi"], errors="coerce").fillna(float("inf"))
    df["psnr_background"] = pd.to_numeric(df["psnr_background"], errors="coerce").fillna(float("inf"))

    # Compute summary
    summary_df = compute_summary_table(df)

    # Save summary with retry
    for attempt in range(5):
        try:
            summary_df.to_csv(summary_path, index=False)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(1.0)
            else:
                summary_df.to_csv(summary_path.parent / "baseline_summary_updated.csv", index=False)

    logger.info("Saved summary table to %s", summary_path)

    # Generate plots
    plot_rate_distortion(df, plots_dir=plots_dir)

    return summary_df


if __name__ == "__main__":
    summary_table = generate_plots_and_summary()
    print("\n" + "=" * 80)
    print("BASELINE CODECS BENCHMARK SUMMARY (Mean across images)")
    print("=" * 80)
    print(summary_table.to_string(index=False))
    print("=" * 80 + "\n")
