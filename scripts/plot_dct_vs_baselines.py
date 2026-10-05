"""Generate comparative rate-distortion plots between custom DCT codec and standard baselines.

Plots saved to results/plots/:
- rd_dct_vs_baselines_psnr.png (bpp vs PSNR)
- rd_dct_vs_baselines_ssim.png (bpp vs SSIM)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medcomp.config import (
    BASELINES_CSV_PATH,
    DCT_RESULTS_CSV_PATH,
    PLOTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("plot_dct_vs_baselines")


def plot_comparative_curves(
    dct_csv: Path = DCT_RESULTS_CSV_PATH,
    baselines_csv: Path = BASELINES_CSV_PATH,
    plots_dir: Path = PLOTS_DIR,
) -> None:
    """Read DCT and baseline results and plot comparative RD curves."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not dct_csv.exists():
        raise FileNotFoundError(f"DCT results not found at {dct_csv}. Run run_dct.py first.")
    if not baselines_csv.exists():
        raise FileNotFoundError(f"Baselines results not found at {baselines_csv}. Run run_baselines.py first.")

    df_dct = pd.read_csv(dct_csv)
    df_base = pd.read_csv(baselines_csv)

    # Convert inf to nan for lossy plotting
    df_base["psnr_full"] = pd.to_numeric(df_base["psnr_full"], errors="coerce")

    # Group DCT by quality
    dct_grouped = df_dct.groupby("quality").agg({
        "bpp": "mean",
        "psnr": "mean",
        "ssim": "mean",
    }).sort_values("bpp").reset_index()

    # Group baselines by codec & setting
    base_grouped = df_base.groupby(["codec", "setting"]).agg({
        "bpp": "mean",
        "psnr_full": lambda s: np.nan if (s == float("inf")).all() else s[~np.isinf(s)].mean(),
        "ssim_full": "mean",
    }).reset_index()

    jpeg_grouped = base_grouped[base_grouped["codec"] == "JPEG"].sort_values("bpp")
    j2k_grouped = base_grouped[base_grouped["codec"] == "JPEG2000"].sort_values("bpp")
    png_row = base_grouped[base_grouped["codec"] == "PNG"]

    # -------------------------------------------------------------
    # 1. Rate-Distortion: bpp vs PSNR
    # -------------------------------------------------------------
    plt.figure(figsize=(8.5, 6), dpi=300)

    # Custom DCT
    plt.plot(
        dct_grouped["bpp"],
        dct_grouped["psnr"],
        marker="o",
        color="#d62728",
        linewidth=2.2,
        markersize=7,
        label="Custom DCT (Scratch Huffman)",
    )

    # Pillow JPEG
    plt.plot(
        jpeg_grouped["bpp"],
        jpeg_grouped["psnr_full"],
        marker="s",
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        markersize=6,
        label="Pillow JPEG (Standard Huffman)",
    )

    # JPEG2000
    plt.plot(
        j2k_grouped["bpp"],
        j2k_grouped["psnr_full"],
        marker="^",
        color="#ff7f0e",
        linestyle="-.",
        linewidth=2.0,
        markersize=6,
        label="JPEG 2000 (DWT)",
    )

    if not png_row.empty:
        png_bpp = png_row["bpp"].values[0]
        plt.axvline(
            x=png_bpp,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.8,
            label=f"PNG Lossless (bpp={png_bpp:.2f}, PSNR=∞)",
        )

    plt.xlabel("Bitrate (bits per pixel - bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("PSNR (dB)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion: Custom DCT vs Standard Baselines (PSNR)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()

    psnr_plot_path = plots_dir / "rd_dct_vs_baselines_psnr.png"
    plt.savefig(psnr_plot_path)
    plt.close()
    logger.info("Saved %s", psnr_plot_path)

    # -------------------------------------------------------------
    # 2. Rate-Distortion: bpp vs SSIM
    # -------------------------------------------------------------
    plt.figure(figsize=(8.5, 6), dpi=300)

    # Custom DCT
    plt.plot(
        dct_grouped["bpp"],
        dct_grouped["ssim"],
        marker="o",
        color="#d62728",
        linewidth=2.2,
        markersize=7,
        label="Custom DCT (Scratch Huffman)",
    )

    # Pillow JPEG
    plt.plot(
        jpeg_grouped["bpp"],
        jpeg_grouped["ssim_full"],
        marker="s",
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        markersize=6,
        label="Pillow JPEG (Standard Huffman)",
    )

    # JPEG2000
    plt.plot(
        j2k_grouped["bpp"],
        j2k_grouped["ssim_full"],
        marker="^",
        color="#ff7f0e",
        linestyle="-.",
        linewidth=2.0,
        markersize=6,
        label="JPEG 2000 (DWT)",
    )

    if not png_row.empty:
        png_bpp = png_row["bpp"].values[0]
        plt.plot(
            [png_bpp],
            [1.0],
            marker="*",
            color="#2ca02c",
            markersize=11,
            label=f"PNG Lossless (bpp={png_bpp:.2f}, SSIM=1.0)",
        )

    plt.xlabel("Bitrate (bits per pixel - bpp)", fontsize=11, fontweight="bold")
    plt.ylabel("Structural Similarity Index (SSIM)", fontsize=11, fontweight="bold")
    plt.title("Rate-Distortion: Custom DCT vs Standard Baselines (SSIM)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()

    ssim_plot_path = plots_dir / "rd_dct_vs_baselines_ssim.png"
    plt.savefig(ssim_plot_path)
    plt.close()
    logger.info("Saved %s", ssim_plot_path)


if __name__ == "__main__":
    plot_comparative_curves()
    print("\nGenerated comparative plots in results/plots/.")
