"""Ablation study script for DWT image codec.

Evaluates on a fixed seeded subset of 20 images at quality 50:
- Wavelets: ["haar", "bior2.2", "bior4.4", "db2"]
- Levels: [3, 4, 5]
- Huffman table option: single-table (False) vs per-level-table (True)

Outputs:
- results/dwt_ablation.csv
- results/plots/dwt_ablation_comparison.png (Bar charts of mean BPP and PSNR)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medcomp.config import (
    DATA_PROCESSED_IMAGES_DIR,
    DWT_ABLATION_CSV_PATH,
    PLOTS_DIR,
)
from medcomp.dwt_codec import dwt_decode, dwt_encode_with_stats
from medcomp.io_utils import list_images, load_image
from medcomp.metrics import bits_per_pixel, compression_ratio, psnr, ssim

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_dwt_ablation")

WAVELETS = ["haar", "bior2.2", "bior4.4", "db2"]
LEVELS_LIST = [3, 4, 5]
TABLE_OPTIONS = [False, True]  # False = single table, True = per-level tables
QUALITY = 50
SUBSET_SIZE = 20
RANDOM_SEED = 42


def run_ablation(
    images_dir: Path = DATA_PROCESSED_IMAGES_DIR,
    csv_path: Path = DWT_ABLATION_CSV_PATH,
    plots_dir: Path = PLOTS_DIR,
) -> pd.DataFrame:
    """Run DWT parameter ablation study and generate plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_images = list_images(images_dir)
    if not all_images:
        logger.error("No images found in %s", images_dir)
        return pd.DataFrame()

    # Seeded subset of 20 images
    rng = np.random.RandomState(RANDOM_SEED)
    subset_indices = rng.choice(len(all_images), size=min(SUBSET_SIZE, len(all_images)), replace=False)
    subset_images = [all_images[i] for i in sorted(subset_indices)]
    logger.info("Selected %d images for ablation study (seed=%d)", len(subset_images), RANDOM_SEED)

    records: list[dict[str, Any]] = []

    # Iterate over all combinations
    total_configs = len(WAVELETS) * len(LEVELS_LIST) * len(TABLE_OPTIONS)
    pbar = tqdm(total=total_configs * len(subset_images), desc="DWT Ablation", unit="run")

    for wav in WAVELETS:
        for lev in LEVELS_LIST:
            for per_lvl in TABLE_OPTIONS:
                for img_path in subset_images:
                    img = load_image(img_path)
                    orig_bytes = img.nbytes
                    num_pixels = img.size

                    t0 = time.perf_counter()
                    comp_bytes, recon_enc, stats = dwt_encode_with_stats(
                        img,
                        quality=QUALITY,
                        wavelet=wav,
                        levels=lev,
                        per_level_tables=per_lvl,
                    )
                    enc_time = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    recon_dec = dwt_decode(comp_bytes)
                    dec_time = time.perf_counter() - t0

                    n_bytes = len(comp_bytes)
                    cr = compression_ratio(orig_bytes, n_bytes)
                    bpp_val = bits_per_pixel(n_bytes, num_pixels)
                    p_val = psnr(img, recon_dec)
                    s_val = ssim(img, recon_dec)

                    records.append({
                        "filename": img_path.name,
                        "wavelet": wav,
                        "levels": lev,
                        "per_level_tables": per_lvl,
                        "table_mode": "per_level" if per_lvl else "single",
                        "compressed_bytes": n_bytes,
                        "compression_ratio": round(cr, 4),
                        "bpp": round(bpp_val, 4),
                        "psnr": round(p_val, 4),
                        "ssim": round(s_val, 4),
                        "encode_time_s": round(enc_time, 6),
                        "decode_time_s": round(dec_time, 6),
                    })
                    pbar.update(1)

    pbar.close()

    df_runs = pd.DataFrame(records)

    # Group by configuration
    group_cols = ["wavelet", "levels", "per_level_tables", "table_mode"]
    metric_cols = [
        "bpp",
        "psnr",
        "ssim",
        "compression_ratio",
        "compressed_bytes",
        "encode_time_s",
        "decode_time_s",
    ]
    df_ablation = df_runs.groupby(group_cols)[metric_cols].mean().reset_index()
    df_ablation = df_ablation.round({
        "bpp": 4,
        "psnr": 4,
        "ssim": 4,
        "compression_ratio": 4,
        "compressed_bytes": 1,
        "encode_time_s": 5,
        "decode_time_s": 5,
    })

    df_ablation.to_csv(csv_path, index=False)
    logger.info("Saved ablation table to %s (%d configurations)", csv_path, len(df_ablation))

    # Generate Bar Chart of Mean BPP and PSNR
    _plot_ablation_bar_chart(df_ablation, plots_dir / "dwt_ablation_comparison.png")

    return df_ablation


def _plot_ablation_bar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create a structured bar chart of mean BPP and PSNR for all ablation configurations."""
    df_plot = df.copy()
    df_plot["config"] = (
        df_plot["wavelet"]
        + " L="
        + df_plot["levels"].astype(str)
        + " ("
        + df_plot["table_mode"]
        + ")"
    )

    # Sort by PSNR descending
    df_plot = df_plot.sort_values(by="psnr", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    x = np.arange(len(df_plot))
    width = 0.65

    # Color palette
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(df_plot)))

    # PSNR subplot
    bars1 = ax1.bar(x, df_plot["psnr"], width, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Mean PSNR (dB)", fontsize=12, fontweight="bold")
    ax1.set_title("DWT Codec Ablation at Q=50: Mean PSNR across Configurations", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(bottom=max(0, df_plot["psnr"].min() - 2), top=df_plot["psnr"].max() + 1.5)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.1,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    # BPP subplot
    bars2 = ax2.bar(x, df_plot["bpp"], width, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Mean Bits Per Pixel (bpp)", fontsize=12, fontweight="bold")
    ax2.set_title("DWT Codec Ablation at Q=50: Mean BPP (Lower is More Compressed)", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_plot["config"], rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(bottom=0, top=df_plot["bpp"].max() * 1.15)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.005,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved ablation comparison chart to %s", output_path)


if __name__ == "__main__":
    ablation_df = run_ablation()
    print("\n" + "=" * 90)
    print("DWT ABLATION RESULTS (Top configurations by PSNR):")
    print("=" * 90)
    print(ablation_df.sort_values(by="psnr", ascending=False).to_string(index=False))
