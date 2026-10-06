"""Generate Rate-Distortion curves and side-by-side visual comparisons.

Produces:
1. Rate-Distortion curves:
   - BPP vs PSNR (dB)
   - BPP vs SSIM
   Comparing:
   - Custom DCT Codec (mine)
   - Custom DWT Codec (mine, bior4.4, levels=4)
   - Standard JPEG Baseline (Pillow)
   - Standard JPEG2000 Baseline (OpenJPEG)
2. Visual side-by-side comparison of a representative chest X-ray at ~0.5 bpp:
   - Original
   - Custom DCT Reconstructed
   - Custom DWT Reconstructed

Outputs saved in results/plots/.
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
    BASELINE_SUMMARY_CSV_PATH,
    DATA_PROCESSED_IMAGES_DIR,
    DCT_SUMMARY_CSV_PATH,
    DWT_SUMMARY_CSV_PATH,
    PLOTS_DIR,
)
from medcomp.dct_codec import dct_decode, dct_encode
from medcomp.dwt_codec import dwt_decode, dwt_encode
from medcomp.io_utils import list_images, load_image
from medcomp.metrics import bits_per_pixel, psnr, ssim

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("plot_dct_vs_dwt")


def plot_rate_distortion_curves(
    dct_summary_path: Path = DCT_SUMMARY_CSV_PATH,
    dwt_summary_path: Path = DWT_SUMMARY_CSV_PATH,
    baseline_summary_path: Path = BASELINE_SUMMARY_CSV_PATH,
    plots_dir: Path = PLOTS_DIR,
) -> tuple[Path, Path]:
    """Plot and save BPP vs PSNR and BPP vs SSIM curves across codecs."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Read summaries
    df_dct = pd.read_csv(dct_summary_path)
    df_dwt = pd.read_csv(dwt_summary_path)
    df_base = pd.read_csv(baseline_summary_path)

    df_jpeg = df_base[df_base["codec"] == "JPEG"].copy()
    df_jp2k = df_base[df_base["codec"] == "JPEG2000"].copy()

    # Style mapping
    codec_styles = {
        "Custom DWT (Ours)": {"color": "#1f77b4", "marker": "o", "linewidth": 2.5, "markersize": 8},
        "Custom DCT (Ours)": {"color": "#ff7f0e", "marker": "s", "linewidth": 2.5, "markersize": 8},
        "JPEG2000 (OpenJPEG)": {"color": "#2ca02c", "marker": "^", "linewidth": 2.0, "linestyle": "--", "markersize": 7},
        "JPEG (Standard)": {"color": "#d62728", "marker": "d", "linewidth": 2.0, "linestyle": ":", "markersize": 7},
    }

    # 1. BPP vs PSNR Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(df_dwt["bpp"], df_dwt["psnr"], label="Custom DWT (Ours)", **codec_styles["Custom DWT (Ours)"])
    ax.plot(df_dct["bpp"], df_dct["psnr"], label="Custom DCT (Ours)", **codec_styles["Custom DCT (Ours)"])
    ax.plot(df_jp2k["bpp"], df_jp2k["psnr_full"], label="JPEG2000 (OpenJPEG)", **codec_styles["JPEG2000 (OpenJPEG)"])
    ax.plot(df_jpeg["bpp"], df_jpeg["psnr_full"], label="JPEG (Standard)", **codec_styles["JPEG (Standard)"])

    ax.set_xlabel("Rate: Bits Per Pixel (bpp)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Distortion: Peak SNR (dB)", fontsize=13, fontweight="bold")
    ax.set_title("Rate-Distortion Performance: BPP vs PSNR (Montgomery Dataset)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95)
    plt.tight_layout()

    psnr_plot_path = plots_dir / "rd_bpp_vs_psnr.png"
    fig.savefig(psnr_plot_path, dpi=300)
    plt.close(fig)
    logger.info("Saved PSNR rate-distortion plot to %s", psnr_plot_path)

    # 2. BPP vs SSIM Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(df_dwt["bpp"], df_dwt["ssim"], label="Custom DWT (Ours)", **codec_styles["Custom DWT (Ours)"])
    ax.plot(df_dct["bpp"], df_dct["ssim"], label="Custom DCT (Ours)", **codec_styles["Custom DCT (Ours)"])
    ax.plot(df_jp2k["bpp"], df_jp2k["ssim_full"], label="JPEG2000 (OpenJPEG)", **codec_styles["JPEG2000 (OpenJPEG)"])
    ax.plot(df_jpeg["bpp"], df_jpeg["ssim_full"], label="JPEG (Standard)", **codec_styles["JPEG (Standard)"])

    ax.set_xlabel("Rate: Bits Per Pixel (bpp)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Structural Similarity Index (SSIM)", fontsize=13, fontweight="bold")
    ax.set_title("Rate-Distortion Performance: BPP vs SSIM (Montgomery Dataset)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95)
    plt.tight_layout()

    ssim_plot_path = plots_dir / "rd_bpp_vs_ssim.png"
    fig.savefig(ssim_plot_path, dpi=300)
    plt.close(fig)
    logger.info("Saved SSIM rate-distortion plot to %s", ssim_plot_path)

    return psnr_plot_path, ssim_plot_path


def plot_side_by_side_visual(
    images_dir: Path = DATA_PROCESSED_IMAGES_DIR,
    plots_dir: Path = PLOTS_DIR,
    target_bpp: float = 0.5,
) -> Path:
    """Generate a side-by-side visual comparison of one chest X-ray at ~0.5 bpp: Original vs DCT vs DWT."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No processed images found in {images_dir}")

    # Pick representative first image
    sample_path = images[0]
    img = load_image(sample_path)
    num_pixels = img.size

    # Search for quality setting closest to target_bpp for DCT and DWT
    dct_best: dict[str, Any] = {}
    dwt_best: dict[str, Any] = {}

    for q in range(10, 95, 5):
        # DCT
        d_bytes, d_recon = dct_encode(img, quality=q)
        d_bpp = bits_per_pixel(len(d_bytes), num_pixels)
        if not dct_best or abs(d_bpp - target_bpp) < abs(dct_best["bpp"] - target_bpp):
            dct_best = {
                "quality": q,
                "bytes": len(d_bytes),
                "bpp": d_bpp,
                "psnr": psnr(img, d_recon),
                "ssim": ssim(img, d_recon),
                "recon": d_recon,
            }

        # DWT
        w_bytes, w_recon = dwt_encode(img, quality=q, wavelet="bior4.4", levels=4)
        w_bpp = bits_per_pixel(len(w_bytes), num_pixels)
        if not dwt_best or abs(w_bpp - target_bpp) < abs(dwt_best["bpp"] - target_bpp):
            dwt_best = {
                "quality": q,
                "bytes": len(w_bytes),
                "bpp": w_bpp,
                "psnr": psnr(img, w_recon),
                "ssim": ssim(img, w_recon),
                "recon": w_recon,
            }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # 1. Original
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(
        f"Original: {sample_path.name}\n(Uncompressed 8.00 bpp)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[0].axis("off")

    # 2. Custom DCT
    axes[1].imshow(dct_best["recon"], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"Custom DCT (Ours)\nQ={dct_best['quality']} | {dct_best['bpp']:.3f} bpp\nPSNR: {dct_best['psnr']:.2f} dB | SSIM: {dct_best['ssim']:.4f}",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[1].axis("off")

    # 3. Custom DWT
    axes[2].imshow(dwt_best["recon"], cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(
        f"Custom DWT (bior4.4, L=4)\nQ={dwt_best['quality']} | {dwt_best['bpp']:.3f} bpp\nPSNR: {dwt_best['psnr']:.2f} dB | SSIM: {dwt_best['ssim']:.4f}",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Visual Reconstruction Comparison at ~{target_bpp:.1f} bpp (Montgomery CXR)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    visual_path = plots_dir / "visual_comparison_0.5bpp.png"
    fig.savefig(visual_path, dpi=300)
    plt.close(fig)
    logger.info("Saved visual comparison to %s", visual_path)
    return visual_path


def main() -> None:
    logger.info("Plotting rate-distortion curves and visual comparisons...")
    p_path, s_path = plot_rate_distortion_curves()
    v_path = plot_side_by_side_visual()
    print("\nGenerated plots:")
    print(f"1. Rate-Distortion (PSNR): {p_path}")
    print(f"2. Rate-Distortion (SSIM): {s_path}")
    print(f"3. Visual Comparison:     {v_path}")


if __name__ == "__main__":
    main()
