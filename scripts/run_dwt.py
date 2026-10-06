"""Benchmark custom DWT image codec on the processed medical dataset.

Evaluates:
- Wavelet: bior4.4
- Levels: 4
- Qualities: [10, 20, 30, 50, 70, 90]
- Metrics:
    filename, wavelet, levels, quality, compressed_bytes, compression_ratio, bpp, psnr, ssim,
    ll_entropy, ll_avg_code_length, detail_entropy, detail_avg_code_length,
    encode_time_s, decode_time_s
Outputs:
- results/dwt_results.csv (Per-image, per-quality detailed results)
- results/dwt_summary.csv (Aggregated mean metrics by quality)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medcomp.config import (
    DATA_PROCESSED_IMAGES_DIR,
    DWT_RESULTS_CSV_PATH,
    DWT_SUMMARY_CSV_PATH,
)
from medcomp.dwt_codec import dwt_decode, dwt_encode_with_stats
from medcomp.io_utils import list_images, load_image
from medcomp.metrics import bits_per_pixel, compression_ratio, psnr, ssim

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_dwt")

DWT_QUALITIES = [10, 20, 30, 50, 70, 90]
DEFAULT_WAVELET = "bior4.4"
DEFAULT_LEVELS = 4


def run_dwt_benchmarks(
    images_dir: Path = DATA_PROCESSED_IMAGES_DIR,
    results_path: Path = DWT_RESULTS_CSV_PATH,
    summary_path: Path = DWT_SUMMARY_CSV_PATH,
    qualities: list[int] = DWT_QUALITIES,
    wavelet: str = DEFAULT_WAVELET,
    levels: int = DEFAULT_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute DWT compression evaluation across all preprocessed images."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    images = list_images(images_dir)
    if not images:
        logger.error("No images found in %s. Run preprocess_all.py first.", images_dir)
        return pd.DataFrame(), pd.DataFrame()

    records: list[dict[str, Any]] = []

    for img_path in tqdm(images, desc="Running custom DWT codec", unit="img"):
        img = load_image(img_path)
        orig_bytes = img.nbytes
        num_pixels = img.size

        for q in qualities:
            t0 = time.perf_counter()
            comp_bytes, recon_enc, stats = dwt_encode_with_stats(
                img, quality=q, wavelet=wavelet, levels=levels
            )
            enc_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            recon_dec = dwt_decode(comp_bytes)
            dec_time = time.perf_counter() - t0

            if not np.array_equal(recon_enc, recon_dec):
                logger.error("Encoder / Decoder mismatch on %s at q=%d", img_path.name, q)

            n_bytes = len(comp_bytes)
            cr = compression_ratio(orig_bytes, n_bytes)
            bpp_val = bits_per_pixel(n_bytes, num_pixels)
            p_val = psnr(img, recon_dec)
            s_val = ssim(img, recon_dec)

            records.append({
                "filename": img_path.name,
                "wavelet": wavelet,
                "levels": levels,
                "quality": q,
                "compressed_bytes": n_bytes,
                "compression_ratio": round(cr, 4),
                "bpp": round(bpp_val, 4),
                "psnr": round(p_val, 4),
                "ssim": round(s_val, 4),
                "ll_entropy": stats["ll_entropy"],
                "ll_avg_code_length": stats["ll_avg_code_length"],
                "detail_entropy": stats["detail_entropy"],
                "detail_avg_code_length": stats["detail_avg_code_length"],
                "encode_time_s": round(enc_time, 6),
                "decode_time_s": round(dec_time, 6),
            })

    df_results = pd.DataFrame(records)
    df_results.to_csv(results_path, index=False)
    logger.info("Saved full results to %s (%d rows)", results_path, len(df_results))

    # Aggregated summary by quality
    numeric_cols = [
        "compressed_bytes",
        "compression_ratio",
        "bpp",
        "psnr",
        "ssim",
        "ll_entropy",
        "ll_avg_code_length",
        "detail_entropy",
        "detail_avg_code_length",
        "encode_time_s",
        "decode_time_s",
    ]
    df_summary = (
        df_results.groupby("quality")[numeric_cols]
        .mean()
        .reset_index()
    )
    df_summary = df_summary.round({
        "compressed_bytes": 1,
        "compression_ratio": 4,
        "bpp": 4,
        "psnr": 4,
        "ssim": 4,
        "ll_entropy": 4,
        "ll_avg_code_length": 4,
        "detail_entropy": 4,
        "detail_avg_code_length": 4,
        "encode_time_s": 5,
        "decode_time_s": 5,
    })
    df_summary.to_csv(summary_path, index=False)
    logger.info("Saved aggregated summary to %s", summary_path)

    return df_results, df_summary


if __name__ == "__main__":
    _, summary = run_dwt_benchmarks()
    print("\n" + "=" * 80)
    print("DWT CODEC (bior4.4, levels=4) SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))
