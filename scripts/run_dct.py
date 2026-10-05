"""Benchmark custom DCT image codec on the processed medical dataset.

Evaluates:
- Qualities: [10, 20, 30, 50, 70, 90]
- Metrics:
    filename, quality, compressed_bytes, compression_ratio, bpp, psnr, ssim,
    dc_entropy, dc_avg_code_length, ac_entropy, ac_avg_code_length,
    encode_time_s, decode_time_s
Outputs:
- results/dct_results.csv (Per-image, per-quality detailed results)
- results/dct_summary.csv (Aggregated mean metrics by quality)
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
    DCT_RESULTS_CSV_PATH,
    DCT_SUMMARY_CSV_PATH,
)
from medcomp.dct_codec import dct_decode, dct_encode_with_stats
from medcomp.io_utils import list_images, load_image
from medcomp.metrics import bits_per_pixel, compression_ratio, psnr, ssim

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_dct")

DCT_QUALITIES = [10, 20, 30, 50, 70, 90]


def run_dct_benchmarks(
    images_dir: Path = DATA_PROCESSED_IMAGES_DIR,
    results_path: Path = DCT_RESULTS_CSV_PATH,
    summary_path: Path = DCT_SUMMARY_CSV_PATH,
    qualities: list[int] = DCT_QUALITIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute DCT compression evaluation across all preprocessed images."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    images = list_images(images_dir)
    if not images:
        logger.error("No images found in %s. Run preprocess_all.py first.", images_dir)
        return pd.DataFrame(), pd.DataFrame()

    records: list[dict[str, Any]] = []

    for img_path in tqdm(images, desc="Running custom DCT codec", unit="img"):
        img = load_image(img_path)
        orig_bytes = img.nbytes
        num_pixels = img.size

        for q in qualities:
            t0 = time.perf_counter()
            comp_bytes, recon_enc, stats = dct_encode_with_stats(img, quality=q)
            enc_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            recon_dec = dct_decode(comp_bytes)
            dec_time = time.perf_counter() - t0

            # Verify decoder matches encoder recon
            if not np.array_equal(recon_enc, recon_dec):
                logger.error("Encoder / Decoder mismatch on %s at q=%d", img_path.name, q)

            n_bytes = len(comp_bytes)
            cr = compression_ratio(orig_bytes, n_bytes)
            bpp_val = bits_per_pixel(n_bytes, num_pixels)
            p_val = psnr(img, recon_dec)
            s_val = ssim(img, recon_dec)

            records.append({
                "filename": img_path.name,
                "quality": q,
                "compressed_bytes": n_bytes,
                "compression_ratio": round(cr, 4),
                "bpp": round(bpp_val, 4),
                "psnr": round(p_val, 4),
                "ssim": round(s_val, 4),
                "dc_entropy": stats["dc_entropy"],
                "dc_avg_code_length": stats["dc_avg_code_length"],
                "ac_entropy": stats["ac_entropy"],
                "ac_avg_code_length": stats["ac_avg_code_length"],
                "encode_time_s": round(enc_time, 6),
                "decode_time_s": round(dec_time, 6),
            })

    df_results = pd.DataFrame(records)

    # Compute summary means by quality
    summary_cols = [
        "compressed_bytes",
        "compression_ratio",
        "bpp",
        "psnr",
        "ssim",
        "dc_entropy",
        "dc_avg_code_length",
        "ac_entropy",
        "ac_avg_code_length",
        "encode_time_s",
        "decode_time_s",
    ]
    df_summary = df_results.groupby("quality", as_index=False)[summary_cols].mean()

    # Format summary for readability
    for c in ["compressed_bytes"]:
        df_summary[c] = df_summary[c].round(1)
    for c in ["compression_ratio", "bpp", "psnr", "ssim", "dc_entropy", "dc_avg_code_length", "ac_entropy", "ac_avg_code_length"]:
        df_summary[c] = df_summary[c].round(4)
    for c in ["encode_time_s", "decode_time_s"]:
        df_summary[c] = df_summary[c].round(5)

    # Save CSVs with retry loop
    for path, df in [(results_path, df_results), (summary_path, df_summary)]:
        for attempt in range(5):
            try:
                df.to_csv(path, index=False)
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(1.0)
                else:
                    alt_path = path.parent / f"{path.stem}_alt.csv"
                    df.to_csv(alt_path, index=False)
                    logger.warning("Primary %s locked, saved to %s", path, alt_path)

    logger.info("Saved detailed results to %s (%d rows)", results_path, len(df_results))
    logger.info("Saved summary table to %s (%d rows)", summary_path, len(df_summary))
    return df_results, df_summary


if __name__ == "__main__":
    _, summary = run_dct_benchmarks()
    print("\n" + "=" * 80)
    print("CUSTOM DCT CODEC BENCHMARK SUMMARY (Mean across images)")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")
