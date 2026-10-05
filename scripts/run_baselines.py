"""Benchmark standard baseline codecs (JPEG, JPEG2000, PNG) on preprocessed medical images.

Evaluates:
- JPEG at quality in [10, 20, 30, 50, 70, 90]
- JPEG2000 at rate layers [5, 10, 20, 40, 80]
- PNG (lossless deflate at compress_level=9)

Records:
- filename, codec, setting, compressed_bytes, compression_ratio, bpp,
  psnr_full, ssim_full, psnr_roi, psnr_background, roi_lossless,
  encode_time_s, decode_time_s
Outputs:
- results/baselines.csv
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medcomp.baselines import check_jpeg2000_support
from medcomp.config import (
    BASELINES_CSV_PATH,
    DATA_PROCESSED_IMAGES_DIR,
    DATA_PROCESSED_MASKS_DIR,
)
from medcomp.io_utils import list_images, load_image
from medcomp.mask_utils import load_mask
from medcomp.metrics import (
    bits_per_pixel,
    compression_ratio,
    is_lossless,
    masked_psnr,
    psnr,
    ssim,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_baselines")

JPEG_QUALITIES = [10, 20, 30, 50, 70, 90]
JPEG2000_RATIOS = [5, 10, 20, 40, 80]


def benchmark_jpeg(
    img: np.ndarray,
    quality: int,
) -> tuple[bytes, np.ndarray, float, float]:
    """Execute in-memory JPEG compression and return (bytes, recon, enc_time, dec_time)."""
    t_start_enc = time.perf_counter()
    pil_img = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    comp_bytes = buf.getvalue()
    enc_time = time.perf_counter() - t_start_enc

    t_start_dec = time.perf_counter()
    buf.seek(0)
    with Image.open(buf) as dec:
        recon = np.array(dec.convert("L"), dtype=np.uint8)
    dec_time = time.perf_counter() - t_start_dec

    return comp_bytes, recon, enc_time, dec_time


def benchmark_jpeg2000(
    img: np.ndarray,
    ratio: int | float,
) -> tuple[bytes, np.ndarray, float, float]:
    """Execute in-memory JPEG2000 compression and return (bytes, recon, enc_time, dec_time)."""
    t_start_enc = time.perf_counter()
    pil_img = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil_img.save(
        buf,
        format="JPEG2000",
        quality_mode="rates",
        quality_layers=[float(ratio)],
    )
    comp_bytes = buf.getvalue()
    enc_time = time.perf_counter() - t_start_enc

    t_start_dec = time.perf_counter()
    buf.seek(0)
    with Image.open(buf) as dec:
        recon = np.array(dec.convert("L"), dtype=np.uint8)
    dec_time = time.perf_counter() - t_start_dec

    return comp_bytes, recon, enc_time, dec_time


def benchmark_png(
    img: np.ndarray,
    compress_level: int = 9,
) -> tuple[bytes, np.ndarray, float, float]:
    """Execute in-memory PNG compression and return (bytes, recon, enc_time, dec_time)."""
    t_start_enc = time.perf_counter()
    pil_img = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=compress_level)
    comp_bytes = buf.getvalue()
    enc_time = time.perf_counter() - t_start_enc

    t_start_dec = time.perf_counter()
    buf.seek(0)
    with Image.open(buf) as dec:
        recon = np.array(dec.convert("L"), dtype=np.uint8)
    dec_time = time.perf_counter() - t_start_dec

    return comp_bytes, recon, enc_time, dec_time


def evaluate_baseline_run(
    img_name: str,
    orig: np.ndarray,
    mask: np.ndarray,
    codec_name: str,
    setting_val: str | int,
    comp_bytes: bytes,
    recon: np.ndarray,
    enc_time: float,
    dec_time: float,
) -> dict[str, str | int | float | bool]:
    """Compute and structure evaluation metrics for a single compression run."""
    orig_bytes = orig.nbytes
    num_pixels = orig.size
    n_bytes = len(comp_bytes)

    cr = compression_ratio(orig_bytes, n_bytes)
    bpp = bits_per_pixel(n_bytes, num_pixels)

    p_full = psnr(orig, recon)
    s_full = ssim(orig, recon)
    p_roi = masked_psnr(orig, recon, mask, region="roi")
    p_bg = masked_psnr(orig, recon, mask, region="background")
    roi_exact = is_lossless(orig, recon, mask=mask)

    return {
        "filename": img_name,
        "codec": codec_name,
        "setting": setting_val,
        "compressed_bytes": n_bytes,
        "compression_ratio": round(cr, 4),
        "bpp": round(bpp, 4),
        "psnr_full": round(p_full, 4) if not np.isinf(p_full) else float("inf"),
        "ssim_full": round(s_full, 4),
        "psnr_roi": round(p_roi, 4) if not np.isinf(p_roi) else float("inf"),
        "psnr_background": round(p_bg, 4) if not np.isinf(p_bg) else float("inf"),
        "roi_lossless": roi_exact,
        "encode_time_s": round(enc_time, 6),
        "decode_time_s": round(dec_time, 6),
    }


def run_all_baselines(
    images_dir: Path = DATA_PROCESSED_IMAGES_DIR,
    masks_dir: Path = DATA_PROCESSED_MASKS_DIR,
    out_csv: Path = BASELINES_CSV_PATH,
) -> pd.DataFrame:
    """Benchmark all baseline codecs on the preprocessed dataset.

    Args:
        images_dir: Directory containing preprocessed 512x512 images.
        masks_dir: Directory containing preprocessed 512x512 masks.
        out_csv: Path to save the output evaluation CSV.

    Returns:
        pd.DataFrame: Compiled benchmark metrics.
    """
    if not check_jpeg2000_support():
        logger.error(
            "Pillow lacks JPEG2000 support. Please reinstall pillow with OpenJPEG."
        )
        raise RuntimeError("Pillow lacks JPEG2000 support.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    images = list_images(images_dir)
    if not images:
        logger.error("No processed images found in %s. Please run preprocess_all.py first.", images_dir)
        return pd.DataFrame()

    results: list[dict[str, Any]] = []

    for img_path in tqdm(images, desc="Running baseline codecs", unit="img"):
        stem = img_path.stem
        mask_path = masks_dir / f"{stem}.png"
        if not mask_path.exists():
            logger.warning("Mask not found for %s, skipping.", img_path.name)
            continue

        orig = load_image(img_path)
        mask = load_mask(mask_path)

        # 1. JPEG runs
        for q in JPEG_QUALITIES:
            comp_bytes, recon, enc_t, dec_t = benchmark_jpeg(orig, quality=q)
            row = evaluate_baseline_run(
                img_path.name, orig, mask, "JPEG", q, comp_bytes, recon, enc_t, dec_t
            )
            results.append(row)

        # 2. JPEG2000 runs
        for r in JPEG2000_RATIOS:
            comp_bytes, recon, enc_t, dec_t = benchmark_jpeg2000(orig, ratio=r)
            row = evaluate_baseline_run(
                img_path.name, orig, mask, "JPEG2000", r, comp_bytes, recon, enc_t, dec_t
            )
            results.append(row)

        # 3. PNG lossless run
        comp_bytes, recon, enc_t, dec_t = benchmark_png(orig, compress_level=9)
        row = evaluate_baseline_run(
            img_path.name, orig, mask, "PNG", "lossless", comp_bytes, recon, enc_t, dec_t
        )
        results.append(row)

    df = pd.DataFrame(results)

    # Save to CSV with retry resilience on Windows
    for attempt in range(5):
        try:
            df.to_csv(out_csv, index=False)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(1.0)
            else:
                fallback = out_csv.parent / "baselines_updated.csv"
                df.to_csv(fallback, index=False)
                logger.warning("Primary %s locked, saved to %s", out_csv, fallback)

    logger.info("Saved %d benchmark results to %s", len(df), out_csv)
    return df


if __name__ == "__main__":
    df_results = run_all_baselines()
    print(f"\nCompleted baselines evaluation: {len(df_results)} rows recorded.")
