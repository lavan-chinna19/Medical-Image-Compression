"""Script to prepare and preprocess Region of Interest (ROI) masks for raw medical images.

Workflow:
1. Iterates over all images in data/raw/
2. Locates corresponding left and right lung masks in data/masks/leftmask and rightmask
3. Warns if any mask is missing and merges available masks
4. Checks mask dimensions against raw image shape, resizing via nearest-neighbor if needed
5. Saves merged binary ROI mask (values 0 and 255) to data/masks/roi/<image_name>.png
6. Generates red outline overlay previews in results/mask_previews/
7. Compiles a summary report to results/mask_report.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path if executed directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medcomp.config import (
    DATA_MASKS_DIR,
    DATA_RAW_DIR,
    MASKS_LEFT_DIR,
    MASKS_RIGHT_DIR,
    MASKS_ROI_DIR,
    MASK_PREVIEWS_DIR,
    MASK_REPORT_PATH,
    RESULTS_DIR,
)
from medcomp.io_utils import list_images, load_image, save_image
from medcomp.mask_utils import (
    load_mask,
    merge_masks,
    resize_mask_to_image,
    roi_fraction,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("prepare_masks")


def _find_mask_file(folder: Path, stem: str) -> Path | None:
    """Find a file in folder whose stem matches `stem`, case-insensitive."""
    if not folder.exists() or not folder.is_dir():
        return None
    stem_lower = stem.lower()
    for p in folder.iterdir():
        if p.is_file() and p.stem.lower() == stem_lower:
            return p
    return None


def prepare_all_masks(
    raw_dir: Path = DATA_RAW_DIR,
    left_dir: Path = MASKS_LEFT_DIR,
    right_dir: Path = MASKS_RIGHT_DIR,
    roi_out_dir: Path = MASKS_ROI_DIR,
    previews_dir: Path = MASK_PREVIEWS_DIR,
    report_path: Path = MASK_REPORT_PATH,
) -> pd.DataFrame:
    """Process all raw images, merge masks, create overlays, and write CSV report.

    Args:
        raw_dir: Directory containing input raw images.
        left_dir: Directory containing left lung masks.
        right_dir: Directory containing right lung masks.
        roi_out_dir: Output directory for merged ROI masks.
        previews_dir: Output directory for overlay visualization images.
        report_path: Output CSV filepath for mask report.

    Returns:
        pd.DataFrame: Summary dataframe containing report metrics.
    """
    roi_out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    images = list_images(raw_dir)
    if not images:
        logger.warning("No images found in %s", raw_dir)
        return pd.DataFrame()

    records: list[dict[str, str | float]] = []

    for img_path in tqdm(images, desc="Preparing masks", unit="img"):
        stem = img_path.stem
        filename = img_path.name

        # 1. Load image to verify dimensions
        img = load_image(img_path)
        img_h, img_w = img.shape
        img_shape_str = f"({img_h}, {img_w})"

        # 2. Check for left and right masks
        left_file = _find_mask_file(left_dir, stem)
        right_file = _find_mask_file(right_dir, stem)

        missing_parts: list[str] = []
        if left_file is None:
            missing_parts.append("left")
            print(f"[WARNING] Image '{filename}' is missing left mask in {left_dir}")
            logger.warning("Image '%s' is missing left mask in %s", filename, left_dir)
        if right_file is None:
            missing_parts.append("right")
            print(f"[WARNING] Image '{filename}' is missing right mask in {right_dir}")
            logger.warning("Image '%s' is missing right mask in %s", filename, right_dir)

        # 3. Load available masks
        left_mask = load_mask(left_file) if left_file is not None else None
        right_mask = load_mask(right_file) if right_file is not None else None

        if left_mask is None and right_mask is None:
            print(f"[ERROR] Image '{filename}' has no masks available.")
            logger.error("Image '%s' has no masks available.", filename)
            records.append({
                "filename": filename,
                "image_shape": img_shape_str,
                "mask_shape": "N/A",
                "roi_fraction": 0.0,
                "status": "missing_both",
            })
            continue

        # Initial mask shape before alignment/resize
        raw_mask_shape = (
            left_mask.shape if left_mask is not None else right_mask.shape
        )
        mask_shape_str = f"({raw_mask_shape[0]}, {raw_mask_shape[1]})"

        # Align shapes between left and right if both exist
        if left_mask is not None and right_mask is not None:
            if left_mask.shape != right_mask.shape:
                print(
                    f"[WARNING] Shape mismatch between left {left_mask.shape} "
                    f"and right {right_mask.shape} for '{filename}'. Resizing right mask."
                )
                logger.warning(
                    "Shape mismatch between left %s and right %s for '%s'. Resizing right mask.",
                    left_mask.shape,
                    right_mask.shape,
                    filename,
                )
                right_mask = resize_mask_to_image(right_mask, left_mask.shape)

        merged = merge_masks(left_mask, right_mask)

        # 4. Check if merged mask shape matches image shape
        resized_flag = False
        if merged.shape != (img_h, img_w):
            print(
                f"[WARNING] Mask shape {merged.shape} != image shape {(img_h, img_w)} "
                f"for '{filename}'. Resizing mask with nearest-neighbor."
            )
            logger.warning(
                "Mask shape %s != image shape %s for '%s'. Resizing mask with nearest-neighbor.",
                merged.shape,
                (img_h, img_w),
                filename,
            )
            merged = resize_mask_to_image(merged, (img_h, img_w))
            resized_flag = True

        # 5. Compute ROI metrics
        fraction = roi_fraction(merged)

        # Determine status string
        status_items: list[str] = []
        if missing_parts:
            status_items.append(f"missing_{'_and_'.join(missing_parts)}")
        if resized_flag:
            status_items.append("resized")
        if not status_items:
            status = "ok"
        else:
            status = ", ".join(status_items)

        # 6. Save merged ROI mask (values 0 and 255)
        # Use .png extension as required
        roi_filename = f"{stem}.png"
        roi_save_path = roi_out_dir / roi_filename
        roi_uint8 = (merged.astype(np.uint8)) * 255
        save_image(roi_save_path, roi_uint8)

        # 7. Create and save overlay preview (ROI outline in red)
        # Convert grayscale image to 3-channel BGR
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(
            merged.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        # In BGR, red is (0, 0, 255)
        cv2.drawContours(bgr, contours, -1, (0, 0, 255), thickness=2)

        preview_save_path = previews_dir / f"{stem}_preview.png"
        # Encode and save to handle Windows paths
        success, encoded = cv2.imencode(".png", bgr)
        if success:
            preview_save_path.write_bytes(encoded.tobytes())
        else:
            logger.error("Failed to save preview for %s", filename)

        records.append({
            "filename": filename,
            "image_shape": img_shape_str,
            "mask_shape": mask_shape_str,
            "roi_fraction": round(fraction, 6),
            "status": status,
        })

    df = pd.DataFrame(records)
    import time
    for attempt in range(5):
        try:
            df.to_csv(report_path, index=False)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(1.0)
            else:
                # If still locked by external app, write to fallback file
                fallback_path = report_path.parent / "mask_report_updated.csv"
                df.to_csv(fallback_path, index=False)
                logger.warning(
                    "Primary report %s was locked. Saved to fallback %s",
                    report_path,
                    fallback_path,
                )
    logger.info("Saved mask report to %s (%d rows)", report_path, len(df))
    return df


if __name__ == "__main__":
    df_report = prepare_all_masks()
    if not df_report.empty:
        print("\n=== Mask Preparation Summary ===")
        print(f"Total images processed: {len(df_report)}")
        print(f"Status distribution:\n{df_report['status'].value_counts().to_string()}")
        print(f"\nAverage ROI fraction: {df_report['roi_fraction'].mean():.4f}")
        print(f"Report saved to: {MASK_REPORT_PATH}")
