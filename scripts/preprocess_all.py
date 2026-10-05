"""Preprocess all raw images and ROI masks to standardized 512x512 resolution.

Saves:
- Standardized images to data/processed/images/<stem>.png
- Standardized binary masks to data/processed/masks/<stem>.png (values 0 and 255)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to sys.path for direct invocation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medcomp.config import (
    DATA_MASKS_DIR,
    DATA_PROCESSED_IMAGES_DIR,
    DATA_PROCESSED_MASKS_DIR,
    DATA_RAW_DIR,
    TARGET_SIZE,
)
from medcomp.io_utils import list_images, save_image
from medcomp.preprocess import load_pair

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("preprocess_all")


def preprocess_dataset(
    raw_dir: Path = DATA_RAW_DIR,
    masks_root: Path = DATA_MASKS_DIR,
    out_images_dir: Path = DATA_PROCESSED_IMAGES_DIR,
    out_masks_dir: Path = DATA_PROCESSED_MASKS_DIR,
    target_size: int = TARGET_SIZE,
) -> int:
    """Batch preprocess all raw images and masks to target resolution.

    Args:
        raw_dir: Directory containing input raw images.
        masks_root: Directory containing masks.
        out_images_dir: Output directory for processed images.
        out_masks_dir: Output directory for processed masks.
        target_size: Resolution to resize to (target_size x target_size).

    Returns:
        int: Number of image-mask pairs successfully processed.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(raw_dir)
    if not images:
        logger.warning("No raw images found in %s", raw_dir)
        return 0

    count = 0
    for img_path in tqdm(images, desc=f"Preprocessing to {target_size}x{target_size}", unit="img"):
        stem = img_path.stem
        try:
            img_resized, mask_resized = load_pair(
                img_path,
                masks_root=masks_root,
                size=target_size,
            )

            # Save processed image
            img_out_path = out_images_dir / f"{stem}.png"
            save_image(img_out_path, img_resized)

            # Save processed mask (0 and 255)
            mask_out_path = out_masks_dir / f"{stem}.png"
            mask_uint8 = (mask_resized.astype(np.uint8)) * 255
            save_image(mask_out_path, mask_uint8)

            count += 1
        except Exception as e:
            logger.error("Failed to preprocess '%s': %e", img_path.name, e)

    logger.info("Successfully preprocessed %d / %d images to %s", count, len(images), out_images_dir)
    return count


if __name__ == "__main__":
    total = preprocess_dataset()
    print(f"\nCompleted preprocessing {total} image pairs to 512x512.")
