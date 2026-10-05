"""Image and ROI mask preprocessing pipeline for standardized experiment resolution.

Loads raw images and corresponding anatomical/ROI masks, resizing images using
area-averaging (cv2.INTER_AREA) and masks using nearest-neighbor (cv2.INTER_NEAREST)
to ensure exact spatial alignment at standard resolution (e.g. 512x512).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import DATA_MASKS_DIR, TARGET_SIZE
from .io_utils import load_image
from .mask_utils import get_roi_mask, load_mask, resize_mask_to_image


def load_pair(
    image_path: str | Path,
    masks_root: str | Path = DATA_MASKS_DIR,
    size: int = TARGET_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and its corresponding ROI mask, resizing both to (size, size).

    Resizes the image using `cv2.INTER_AREA` (preventing moiré and downsampling
    aliasing) and the mask using nearest-neighbor interpolation to preserve strict
    binary values and spatial alignment.

    Args:
        image_path: Path to the image file.
        masks_root: Directory containing masks (searches 'roi/', or 'leftmask/' + 'rightmask/').
        size: Target square dimension (default: 512).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - img: 2D uint8 grayscale image of shape (size, size).
            - roi: 2D boolean mask of shape (size, size).

    Raises:
        FileNotFoundError: If image or masks cannot be located.
        ValueError: If size <= 0 or image format is invalid.
    """
    if size <= 0:
        raise ValueError(f"Target size must be positive, got {size}")

    img_path = Path(image_path).resolve()
    root = Path(masks_root).resolve()
    stem = img_path.stem

    # 1. Load image
    img = load_image(img_path)

    # 2. Load ROI mask
    # Check precomputed merged mask in roi/ first
    precomputed_roi = root / "roi" / f"{stem}.png"
    if precomputed_roi.exists():
        roi_mask = load_mask(precomputed_roi)
    else:
        # Fallback to merging left and right masks
        roi_mask = get_roi_mask(img_path, root)

    # 3. Resize image with cv2.INTER_AREA
    if img.shape == (size, size):
        img_resized = img.copy()
    else:
        img_resized = cv2.resize(
            img,
            (size, size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8)

    # 4. Resize mask with nearest-neighbor
    roi_resized = resize_mask_to_image(roi_mask, (size, size))

    return img_resized, roi_resized
