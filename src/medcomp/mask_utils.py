"""Mask utilities for medical image Region of Interest (ROI) handling.

Provides functions to load binary masks, merge multi-part anatomical masks
(e.g., left and right lung masks), retrieve ROI masks by image stem,
resize masks using nearest-neighbor interpolation, and extract ROI metrics
(ROI fraction, bounding box).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Load a mask image from disk and convert to a 2D boolean array.

    Pixels with intensity strictly greater than `threshold` are mapped to True.

    Args:
        path: Path to the mask file.
        threshold: Pixel intensity threshold (default 127).

    Returns:
        np.ndarray: 2D boolean array where True indicates ROI.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If file is not a valid 2D image.
    """
    mask_path = Path(path).resolve()
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    if not mask_path.is_file():
        raise ValueError(f"Mask path is not a file: {mask_path}")

    try:
        with open(mask_path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        raise ValueError(f"Failed to read mask '{mask_path}': {e}") from e

    if img is None:
        raise ValueError(f"Failed to decode mask image: {mask_path}")

    if img.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {img.shape}")

    return img > threshold


def merge_masks(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> np.ndarray:
    """Merge two binary masks using logical OR.

    Supports cases where one of the masks may be absent (None).

    Args:
        left: Left mask 2D array, or None.
        right: Right mask 2D array, or None.

    Returns:
        np.ndarray: 2D boolean mask representing the combined ROI.

    Raises:
        ValueError: If both masks are None, or shapes mismatch.
    """
    if left is None and right is None:
        raise ValueError("At least one mask must be provided.")

    if left is None:
        return np.asarray(right, dtype=bool)

    if right is None:
        return np.asarray(left, dtype=bool)

    if left.shape != right.shape:
        raise ValueError(
            f"Mask shape mismatch: left {left.shape} vs right {right.shape}"
        )

    return np.logical_or(left.astype(bool), right.astype(bool))


def resize_mask_to_image(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D boolean mask to target dimensions using nearest-neighbor interpolation.

    Nearest-neighbor interpolation preserves strict binary values without smoothing.

    Args:
        mask: 2D boolean or integer mask array.
        shape: Desired output dimensions as (height, width).

    Returns:
        np.ndarray: Resized 2D boolean mask.

    Raises:
        ValueError: If mask is not 2D or target dimensions are invalid.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got ndim={getattr(mask, 'ndim', None)}")

    target_h, target_w = shape
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Target dimensions must be positive, got {shape}")

    if mask.shape == (target_h, target_w):
        return mask.astype(bool)

    # OpenCV resize expects dsize as (width, height)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def roi_fraction(mask: np.ndarray) -> float:
    """Compute the fraction of ROI (True) pixels relative to the total number of pixels.

    Formula: count(ROI) / total_pixels

    Args:
        mask: Binary mask array.

    Returns:
        float: ROI fraction in the range [0.0, 1.0].
    """
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find the minimal bounding box enclosing all ROI (True) pixels.

    Returns coordinates in half-open Python slice format (r0, r1, c0, c1)
    such that `mask[r0:r1, c0:c1]` extracts the exact bounding box.

    Args:
        mask: 2D boolean mask array.

    Returns:
        tuple[int, int, int, int]: (r0, r1, c0, c1). Returns (0, 0, 0, 0)
        if mask has no ROI pixels.

    Raises:
        ValueError: If mask is not a 2D array.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got ndim={getattr(mask, 'ndim', None)}")

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)

    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]

    r0 = int(r_idx[0])
    r1 = int(r_idx[-1] + 1)
    c0 = int(c_idx[0])
    c1 = int(c_idx[-1] + 1)

    return (r0, r1, c0, c1)


def _find_matching_mask_file(folder: Path, stem: str) -> Path | None:
    """Find a file in folder whose filename stem matches `stem`, ignoring extension."""
    if not folder.exists() or not folder.is_dir():
        return None

    stem_lower = stem.lower()
    for item in folder.iterdir():
        if item.is_file() and item.stem.lower() == stem_lower:
            return item
    return None


def get_roi_mask(
    image_path: str | Path,
    masks_root: str | Path,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Retrieve and combine the left and right lung masks for a given image.

    Locates corresponding mask files in `<masks_root>/leftmask/` and
    `<masks_root>/rightmask/` by matching the image filename stem, ignoring extension.

    Args:
        image_path: Path to the image file.
        masks_root: Root directory containing 'leftmask' and 'rightmask' subdirectories.
        target_shape: Optional target (height, width) to automatically resize the mask.

    Returns:
        np.ndarray: Combined 2D boolean mask.

    Raises:
        FileNotFoundError: If neither left nor right mask exists.
    """
    img_path = Path(image_path)
    root = Path(masks_root).resolve()
    stem = img_path.stem

    left_dir = root / "leftmask"
    right_dir = root / "rightmask"

    left_file = _find_matching_mask_file(left_dir, stem)
    right_file = _find_matching_mask_file(right_dir, stem)

    if left_file is None and right_file is None:
        raise FileNotFoundError(
            f"No matching left or right mask found for stem '{stem}' in {root}"
        )

    left_mask = load_mask(left_file) if left_file is not None else None
    right_mask = load_mask(right_file) if right_file is not None else None

    # Handle shape alignment if both exist but have differing shapes
    if left_mask is not None and right_mask is not None:
        if left_mask.shape != right_mask.shape:
            logger.warning(
                "Left mask shape %s != right mask shape %s for %s. Resizing right mask.",
                left_mask.shape,
                right_mask.shape,
                stem,
            )
            right_mask = resize_mask_to_image(right_mask, left_mask.shape)

    merged = merge_masks(left_mask, right_mask)

    if target_shape is not None and merged.shape != target_shape:
        merged = resize_mask_to_image(merged, target_shape)

    return merged
