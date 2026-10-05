"""Unit tests for medcomp mask_utils module."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from medcomp.mask_utils import (
    bounding_box,
    get_roi_mask,
    load_mask,
    merge_masks,
    resize_mask_to_image,
    roi_fraction,
)


def test_merge_masks_synthetic():
    """Verify logical OR combination of synthetic left and right masks."""
    # 20x20 canvas: left mask occupies cols 2..8, right mask occupies cols 12..18
    left = np.zeros((20, 20), dtype=bool)
    left[4:16, 2:8] = True

    right = np.zeros((20, 20), dtype=bool)
    right[4:16, 12:18] = True

    merged = merge_masks(left, right)
    assert merged.dtype == bool
    assert merged.shape == (20, 20)

    # Check non-overlapping union
    assert np.all(merged[4:16, 2:8])
    assert np.all(merged[4:16, 12:18])
    assert not np.any(merged[4:16, 8:12])  # Middle separator remains background

    # Test single-mask fallback
    assert np.array_equal(merge_masks(left, None), left)
    assert np.array_equal(merge_masks(None, right), right)

    # Test shape mismatch error
    with pytest.raises(ValueError, match="shape mismatch"):
        merge_masks(left, np.zeros((10, 10), dtype=bool))


def test_resize_mask_preserves_binary_values():
    """Verify nearest-neighbor resizing strictly preserves binary (bool) values without artifacts."""
    orig_mask = np.zeros((20, 20), dtype=bool)
    orig_mask[5:15, 5:15] = True  # 10x10 square

    # Resize up to (50, 60)
    upscaled = resize_mask_to_image(orig_mask, (50, 60))
    assert upscaled.shape == (50, 60)
    assert upscaled.dtype == bool
    assert set(np.unique(upscaled)).issubset({False, True})

    # Resize down to (8, 10)
    downscaled = resize_mask_to_image(orig_mask, (8, 10))
    assert downscaled.shape == (8, 10)
    assert downscaled.dtype == bool
    assert set(np.unique(downscaled)).issubset({False, True})

    # Resize to identical shape returns exact array
    same_shape = resize_mask_to_image(orig_mask, (20, 20))
    assert np.array_equal(same_shape, orig_mask)


def test_roi_fraction_known():
    """Verify ROI fraction computation on known geometry."""
    # 100x100 = 10,000 pixels. A 50x50 block contains 2,500 pixels (fraction = 0.25)
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True

    frac = roi_fraction(mask)
    assert frac == pytest.approx(0.25)

    # All-zero mask
    assert roi_fraction(np.zeros((50, 50), dtype=bool)) == 0.0

    # All-one mask
    assert roi_fraction(np.ones((50, 50), dtype=bool)) == 1.0


def test_bounding_box_known_rectangle():
    """Verify bounding box coordinates (r0, r1, c0, c1) for a known rectangle."""
    mask = np.zeros((100, 100), dtype=bool)
    # Rectangle spanning rows [20, 60) and cols [30, 80)
    mask[20:60, 30:80] = True

    r0, r1, c0, c1 = bounding_box(mask)
    assert (r0, r1, c0, c1) == (20, 60, 30, 80)

    # Sub-slice must match the active rectangle
    assert mask[r0:r1, c0:c1].all()
    assert (r1 - r0, c1 - c0) == (40, 50)

    # Empty mask returns (0, 0, 0, 0)
    assert bounding_box(np.zeros((40, 40), dtype=bool)) == (0, 0, 0, 0)


def test_load_mask(tmp_path: Path):
    """Test load_mask reads image and thresholds > 127 to boolean."""
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:20, 10:20] = 255
    img[0:5, 0:5] = 100  # Below 127 threshold

    mask_file = tmp_path / "test_mask.png"
    cv2.imwrite(str(mask_file), img)

    loaded = load_mask(mask_file, threshold=127)
    assert loaded.shape == (30, 30)
    assert loaded.dtype == bool
    assert np.all(loaded[10:20, 10:20])
    assert not np.any(loaded[0:5, 0:5])


def test_get_roi_mask(tmp_path: Path):
    """Test get_roi_mask finds left and right masks by stem ignoring extensions."""
    masks_root = tmp_path / "masks"
    left_dir = masks_root / "leftmask"
    right_dir = masks_root / "rightmask"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)

    # Synthetic left and right masks for "patient_001"
    left_img = np.zeros((40, 40), dtype=np.uint8)
    left_img[5:35, 5:18] = 255
    cv2.imwrite(str(left_dir / "patient_001.png"), left_img)

    right_img = np.zeros((40, 40), dtype=np.uint8)
    right_img[5:35, 22:35] = 255
    cv2.imwrite(str(right_dir / "patient_001.png"), right_img)

    # Image path with different extension (.jpg) to test stem matching
    image_path = tmp_path / "raw" / "patient_001.jpg"

    combined = get_roi_mask(image_path, masks_root)
    assert combined.shape == (40, 40)
    assert combined.dtype == bool
    assert np.all(combined[5:35, 5:18])
    assert np.all(combined[5:35, 22:35])
    assert not np.any(combined[5:35, 18:22])
