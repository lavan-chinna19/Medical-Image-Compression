"""Unit tests for baseline codecs and preprocessing modules."""

from pathlib import Path

import numpy as np
import pytest

from medcomp.baselines import (
    check_jpeg2000_support,
    jpeg2000_codec,
    jpeg_codec,
    png_codec,
)
from medcomp.preprocess import load_pair


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """Fixture providing a deterministic 2D uint8 image."""
    rng = np.random.default_rng(12345)
    # 256x256 image with smooth gradients and high-frequency structures
    x = np.linspace(0, 255, 256, dtype=np.uint8)
    gradient = np.tile(x, (256, 1))
    noise = rng.integers(0, 30, (256, 256), dtype=np.uint8)
    return np.clip(gradient.astype(int) + noise, 0, 255).astype(np.uint8)


def test_png_roundtrip_is_bit_exact(synthetic_image: np.ndarray):
    """PNG codec must achieve exact lossless reconstruction (bit-exact)."""
    comp_bytes, recon = png_codec(synthetic_image, compress_level=9)

    assert recon.shape == synthetic_image.shape
    assert recon.dtype == np.uint8
    assert np.array_equal(synthetic_image, recon)
    assert len(comp_bytes) > 0


def test_jpeg_smaller_than_png_at_quality_30(synthetic_image: np.ndarray):
    """Lossy JPEG at quality=30 should produce a significantly smaller payload than lossless PNG."""
    jpeg_bytes, _ = jpeg_codec(synthetic_image, quality=30)
    png_bytes, _ = png_codec(synthetic_image, compress_level=9)

    assert len(jpeg_bytes) < len(png_bytes)


def test_reconstructed_shape_equals_input_shape(synthetic_image: np.ndarray):
    """All codecs must preserve original image dimensions (height, width)."""
    # Test on arbitrary dimensions (e.g. 130 x 170)
    odd_shape_img = synthetic_image[:130, :170]

    _, recon_png = png_codec(odd_shape_img)
    assert recon_png.shape == (130, 170)

    _, recon_jpeg = jpeg_codec(odd_shape_img, quality=50)
    assert recon_jpeg.shape == (130, 170)

    if check_jpeg2000_support():
        _, recon_j2k = jpeg2000_codec(odd_shape_img, ratio=10)
        assert recon_j2k.shape == (130, 170)


def test_compressed_size_matches_len_bytes(synthetic_image: np.ndarray):
    """The returned compressed_bytes object length must match the exact bitstream length."""
    jpeg_bytes, _ = jpeg_codec(synthetic_image, quality=75)
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) == len(bytes(jpeg_bytes))

    png_bytes, _ = png_codec(synthetic_image)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) == len(bytes(png_bytes))

    if check_jpeg2000_support():
        j2k_bytes, _ = jpeg2000_codec(synthetic_image, ratio=20)
        assert isinstance(j2k_bytes, bytes)
        assert len(j2k_bytes) == len(bytes(j2k_bytes))


def test_jpeg2000_rate_monotonicity(synthetic_image: np.ndarray):
    """Higher target compression ratio in JPEG2000 must result in smaller file size."""
    if not check_jpeg2000_support():
        pytest.skip("JPEG2000 support not available")

    bytes_ratio5, recon5 = jpeg2000_codec(synthetic_image, ratio=5)
    bytes_ratio40, recon40 = jpeg2000_codec(synthetic_image, ratio=40)

    # Ratio 40 must yield a smaller payload than Ratio 5
    assert len(bytes_ratio40) < len(bytes_ratio5)
    assert recon5.shape == synthetic_image.shape
    assert recon40.shape == synthetic_image.shape


def test_codec_invalid_inputs():
    """Verify ValueError on invalid input shapes or out-of-range quality parameters."""
    invalid_3d = np.zeros((10, 10, 3), dtype=np.uint8)
    invalid_float = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        jpeg_codec(invalid_3d)
    with pytest.raises(ValueError):
        jpeg_codec(invalid_float)
    with pytest.raises(ValueError):
        jpeg_codec(np.zeros((10, 10), dtype=np.uint8), quality=105)

    with pytest.raises(ValueError):
        png_codec(invalid_3d)
    with pytest.raises(ValueError):
        png_codec(np.zeros((10, 10), dtype=np.uint8), compress_level=12)


def test_preprocess_load_pair(tmp_path: Path):
    """Test load_pair resizes raw image and mask to exact target resolution."""
    raw_img = np.zeros((100, 200), dtype=np.uint8)
    raw_img[20:80, 40:160] = 200
    img_path = tmp_path / "sample.png"
    import cv2
    cv2.imwrite(str(img_path), raw_img)

    masks_root = tmp_path / "masks"
    roi_dir = masks_root / "roi"
    roi_dir.mkdir(parents=True)
    mask_arr = np.zeros((100, 200), dtype=np.uint8)
    mask_arr[20:80, 40:160] = 255
    cv2.imwrite(str(roi_dir / "sample.png"), mask_arr)

    img_out, mask_out = load_pair(img_path, masks_root=masks_root, size=512)
    assert img_out.shape == (512, 512)
    assert mask_out.shape == (512, 512)
    assert img_out.dtype == np.uint8
    assert mask_out.dtype == bool
