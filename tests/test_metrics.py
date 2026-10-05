"""Unit tests for medcomp metrics module."""

import numpy as np
import pytest

from medcomp.metrics import (
    bits_per_pixel,
    compression_ratio,
    is_lossless,
    masked_psnr,
    mse,
    psnr,
    shannon_entropy,
    ssim,
)


def test_identical_images_psnr_and_ssim():
    """Identical images must yield PSNR = inf and SSIM = 1.0."""
    rng = np.random.default_rng(42)
    orig = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    recon = orig.copy()

    assert mse(orig, recon) == 0.0
    val_psnr = psnr(orig, recon)
    assert np.isinf(val_psnr) and val_psnr > 0
    assert ssim(orig, recon) == 1.0


def test_known_mse_and_psnr():
    """Verify MSE and PSNR on a synthetic pair with known analytical error."""
    orig = np.zeros((10, 10), dtype=np.uint8)
    recon = np.full((10, 10), 3, dtype=np.uint8)

    # Difference is 3 everywhere; MSE must be 3^2 = 9.0
    calculated_mse = mse(orig, recon)
    assert calculated_mse == pytest.approx(9.0)

    # Expected PSNR: 10 * log10(255^2 / 9) = 10 * log10(7225) ≈ 38.588 dB
    expected_psnr = 10.0 * np.log10((255.0 ** 2) / 9.0)
    calculated_psnr = psnr(orig, recon, data_range=255.0)
    assert calculated_psnr == pytest.approx(expected_psnr, rel=1e-5)


def test_entropy_constant_image():
    """Entropy of an image with identical pixel values must be exactly 0.0."""
    constant_img = np.full((128, 128), 173, dtype=np.uint8)
    assert shannon_entropy(constant_img) == 0.0

    # Also test empty array edge case
    assert shannon_entropy(np.array([], dtype=np.uint8)) == 0.0


def test_entropy_uniform_8bit():
    """Entropy of uniform 8-bit symbols must equal 8.0 bits/symbol.

    Tested both analytically (exact uniform histogram) and empirically
    (random uniform distribution across 0..255).
    """
    # 1. Exact uniform representation (each byte 0..255 appears 100 times)
    exact_uniform = np.repeat(np.arange(256, dtype=np.uint8), 100)
    assert shannon_entropy(exact_uniform) == pytest.approx(8.0, abs=1e-9)

    # 2. Random uniform 8-bit image with large sample
    rng = np.random.default_rng(12345)
    random_img = rng.integers(0, 256, size=(500, 500), dtype=np.uint8)
    # With 250,000 samples, entropy should be very close to 8 bits
    assert shannon_entropy(random_img) == pytest.approx(8.0, abs=0.02)


def test_is_lossless_cases():
    """Test is_lossless for full-image and masked ROI evaluations."""
    orig = np.zeros((20, 20), dtype=np.uint8)
    recon = orig.copy()

    # 1. Identical -> lossless
    assert is_lossless(orig, recon) is True

    # 2. Single pixel altered -> lossy
    recon[5, 5] = 1
    assert is_lossless(orig, recon) is False

    # 3. Masked ROI evaluation
    # Define ROI mask covering top-left 10x10 block
    mask = np.zeros((20, 20), dtype=bool)
    mask[:10, :10] = True

    # Alter background only (pixel at 15, 15)
    recon_bg_altered = orig.copy()
    recon_bg_altered[15, 15] = 99

    # Whole image is not lossless
    assert is_lossless(orig, recon_bg_altered) is False
    # But ROI region IS lossless!
    assert is_lossless(orig, recon_bg_altered, mask=mask) is True

    # Alter inside ROI (pixel at 2, 2)
    recon_roi_altered = orig.copy()
    recon_roi_altered[2, 2] = 1
    assert is_lossless(orig, recon_roi_altered, mask=mask) is False


def test_compression_ratio_and_bpp():
    """Test compression ratio and bits per pixel formulas."""
    orig_bytes = 512 * 512  # 262,144 bytes (8-bit 512x512)
    comp_bytes = 65536      # 4:1 compression

    cr = compression_ratio(orig_bytes, comp_bytes)
    assert cr == pytest.approx(4.0)

    # 65536 bytes * 8 bits / (512 * 512) pixels = 2.0 bpp
    bpp = bits_per_pixel(comp_bytes, 512 * 512)
    assert bpp == pytest.approx(2.0)

    # Error handling for invalid inputs
    with pytest.raises(ValueError):
        compression_ratio(orig_bytes, 0)
    with pytest.raises(ValueError):
        bits_per_pixel(comp_bytes, 0)


def test_masked_psnr():
    """Test masked PSNR for ROI and background regions."""
    orig = np.full((10, 10), 100, dtype=np.uint8)
    recon = orig.copy()

    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True  # Top half is ROI

    # Alter background only (bottom half)
    recon[5:, :] = 103

    # ROI should be identical -> inf PSNR
    assert np.isinf(masked_psnr(orig, recon, mask, region="roi"))

    # Background has difference 3 -> 10 * log10(255^2 / 9) ≈ 38.588 dB
    expected_bg_psnr = 10.0 * np.log10((255.0 ** 2) / 9.0)
    assert masked_psnr(orig, recon, mask, region="background") == pytest.approx(
        expected_bg_psnr, rel=1e-5
    )

    # Invalid region raises ValueError
    with pytest.raises(ValueError, match="Invalid region"):
        masked_psnr(orig, recon, mask, region="invalid_region")


def test_shape_mismatch_errors():
    """Ensure dimension mismatch raises ValueError across metrics."""
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 12), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError, match="Shape mismatch"):
        mse(a, b)
    with pytest.raises(ValueError, match="Shape mismatch"):
        psnr(a, b)
    with pytest.raises(ValueError, match="Shape mismatch"):
        ssim(a, b)
    with pytest.raises(ValueError, match="Shape mismatch"):
        is_lossless(a, b)
    with pytest.raises(ValueError, match="Shape mismatch"):
        masked_psnr(a, b, mask)
