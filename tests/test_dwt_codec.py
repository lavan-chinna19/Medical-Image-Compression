"""Unit tests for the custom DWT image codec and entropy coding integration."""

from __future__ import annotations

import numpy as np
import pytest

from medcomp.config import DATA_PROCESSED_IMAGES_DIR
from medcomp.dwt_codec import dwt_decode, dwt_encode
from medcomp.io_utils import list_images, load_image
from medcomp.metrics import psnr


def test_dwt_decode_equals_encoder_recon_bit_exact() -> None:
    """Verify that standalone dwt_decode matches the encoder's internal recon bit-for-bit."""
    image_paths = list_images(DATA_PROCESSED_IMAGES_DIR)
    assert len(image_paths) > 0, "No processed images found for testing."

    img = load_image(image_paths[0])
    for quality in [20, 50, 80]:
        data, recon = dwt_encode(img, quality=quality, wavelet="bior4.4", levels=4)
        decoded = dwt_decode(data)

        assert decoded.shape == img.shape
        assert np.array_equal(decoded, recon), f"Decoded image differed from recon at quality={quality}"


def test_dwt_non_multiple_of_2_levels_dimensions() -> None:
    """Verify codec handles arbitrary non-multiple dimensions such as 67x93."""
    rng = np.random.RandomState(42)
    odd_img = rng.randint(0, 256, (67, 93), dtype=np.uint8)

    data, recon = dwt_encode(odd_img, quality=50, wavelet="bior4.4", levels=4)
    decoded = dwt_decode(data)

    assert decoded.shape == (67, 93)
    assert np.array_equal(decoded, recon)


def test_dwt_constant_and_zero_images() -> None:
    """Verify codec correctly handles homogeneous (constant and all-zero) images."""
    for val in [0, 128, 255]:
        img = np.full((128, 128), val, dtype=np.uint8)
        data, recon = dwt_encode(img, quality=50, wavelet="bior4.4", levels=4)
        decoded = dwt_decode(data)

        assert decoded.shape == (128, 128)
        assert np.array_equal(decoded, recon)


def test_dwt_size_decreases_as_quality_decreases() -> None:
    """Verify that compressed size decreases monotonically as quality decreases."""
    image_paths = list_images(DATA_PROCESSED_IMAGES_DIR)
    img = load_image(image_paths[0])

    qualities = [90, 70, 50, 30, 10]
    sizes: list[int] = []

    for q in qualities:
        data, _ = dwt_encode(img, quality=q, wavelet="bior4.4", levels=4)
        sizes.append(len(data))

    for i in range(len(sizes) - 1):
        assert sizes[i] > sizes[i + 1], (
            f"Expected size at quality {qualities[i]} ({sizes[i]}) > "
            f"size at quality {qualities[i+1]} ({sizes[i+1]})"
        )


def test_dwt_psnr_quality_90_above_38db() -> None:
    """Verify that PSNR at quality 90 exceeds 38 dB on real processed chest X-ray images."""
    image_paths = list_images(DATA_PROCESSED_IMAGES_DIR)
    img = load_image(image_paths[0])

    data, recon = dwt_encode(img, quality=90, wavelet="bior4.4", levels=4)
    decoded = dwt_decode(data)

    recon_psnr = psnr(img, decoded)
    assert recon_psnr > 38.0, f"Expected PSNR > 38.0 dB at Q=90, got {recon_psnr:.2f} dB"


def test_dwt_output_is_deterministic() -> None:
    """Verify that encoding the same image multiple times produces identical bitstreams."""
    image_paths = list_images(DATA_PROCESSED_IMAGES_DIR)
    img = load_image(image_paths[0])

    data1, recon1 = dwt_encode(img, quality=50, wavelet="bior4.4", levels=4)
    data2, recon2 = dwt_encode(img, quality=50, wavelet="bior4.4", levels=4)

    assert data1 == data2, "Bitstreams were not identical across consecutive runs."
    assert np.array_equal(recon1, recon2), "Reconstructions were not identical."


@pytest.mark.parametrize("wavelet", ["haar", "bior2.2", "bior4.4", "db2"])
@pytest.mark.parametrize("levels", [3, 4, 5])
def test_dwt_wavelet_and_levels_combinations(wavelet: str, levels: int) -> None:
    """Verify roundtrip bit-exactness across all four wavelets and levels 3, 4, 5."""
    rng = np.random.RandomState(42)
    test_img = rng.randint(0, 256, (128, 128), dtype=np.uint8)

    data, recon = dwt_encode(test_img, quality=50, wavelet=wavelet, levels=levels)
    decoded = dwt_decode(data)

    assert decoded.shape == (128, 128)
    assert np.array_equal(decoded, recon)


def test_dwt_per_level_tables_roundtrip() -> None:
    """Verify roundtrip with per_level_tables=True option."""
    image_paths = list_images(DATA_PROCESSED_IMAGES_DIR)
    img = load_image(image_paths[0])

    data, recon = dwt_encode(img, quality=50, wavelet="bior4.4", levels=4, per_level_tables=True)
    decoded = dwt_decode(data)

    assert np.array_equal(decoded, recon)
