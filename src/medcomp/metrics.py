"""Quality and compression metrics for medical image compression evaluation.

Functions included:
- mse: Mean Squared Error
- psnr: Peak Signal-to-Noise Ratio (inf on identical images)
- ssim: Structural Similarity Index (via scikit-image)
- compression_ratio: Original bytes / compressed bytes
- bits_per_pixel: Total bits / total pixels
- shannon_entropy: Empirical Shannon entropy in bits/symbol
- masked_psnr: PSNR computed over ROI or background regions
- is_lossless: Exact equality check over whole image or ROI mask
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .config import DEFAULT_DATA_RANGE


def mse(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute the Mean Squared Error (MSE) between original and reconstructed images.

    Args:
        orig: Original image array.
        recon: Reconstructed image array.

    Returns:
        float: Mean squared error.

    Raises:
        ValueError: If array shapes do not match.
    """
    if orig.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: orig {orig.shape} vs recon {recon.shape}"
        )
    diff = orig.astype(np.float64) - recon.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(
    orig: np.ndarray,
    recon: np.ndarray,
    data_range: float = DEFAULT_DATA_RANGE,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    For identical images (MSE = 0), returns float('inf').

    Args:
        orig: Original image array.
        recon: Reconstructed image array.
        data_range: Dynamic range of pixel values (default: 255.0 for 8-bit).

    Returns:
        float: PSNR value in dB, or infinity if images are identical.

    Raises:
        ValueError: If array shapes do not match or data_range <= 0.
    """
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")

    err = mse(orig, recon)
    if err == 0.0:
        return float("inf")

    return float(10.0 * np.log10((data_range ** 2) / err))


def ssim(
    orig: np.ndarray,
    recon: np.ndarray,
    data_range: float = DEFAULT_DATA_RANGE,
) -> float:
    """Compute Structural Similarity Index (SSIM) between two 2D images.

    Uses `skimage.metrics.structural_similarity`.

    Args:
        orig: Original 2D image array.
        recon: Reconstructed 2D image array.
        data_range: Dynamic range of pixel values (default: 255.0 for 8-bit).

    Returns:
        float: SSIM index in [-1, 1], with 1.0 indicating identical structure.

    Raises:
        ValueError: If shapes do not match or arrays are not 2D.
    """
    if orig.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: orig {orig.shape} vs recon {recon.shape}"
        )
    if orig.ndim != 2:
        raise ValueError(f"SSIM expects 2D grayscale images, got {orig.ndim}D")

    # If arrays are identical, SSIM is 1.0 by definition
    if np.array_equal(orig, recon):
        return 1.0

    return float(
        structural_similarity(
            orig,
            recon,
            data_range=data_range,
        )
    )


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Compute the compression ratio: (original byte size) / (compressed byte size).

    A ratio > 1 indicates data reduction (e.g., 4.0 means 4:1 compression).

    Args:
        original_bytes: Uncompressed file or raw pixel byte count.
        compressed_bytes: Compressed bitstream byte count.

    Returns:
        float: Compression ratio.

    Raises:
        ValueError: If compressed_bytes <= 0 or original_bytes < 0.
    """
    if compressed_bytes <= 0:
        raise ValueError(
            f"compressed_bytes must be positive, got {compressed_bytes}"
        )
    if original_bytes < 0:
        raise ValueError(
            f"original_bytes cannot be negative, got {original_bytes}"
        )
    return float(original_bytes / compressed_bytes)


def bits_per_pixel(compressed_bytes: int, num_pixels: int) -> float:
    """Compute average bits per pixel (bpp).

    Formula: (compressed_bytes * 8) / num_pixels

    Args:
        compressed_bytes: Size of compressed data in bytes.
        num_pixels: Total number of pixels in image (H * W).

    Returns:
        float: Rate in bits per pixel (bpp).

    Raises:
        ValueError: If num_pixels <= 0 or compressed_bytes < 0.
    """
    if num_pixels <= 0:
        raise ValueError(f"num_pixels must be positive, got {num_pixels}")
    if compressed_bytes < 0:
        raise ValueError(
            f"compressed_bytes cannot be negative, got {compressed_bytes}"
        )
    return float((compressed_bytes * 8.0) / num_pixels)


def shannon_entropy(array: np.ndarray) -> float:
    """Compute empirical Shannon entropy in bits per symbol.

    Formula: H = -sum(p_i * log2(p_i)) where p_i is empirical frequency.

    Args:
        array: Input numpy array of symbols/pixel values (any shape).

    Returns:
        float: Entropy in bits/symbol. Returns 0.0 for uniform constant arrays or empty.
    """
    arr = np.asarray(array)
    if arr.size == 0:
        return 0.0

    _, counts = np.unique(arr, return_counts=True)
    if counts.size <= 1:
        return 0.0

    probs = counts / arr.size
    # Filter out zero probabilities (theoretically not present in unique counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def masked_psnr(
    orig: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray,
    region: str = "roi",
    data_range: float = DEFAULT_DATA_RANGE,
) -> float:
    """Compute PSNR selectively over ROI or background regions defined by a mask.

    Args:
        orig: Original image array.
        recon: Reconstructed image array.
        mask: Binary/boolean mask where True (> 0) denotes ROI pixels.
        region: Target region to evaluate: 'roi' or 'background'.
        data_range: Peak signal value (default: 255.0).

    Returns:
        float: PSNR in dB for the selected region, or inf if identical.

    Raises:
        ValueError: If shapes mismatch, region is invalid, or region has 0 pixels.
    """
    if orig.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: orig {orig.shape} vs recon {recon.shape}"
        )
    if orig.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: orig {orig.shape} vs mask {mask.shape}"
        )

    norm_region = region.strip().lower()
    if norm_region == "roi":
        sel = mask.astype(bool)
    elif norm_region == "background":
        sel = ~mask.astype(bool)
    else:
        raise ValueError(
            f"Invalid region '{region}'. Must be either 'roi' or 'background'."
        )

    num_pixels = np.count_nonzero(sel)
    if num_pixels == 0:
        raise ValueError(f"Mask region '{region}' contains 0 pixels.")

    orig_sel = orig[sel].astype(np.float64)
    recon_sel = recon[sel].astype(np.float64)

    region_mse = float(np.mean((orig_sel - recon_sel) ** 2))
    if region_mse == 0.0:
        return float("inf")

    return float(10.0 * np.log10((data_range ** 2) / region_mse))


def is_lossless(
    orig: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray | None = None,
) -> bool:
    """Check if reconstruction is strictly lossless using exact equality.

    If mask is provided, checks equality only on the ROI region (where mask is True).

    Args:
        orig: Original image array.
        recon: Reconstructed image array.
        mask: Optional binary/boolean mask of same shape.

    Returns:
        bool: True if pixel values are strictly equal, False otherwise.

    Raises:
        ValueError: If shapes mismatch.
    """
    if orig.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: orig {orig.shape} vs recon {recon.shape}"
        )

    if mask is None:
        return bool(np.array_equal(orig, recon))

    if orig.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: orig {orig.shape} vs mask {mask.shape}"
        )

    sel = mask.astype(bool)
    return bool(np.array_equal(orig[sel], recon[sel]))
