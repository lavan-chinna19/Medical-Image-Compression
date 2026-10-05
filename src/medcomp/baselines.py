"""Baseline image compression codecs using standardized formats (JPEG, JPEG2000, PNG).

Each codec executes in-memory via `io.BytesIO` using Pillow, measuring exact
compressed byte counts without file I/O overhead.

Codecs provided:
- jpeg_codec: DCT-based baseline lossy compression
- jpeg2000_codec: DWT-based baseline lossy compression (OpenJPEG with rate control)
- png_codec: Deflate-based baseline lossless compression (compress_level=9)
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, features


def check_jpeg2000_support() -> bool:
    """Check whether Pillow was built with JPEG2000 (OpenJPEG) support."""
    return bool(features.check("jpg_2000"))


def jpeg_codec(
    img: np.ndarray,
    quality: int = 75,
) -> tuple[bytes, np.ndarray]:
    """Compress and reconstruct a 2D grayscale image using standard DCT-based JPEG.

    Args:
        img: 2D uint8 numpy array of shape (H, W).
        quality: JPEG compression quality factor in [1, 95] (default: 75).

    Returns:
        tuple[bytes, np.ndarray]:
            - compressed_bytes: The exact JPEG bitstream bytes.
            - reconstructed: 2D uint8 reconstructed image array.

    Raises:
        ValueError: If input is not a 2D uint8 array or quality is invalid.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(
            f"Expected 2D uint8 array, got shape={getattr(img, 'shape', None)}, "
            f"dtype={getattr(img, 'dtype', None)}"
        )
    if not (1 <= quality <= 100):
        raise ValueError(f"JPEG quality must be between 1 and 100, got {quality}")

    pil_img = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    comp_bytes = buf.getvalue()

    buf.seek(0)
    with Image.open(buf) as decoded:
        recon = np.array(decoded.convert("L"), dtype=np.uint8)

    return comp_bytes, recon


def jpeg2000_codec(
    img: np.ndarray,
    ratio: int | float = 10,
) -> tuple[bytes, np.ndarray]:
    """Compress and reconstruct a 2D grayscale image using DWT-based JPEG2000.

    Uses OpenJPEG via Pillow with rate-based compression layer targeting `ratio`.

    Args:
        img: 2D uint8 numpy array of shape (H, W).
        ratio: Target compression ratio (e.g., 10 for 10:1 compression).

    Returns:
        tuple[bytes, np.ndarray]:
            - compressed_bytes: The exact JPEG2000 bitstream bytes.
            - reconstructed: 2D uint8 reconstructed image array.

    Raises:
        RuntimeError: If Pillow lacks JPEG2000/OpenJPEG support.
        ValueError: If input is not a 2D uint8 array or ratio <= 0.
    """
    if not check_jpeg2000_support():
        raise RuntimeError(
            "Pillow was compiled without OpenJPEG/JPEG2000 support. "
            "Please ensure libopenjp2 is installed or reinstall Pillow: "
            "pip install --force-reinstall pillow"
        )
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(
            f"Expected 2D uint8 array, got shape={getattr(img, 'shape', None)}, "
            f"dtype={getattr(img, 'dtype', None)}"
        )
    if ratio <= 0:
        raise ValueError(f"Compression ratio must be positive, got {ratio}")

    pil_img = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil_img.save(
        buf,
        format="JPEG2000",
        quality_mode="rates",
        quality_layers=[float(ratio)],
    )
    comp_bytes = buf.getvalue()

    buf.seek(0)
    with Image.open(buf) as decoded:
        recon = np.array(decoded.convert("L"), dtype=np.uint8)

    return comp_bytes, recon


def png_codec(
    img: np.ndarray,
    compress_level: int = 9,
) -> tuple[bytes, np.ndarray]:
    """Compress and reconstruct a 2D grayscale image losslessly using PNG.

    Args:
        img: 2D uint8 numpy array of shape (H, W).
        compress_level: Zlib compression level in [0, 9] (default: 9 for max compression).

    Returns:
        tuple[bytes, np.ndarray]:
            - compressed_bytes: The exact PNG bitstream bytes.
            - reconstructed: 2D uint8 bit-exact reconstructed image array.

    Raises:
        ValueError: If input is not a 2D uint8 array or compress_level is invalid.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(
            f"Expected 2D uint8 array, got shape={getattr(img, 'shape', None)}, "
            f"dtype={getattr(img, 'dtype', None)}"
        )
    if not (0 <= compress_level <= 9):
        raise ValueError(f"compress_level must be between 0 and 9, got {compress_level}")

    pil_img = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", compress_level=compress_level)
    comp_bytes = buf.getvalue()

    buf.seek(0)
    with Image.open(buf) as decoded:
        recon = np.array(decoded.convert("L"), dtype=np.uint8)

    return comp_bytes, recon
