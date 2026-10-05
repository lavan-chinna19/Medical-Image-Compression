"""Image input/output and preprocessing utilities for medical images.

Supports loading standard formats (PNG, JPG, BMP, TIFF) and DICOM (.dcm),
saving 2D uint8 images, directory scanning, and block-aligned padding/cropping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pydicom

from .config import DEFAULT_BLOCK_SIZE, SUPPORTED_EXTENSIONS


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return it as a 2D uint8 grayscale numpy array.

    Supports:
    - Standard formats: .png, .jpg, .jpeg, .bmp, .tif, .tiff
    - DICOM formats: .dcm (applies RescaleSlope, RescaleIntercept, and windowing)

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray: 2D grayscale image array of shape (H, W) with dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is unsupported, corrupt, or cannot be converted to 2D.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    if ext == ".dcm":
        return _load_dicom_image(file_path)
    return _load_standard_image(file_path)


def _load_standard_image(file_path: Path) -> np.ndarray:
    """Read a standard image file and convert to 2D uint8 grayscale."""
    try:
        with open(file_path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        raise ValueError(f"Failed to read image '{file_path}': {e}") from e

    if img is None:
        raise ValueError(f"Failed to decode image file: {file_path}")

    return img.astype(np.uint8)


def _load_dicom_image(file_path: Path) -> np.ndarray:
    """Read a DICOM (.dcm) file, applying rescale slope/intercept and windowing to 8-bit."""
    try:
        dcm = pydicom.dcmread(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to parse DICOM file '{file_path}': {e}") from e

    if not hasattr(dcm, "pixel_array"):
        raise ValueError(f"DICOM file does not contain pixel data: {file_path}")

    arr = dcm.pixel_array.astype(np.float32)

    # If multi-frame or 3D, extract 2D slice or convert
    if arr.ndim == 3:
        if arr.shape[2] in (3, 4):  # RGB / RGBA
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            # Multi-slice volume: take the middle slice
            arr = arr[arr.shape[0] // 2]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported DICOM array dimensions: {arr.shape}")

    # 1. Apply RescaleSlope and RescaleIntercept (Hounsfield units for CT)
    slope = getattr(dcm, "RescaleSlope", 1.0)
    intercept = getattr(dcm, "RescaleIntercept", 0.0)

    try:
        slope = float(slope)
        intercept = float(intercept)
    except (TypeError, ValueError):
        slope = 1.0
        intercept = 0.0

    arr = arr * slope + intercept

    # 2. Windowing to 8-bit [0, 255]
    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth", None)

    if wc is not None and ww is not None:
        # Handle cases where WindowCenter/Width are MultiValue or lists
        if isinstance(wc, (Sequence, pydicom.multival.MultiValue)) and not isinstance(wc, (str, bytes)):
            wc = wc[0]
        if isinstance(ww, (Sequence, pydicom.multival.MultiValue)) and not isinstance(ww, (str, bytes)):
            ww = ww[0]

        try:
            wc = float(wc)
            ww = float(ww)
        except (TypeError, ValueError):
            wc, ww = None, None

    if wc is not None and ww is not None and ww > 0:
        # Standard DICOM windowing formula
        c_min = wc - 0.5 - (ww - 1.0) / 2.0
        c_max = wc - 0.5 + (ww - 1.0) / 2.0
        arr = np.clip(arr, c_min, c_max)
        if c_max > c_min:
            arr = ((arr - c_min) / (c_max - c_min)) * 255.0
        else:
            arr = np.zeros_like(arr)
    else:
        # Fallback to min-max normalization if window metadata is absent
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val > min_val:
            arr = ((arr - min_val) / (max_val - min_val)) * 255.0
        else:
            arr = np.zeros_like(arr)

    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def save_image(path: str | Path, arr: np.ndarray) -> Path:
    """Save a 2D uint8 numpy array to disk as an image.

    Args:
        path: Output file path.
        arr: 2D numpy array with dtype uint8.

    Returns:
        Path: The absolute path to the saved file.

    Raises:
        TypeError: If `arr` is not a numpy array.
        ValueError: If `arr` is not 2D or not uint8, or file cannot be written.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr).__name__}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {arr.dtype}")

    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    ext = target_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension '{ext}' for saving. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Encode to bytes and write to support international/Windows paths cleanly
    success, encoded = cv2.imencode(ext, arr)
    if not success:
        raise ValueError(f"Failed to encode image to '{target_path.name}'")

    target_path.write_bytes(encoded.tobytes())
    return target_path


def list_images(folder: str | Path) -> list[Path]:
    """Scan a folder and return a sorted list of supported image paths.

    Args:
        folder: Directory to search.

    Returns:
        list[Path]: Sorted list of resolved Path objects.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    folder_path = Path(folder).resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    files = [
        p.resolve()
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def pad_to_multiple(
    img: np.ndarray,
    multiple: int = DEFAULT_BLOCK_SIZE,
    mode: str = "edge",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad a 2D image so both height and width are multiples of `multiple`.

    Required for block-based transforms such as 8x8 DCT.

    Args:
        img: 2D numpy array of shape (H, W).
        multiple: Block dimension (default 8).
        mode: Padding mode (default 'edge' to minimize boundary discontinuities).

    Returns:
        tuple[np.ndarray, tuple[int, int]]:
            - padded_img: Padded 2D array of shape (padded_H, padded_W).
            - orig_shape: Tuple (H, W) of the original image dimensions.

    Raises:
        ValueError: If img is not 2D or multiple <= 0.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={getattr(img, 'ndim', None)}")
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")

    h, w = img.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    orig_shape = (h, w)
    if pad_h == 0 and pad_w == 0:
        return img.copy(), orig_shape

    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode=mode)
    return padded, orig_shape


def crop_to_shape(
    img: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Crop an image back to its original target dimensions.

    Inverse helper for `pad_to_multiple`.

    Args:
        img: 2D numpy array of shape (padded_H, padded_W).
        target_shape: Tuple (orig_H, orig_W) of desired dimensions.

    Returns:
        np.ndarray: Cropped 2D array of shape target_shape.

    Raises:
        ValueError: If img is not 2D or target_shape exceeds current dimensions.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={getattr(img, 'ndim', None)}")

    orig_h, orig_w = target_shape
    if orig_h < 0 or orig_w < 0:
        raise ValueError(f"Target shape dimensions must be >= 0: {target_shape}")
    if orig_h > img.shape[0] or orig_w > img.shape[1]:
        raise ValueError(
            f"Target shape {target_shape} exceeds current image shape {img.shape}"
        )

    return img[:orig_h, :orig_w].copy()


# Convenience alias for inverse crop
unpad_image = crop_to_shape
