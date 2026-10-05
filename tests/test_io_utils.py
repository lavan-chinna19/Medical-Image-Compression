"""Unit tests for medcomp io_utils module."""

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import pytest

from medcomp.io_utils import (
    crop_to_shape,
    list_images,
    load_image,
    pad_to_multiple,
    save_image,
    unpad_image,
)


def test_pad_to_multiple_and_crop():
    """Verify block padding to multiples of 8 and lossless unpadding."""
    rng = np.random.default_rng(42)
    # Odd dimension (13 x 27) -> padded should be (16 x 32)
    img = rng.integers(0, 256, size=(13, 27), dtype=np.uint8)

    padded, orig_shape = pad_to_multiple(img, multiple=8)
    assert orig_shape == (13, 27)
    assert padded.shape == (16, 32)
    assert padded.dtype == np.uint8

    # Inverse crop should restore exact original content
    restored = crop_to_shape(padded, orig_shape)
    assert restored.shape == (13, 27)
    assert np.array_equal(restored, img)

    # Test alias unpad_image
    restored_alias = unpad_image(padded, orig_shape)
    assert np.array_equal(restored_alias, img)


def test_pad_already_multiple():
    """If image dimensions are already divisible by 8, shape is unchanged."""
    img = np.zeros((16, 24), dtype=np.uint8)
    padded, orig_shape = pad_to_multiple(img, multiple=8)
    assert orig_shape == (16, 24)
    assert padded.shape == (16, 24)
    assert np.array_equal(padded, img)


def test_save_and_load_standard_images(tmp_path: Path):
    """Test saving and loading PNG, BMP, TIFF (lossless) and JPG formats."""
    rng = np.random.default_rng(99)
    img = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)

    for ext in (".png", ".bmp", ".tif"):
        file_path = tmp_path / f"test_img{ext}"
        saved_path = save_image(file_path, img)
        assert saved_path.exists()

        loaded = load_image(file_path)
        assert loaded.shape == (32, 32)
        assert loaded.dtype == np.uint8
        assert np.array_equal(loaded, img)

    # JPEG is lossy, so check shapes and uint8 type
    jpg_path = tmp_path / "test_img.jpg"
    save_image(jpg_path, img)
    loaded_jpg = load_image(jpg_path)
    assert loaded_jpg.shape == (32, 32)
    assert loaded_jpg.dtype == np.uint8


def test_list_images(tmp_path: Path):
    """Test listing and filtering supported image files in a folder."""
    # Create supported and unsupported files
    (tmp_path / "b_scan.png").touch()
    (tmp_path / "a_scan.jpg").touch()
    (tmp_path / "c_scan.dcm").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "summary.csv").touch()
    sub_dir = tmp_path / "subfolder"
    sub_dir.mkdir()

    images = list_images(tmp_path)
    file_names = [p.name for p in images]

    assert file_names == ["a_scan.jpg", "b_scan.png", "c_scan.dcm"]


def test_dicom_loading_with_windowing(tmp_path: Path):
    """Test loading a synthetic DICOM file with slope, intercept, and windowing."""
    dcm_path = tmp_path / "synthetic.dcm"

    # Create DICOM metadata
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(dcm_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "OT"
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Rows = 32
    ds.Columns = 32
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # CT-like values (Hounsfield units)
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.WindowCenter = 40.0
    ds.WindowWidth = 400.0

    # Raw pixel values: stored = HU - intercept = HU + 1024
    raw_pixels = np.linspace(800, 1500, 32 * 32, dtype=np.uint16).reshape((32, 32))
    ds.PixelData = raw_pixels.tobytes()

    ds.save_as(str(dcm_path), enforce_file_format=True)

    # Load with medcomp io_utils
    loaded = load_image(dcm_path)
    assert loaded.shape == (32, 32)
    assert loaded.dtype == np.uint8
    # Ensure windowing transformed to valid dynamic range
    assert loaded.min() >= 0
    assert loaded.max() <= 255
