# Medical Image Compression with Lossless ROI Preservation

A university data compression project implementing and evaluating medical image compression pipelines (DCT, DWT, and Entropy Coding) with strict lossless preservation of clinician-selected Regions of Interest (ROI).

---

## Project Structure

```text
medcomp/
├── src/medcomp/
│   ├── __init__.py      # Package exports and version
│   ├── io_utils.py      # Image I/O (PNG, JPG, BMP, TIFF, DICOM) and padding/crop helpers
│   ├── mask_utils.py    # Binary mask loader, merger, NN-resizer, ROI fraction, bounding box
│   ├── metrics.py       # Quality (MSE, PSNR, SSIM, Masked PSNR) and rate metrics (CR, bpp, Entropy)
│   └── config.py        # Project paths and configuration constants
├── data/
│   ├── raw/             # Raw input medical images (place .dcm, .png, etc. here)
│   └── masks/           # Anatomical masks
│       ├── leftmask/    # Left lung masks
│       ├── rightmask/   # Right lung masks
│       └── roi/         # Generated merged ROI masks (0 and 255)
├── scripts/
│   └── prepare_masks.py # Preprocesses masks, logs warnings, creates previews & CSV report
├── tests/
│   ├── test_metrics.py  # Unit tests for PSNR, SSIM, Entropy, Lossless checks, Masked PSNR
│   ├── test_mask_utils.py # Unit tests for mask merging, resizing, bbox, ROI fraction
│   └── test_io_utils.py # Unit tests for format loading, DICOM windowing, pad/unpad
├── results/             # Compression output artifacts, mask previews, and evaluation reports
│   ├── mask_previews/   # Image overlays with ROI boundary outlined in red
│   └── mask_report.csv  # Summary report of all images and masks
├── pytest.ini           # Pytest test configuration
├── requirements.txt     # Python package dependencies
├── README.md            # Project documentation and setup guide
└── .gitignore           # Git ignore rules for virtual environments, caches, and datasets
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10+)

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running Tests

Run the full test suite using `pytest`:

```bash
pytest -v
```

To run a specific test file:
```bash
pytest tests/test_metrics.py -v
pytest tests/test_io_utils.py -v
```

---

## Phase 1 Modules Overview

### `medcomp.io_utils`
- `load_image(path)`: Loads grayscale images in standard formats (`.png`, `.jpg`, `.bmp`, `.tif`) and DICOM (`.dcm`). For DICOM, it applies `RescaleSlope`, `RescaleIntercept` (Hounsfield Units for CT), and maps to an 8-bit window (`[0, 255] uint8`).
- `save_image(path, arr)`: Saves 2D uint8 images.
- `list_images(folder)`: Returns a sorted list of supported images in a directory.
- `pad_to_multiple(img, multiple=8)`: Pads 2D images with edge-replication to multiples of 8 for DCT block compatibility.
- `crop_to_shape(img, target_shape)` / `unpad_image`: Inverts padding back to the original dimensions.

### `medcomp.mask_utils`
- `load_mask(path, threshold=127)`: Loads mask image and returns a 2D boolean array (`img > threshold`).
- `merge_masks(left, right)`: Logical OR union between left and right masks (handles single mask if one is None).
- `get_roi_mask(image_path, masks_root)`: Locates matching left and right masks by image filename stem, aligns shapes, and merges.
- `resize_mask_to_image(mask, shape)`: Resizes mask with nearest-neighbor interpolation to strictly preserve binary values.
- `roi_fraction(mask)`: Returns ratio of ROI (True) pixels to total pixels (`[0.0, 1.0]`).
- `bounding_box(mask)`: Returns bounding box coordinates `(r0, r1, c0, c1)` in Python slice notation `[r0:r1, c0:c1]`.

### `medcomp.metrics`
- `mse(orig, recon)`: Mean Squared Error.
- `psnr(orig, recon, data_range=255.0)`: Peak Signal-to-Noise Ratio (returns `inf` for identical images).
- `ssim(orig, recon, data_range=255.0)`: Structural Similarity Index.
- `compression_ratio(original_bytes, compressed_bytes)`: Uncompressed to compressed size ratio.
- `bits_per_pixel(compressed_bytes, num_pixels)`: Rate in bits per pixel (bpp).
- `shannon_entropy(array)`: Empirical Shannon entropy in bits/symbol (0.0 for uniform images, ≈ 8.0 for uniform 8-bit).
- `masked_psnr(orig, recon, mask, region="roi"|"background")`: Region-specific PSNR.
- `is_lossless(orig, recon, mask=None)`: Strict exact equality check across the entire image or restricted to an ROI mask.

---

## Mask Preparation Pipeline

To prepare merged masks, generate preview overlays, and produce the summary report:

```powershell
python scripts/prepare_masks.py
```

This outputs:
- Merged binary masks in `data/masks/roi/<image_name>.png` (values 0 and 255)
- Visual previews in `results/mask_previews/<stem>_preview.png` (with ROI outline in red)
- Detailed validation report in `results/mask_report.csv`
