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
│   ├── preprocess.py    # Resizes image/ROI pairs to 512x512 with anti-aliasing & alignment
│   ├── entropy.py       # Scratch BitWriter, BitReader, Canonical Huffman, table serialization
│   ├── dct_codec.py     # Custom 8x8 block DCT codec with adaptive Huffman entropy coding
│   ├── baselines.py     # In-memory JPEG, JPEG2000, and PNG baseline codecs
│   ├── metrics.py       # Quality (MSE, PSNR, SSIM, Masked PSNR) and rate metrics (CR, bpp, Entropy)
│   └── config.py        # Project paths and configuration constants (TARGET_SIZE=512)
├── data/
│   ├── raw/             # Raw input medical images (place .dcm, .png, etc. here)
│   ├── masks/           # Anatomical masks (leftmask/, rightmask/, and merged roi/)
│   └── processed/       # Standardized 512x512 dataset (images/ and masks/)
├── scripts/
│   ├── prepare_masks.py # Preprocesses masks, logs warnings, creates previews & CSV report
│   ├── preprocess_all.py# Batch resizes raw images and masks to standardized 512x512
│   ├── run_baselines.py # Benchmarks JPEG, JPEG2000, and PNG across multiple rate points
│   ├── plot_baselines.py# Generates RD curves (PSNR, SSIM, ROI PSNR) and summary CSV
│   ├── run_dct.py       # Benchmarks custom DCT codec across [10, 20, 30, 50, 70, 90]
│   └── plot_dct_vs_baselines.py # Comparative RD plots (Custom DCT vs JPEG vs JPEG2000)
├── tests/
│   ├── test_metrics.py  # Unit tests for PSNR, SSIM, Entropy, Lossless checks, Masked PSNR
│   ├── test_mask_utils.py # Unit tests for mask merging, resizing, bbox, ROI fraction
│   ├── test_baselines.py# Unit tests for baseline codecs, bit-exactness, rate monotonicity
│   ├── test_entropy.py  # Unit tests for BitWriter/Reader, Canonical Huffman, H <= L < H+1
│   ├── test_dct_codec.py# Unit tests for custom DCT codec, bit-exact decode, Q-scaling
│   └── test_io_utils.py # Unit tests for format loading, DICOM windowing, pad/unpad
├── results/             # Compression output artifacts, mask previews, and evaluation reports
│   ├── plots/           # Rate-distortion curves (including rd_dct_vs_baselines_psnr/ssim)
│   ├── baselines.csv    # Raw baseline benchmark data (444 evaluation runs)
│   ├── baseline_summary.csv # Mean baseline rate-distortion table
│   ├── dct_results.csv  # Detailed custom DCT runs across all images and qualities
│   ├── dct_summary.csv  # Mean rate-distortion and entropy stats by quality
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

### `medcomp.entropy`
- `BitWriter`: Accumulates individual bits and writes padded byte buffers.
- `BitReader`: Reads arbitrary bit lengths from raw byte arrays.
- `build_huffman_code_lengths(freqs)`: Constructs optimal prefix code lengths via `heapq` (single-symbol handled as length 1).
- `canonical_codes(lengths)`: Deterministic canonical Huffman code generator.
- `huffman_encode` & `huffman_decode`: Full bitstream encoding and fast trie-based decoding.
- `average_code_length` & `shannon_entropy_from_freqs`: Shannon entropy theorem verifier ($H \le L < H + 1$).
- `serialize_code_lengths` & `deserialize_code_lengths`: Compact table serialization (2 bytes + 2 bytes per symbol).

### `medcomp.dct_codec`
- `dct_encode(img, quality=50, q_table=None)`: Vectorized 8x8 block DCT (`scipy.fft.dctn`), standard IJG quantization scaling, differential DC coding, zigzag scan, run-length AC coding with amplitude categories, adaptive canonical Huffman tables, and compact header assembly.
- `dct_decode(data, q_table=None)`: Bitstream header parsing, canonical Huffman table reconstruction, block dequantization, 2D IDCT, and exact image cropping.
- `get_quantization_table(quality)`: Scaled JPEG luminance quantization matrix.

---

## Pipelines & Benchmark Scripts

### 1. Preprocess Dataset (512x512)
```powershell
python scripts/preprocess_all.py
```

### 2. Run Baseline Codecs (JPEG, JPEG2000, PNG)
```powershell
python scripts/run_baselines.py
python scripts/plot_baselines.py
```

### 3. Run Custom DCT Codec & Comparative Analysis
```powershell
python scripts/run_dct.py
python scripts/plot_dct_vs_baselines.py
```
