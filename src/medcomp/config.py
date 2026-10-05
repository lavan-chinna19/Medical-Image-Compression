"""Configuration module for paths, supported image extensions, and project constants."""

from pathlib import Path

# Project root directory (3 levels up from src/medcomp/config.py: src/medcomp -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_PROCESSED_IMAGES_DIR = DATA_PROCESSED_DIR / "images"
DATA_PROCESSED_MASKS_DIR = DATA_PROCESSED_DIR / "masks"
DATA_MASKS_DIR = DATA_DIR / "masks"
MASKS_LEFT_DIR = DATA_MASKS_DIR / "leftmask"
MASKS_RIGHT_DIR = DATA_MASKS_DIR / "rightmask"
MASKS_ROI_DIR = DATA_MASKS_DIR / "roi"

# Results directories and report paths
RESULTS_DIR = PROJECT_ROOT / "results"
MASK_PREVIEWS_DIR = RESULTS_DIR / "mask_previews"
MASK_REPORT_PATH = RESULTS_DIR / "mask_report.csv"
BASELINES_CSV_PATH = RESULTS_DIR / "baselines.csv"
BASELINE_SUMMARY_CSV_PATH = RESULTS_DIR / "baseline_summary.csv"
DCT_RESULTS_CSV_PATH = RESULTS_DIR / "dct_results.csv"
DCT_SUMMARY_CSV_PATH = RESULTS_DIR / "dct_summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

# Supported image file extensions
SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".dcm",
)

# Standard DCT block dimension
DEFAULT_BLOCK_SIZE = 8

# Standard experimental image dimension (512x512)
TARGET_SIZE = 512

# Default peak value for 8-bit grayscale images
DEFAULT_DATA_RANGE = 255.0
