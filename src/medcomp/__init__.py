"""Medical Image Compression (medcomp) package.

Phase 1: Scaffolding, data loading/saving, and evaluation metrics.
"""

from .baselines import jpeg2000_codec, jpeg_codec, png_codec
from .config import (
    BASELINES_CSV_PATH,
    BASELINE_SUMMARY_CSV_PATH,
    DATA_MASKS_DIR,
    DATA_PROCESSED_DIR,
    DATA_PROCESSED_IMAGES_DIR,
    DATA_PROCESSED_MASKS_DIR,
    DATA_RAW_DIR,
    DEFAULT_BLOCK_SIZE,
    MASKS_ROI_DIR,
    MASK_PREVIEWS_DIR,
    MASK_REPORT_PATH,
    PLOTS_DIR,
    RESULTS_DIR,
    SUPPORTED_EXTENSIONS,
    TARGET_SIZE,
)
from .io_utils import crop_to_shape, list_images, load_image, pad_to_multiple, save_image
from .mask_utils import (
    bounding_box,
    get_roi_mask,
    load_mask,
    merge_masks,
    resize_mask_to_image,
    roi_fraction,
)
from .metrics import (
    bits_per_pixel,
    compression_ratio,
    is_lossless,
    masked_psnr,
    mse,
    psnr,
    shannon_entropy,
    ssim,
)
from .preprocess import load_pair

__version__ = "0.3.0"

__all__ = [
    "TARGET_SIZE",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "DATA_PROCESSED_IMAGES_DIR",
    "DATA_PROCESSED_MASKS_DIR",
    "DATA_MASKS_DIR",
    "MASKS_ROI_DIR",
    "RESULTS_DIR",
    "MASK_PREVIEWS_DIR",
    "MASK_REPORT_PATH",
    "BASELINES_CSV_PATH",
    "BASELINE_SUMMARY_CSV_PATH",
    "PLOTS_DIR",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_BLOCK_SIZE",
    "load_image",
    "save_image",
    "list_images",
    "pad_to_multiple",
    "crop_to_shape",
    "load_mask",
    "merge_masks",
    "get_roi_mask",
    "resize_mask_to_image",
    "roi_fraction",
    "bounding_box",
    "load_pair",
    "jpeg_codec",
    "jpeg2000_codec",
    "png_codec",
    "mse",
    "psnr",
    "ssim",
    "compression_ratio",
    "bits_per_pixel",
    "shannon_entropy",
    "masked_psnr",
    "is_lossless",
]
