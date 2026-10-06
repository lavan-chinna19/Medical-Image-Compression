"""Build comprehensive metadata table and cross-validation folds for Montgomery dataset.

Parses:
- Image filename conventions: stem, filename, label (0 normal / 1 TB), patient_id
- Clinical readings (data/readings/): reading_text_label, age, sex
- Image and mask geometry: original_shape, roi_fraction
- 5-fold cross-validation: StratifiedKFold (shuffle=True, random_state=42)
  (grouped by patient_id if duplicates exist)

Outputs:
- data/metadata.csv: Full dataset metadata and fold assignments
- results/metadata_summary.txt: Overall and per-fold class distribution summary
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medcomp.config import (
    DATA_MASKS_DIR,
    DATA_RAW_DIR,
    DATA_READINGS_DIR,
    METADATA_CSV_PATH,
    METADATA_SUMMARY_PATH,
    RESULTS_DIR,
)
from medcomp.io_utils import list_images
from medcomp.mask_utils import load_mask, roi_fraction

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_metadata")


def parse_reading_file(
    reading_path: Path,
) -> tuple[str, int | None, str | None, str]:
    """Parse patient demographics and clinical diagnosis from a reading text file.

    Args:
        reading_path: Path to the .txt reading file.

    Returns:
        tuple[str, int | None, str | None, str]:
            - reading_text_label: "normal" or "abnormal"
            - age: integer age if present
            - sex: 'M', 'F', 'O' if present
            - clinical_notes: full raw diagnostic text
    """
    if not reading_path.exists():
        return "N/A", None, None, ""

    text = reading_path.read_text(encoding="utf-8", errors="replace").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    sex: str | None = None
    age: int | None = None
    clinical_notes: list[str] = []

    for line in lines:
        lower = line.lower()
        if "sex:" in lower:
            raw_sex = line.split(":", 1)[1].strip().upper()
            sex = raw_sex if raw_sex else None
        elif "age:" in lower:
            raw_age = line.split(":", 1)[1].strip()
            match = re.search(r"\d+", raw_age)
            if match:
                age = int(match.group(0))
        else:
            clinical_notes.append(line)

    notes = " ".join(clinical_notes).strip()
    reading_label = "normal" if notes.lower() == "normal" else "abnormal"
    return reading_label, age, sex, notes


def build_metadata(
    raw_dir: Path = DATA_RAW_DIR,
    masks_root: Path = DATA_MASKS_DIR,
    readings_dir: Path = DATA_READINGS_DIR,
    out_csv: Path = METADATA_CSV_PATH,
    summary_path: Path = METADATA_SUMMARY_PATH,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build metadata.csv and generate 5-fold cross-validation splits.

    Args:
        raw_dir: Directory containing input images.
        masks_root: Directory containing ROI masks.
        readings_dir: Directory containing clinical reading text files.
        out_csv: Output CSV filepath.
        summary_path: Output summary text filepath.
        n_splits: Number of cross-validation folds.
        random_state: Random state for deterministic shuffling.

    Returns:
        pd.DataFrame: Completed metadata dataframe.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    images = list_images(raw_dir)
    if not images:
        logger.error("No images found in %s", raw_dir)
        return pd.DataFrame()

    logger.info("Found %d raw images. Extracting metadata...", len(images))
    rows: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []

    for img_path in tqdm(images, desc="Building metadata", unit="img"):
        stem = img_path.stem
        filename = img_path.name

        # Label from filename suffix _0 (normal) or _1 (TB)
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("0", "1"):
            raise ValueError(f"Filename '{filename}' does not follow '_0'/'_1' convention.")
        label = int(parts[1])
        patient_id = parts[0]

        # Reading file parsing
        reading_file = readings_dir / f"{stem}.txt"
        reading_label, age, sex, notes = parse_reading_file(reading_file)

        # Flag any label disagreement
        expected_reading = "normal" if label == 0 else "abnormal"
        if reading_label != expected_reading:
            flag_info = {
                "stem": stem,
                "filename": filename,
                "filename_label": label,
                "reading_label": reading_label,
                "clinical_notes": notes,
            }
            disagreements.append(flag_info)
            print(
                f"[DISAGREEMENT] {filename}: filename label={label} ({expected_reading}) "
                f"vs reading='{reading_label}' (notes: {notes})"
            )
            logger.warning("Label disagreement in %s: file=%d vs reading=%s", filename, label, reading_label)

        # Original image dimensions
        img_mat = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_mat is None:
            raise ValueError(f"Failed to read image {img_path}")
        orig_shape_str = f"({img_mat.shape[0]}, {img_mat.shape[1]})"

        # ROI fraction from mask
        roi_mask_file = masks_root / "roi" / f"{stem}.png"
        if not roi_mask_file.exists():
            # Fallback to direct load from left/right mask
            from medcomp.mask_utils import get_roi_mask
            mask = get_roi_mask(img_path, masks_root)
        else:
            mask = load_mask(roi_mask_file)
        frac = roi_fraction(mask)

        rows.append({
            "stem": stem,
            "filename": filename,
            "label": label,
            "patient_id": patient_id,
            "reading_text_label": reading_label,
            "age": age if age is not None else "",
            "sex": sex if sex is not None else "",
            "original_shape": orig_shape_str,
            "roi_fraction": round(frac, 6),
        })

    df = pd.DataFrame(rows)

    # Patient ID duplicate verification
    total_samples = len(df)
    unique_patients = df["patient_id"].nunique()
    has_duplicate_patients = unique_patients < total_samples

    print(f"\nTotal images: {total_samples}, Unique patient IDs: {unique_patients}")
    if has_duplicate_patients:
        print(f"[NOTE] Found duplicate patient IDs. Using StratifiedGroupKFold on patient_id.")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        df["fold"] = -1
        for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["label"], groups=df["patient_id"])):
            df.loc[val_idx, "fold"] = fold_idx
    else:
        print(f"[NOTE] All patient IDs are unique. Using 5-fold StratifiedKFold.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        df["fold"] = -1
        for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["label"])):
            df.loc[val_idx, "fold"] = fold_idx

    # Save metadata.csv with retry resilience on Windows
    for attempt in range(5):
        try:
            df.to_csv(out_csv, index=False)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(1.0)
            else:
                alt = out_csv.parent / "metadata_updated.csv"
                df.to_csv(alt, index=False)
                logger.warning("Primary %s locked, saved to %s", out_csv, alt)

    logger.info("Saved metadata to %s (%d rows)", out_csv, len(df))

    # Compile summary report
    summary_lines = [
        "=" * 60,
        "MONTGOMERY DATASET METADATA & CROSS-VALIDATION SUMMARY",
        "=" * 60,
        f"Total images:          {total_samples}",
        f"Unique patient IDs:    {unique_patients}",
        f"Patient ID duplicates: {'Yes (grouped by patient_id)' if has_duplicate_patients else 'None (1 image per patient)'}",
        f"Label disagreements:   {len(disagreements)}",
        "",
        "CLASS DISTRIBUTION (OVERALL):",
        f"  Normal (Label 0):    {(df['label'] == 0).sum()} ({((df['label'] == 0).mean() * 100):.1f}%)",
        f"  TB     (Label 1):    {(df['label'] == 1).sum()} ({((df['label'] == 1).mean() * 100):.1f}%)",
        "",
        "CROSS-VALIDATION FOLDS (5-fold Stratified):",
    ]

    fold_records: list[dict[str, Any]] = []
    for f in range(n_splits):
        fold_df = df[df["fold"] == f]
        n_tot = len(fold_df)
        n_0 = (fold_df["label"] == 0).sum()
        n_1 = (fold_df["label"] == 1).sum()
        pct_1 = (n_1 / n_tot * 100) if n_tot > 0 else 0.0
        summary_lines.append(
            f"  Fold {f}: Total={n_tot:2d} | Normal (0)={n_0:2d} | TB (1)={n_1:2d} ({pct_1:.1f}% TB)"
        )
        fold_records.append({
            "fold": f,
            "total": n_tot,
            "normal_0": n_0,
            "tb_1": n_1,
            "tb_pct": round(pct_1, 1),
        })

    summary_lines.append("=" * 60)
    summary_text = "\n".join(summary_lines)

    for attempt in range(5):
        try:
            summary_path.write_text(summary_text, encoding="utf-8")
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(1.0)
            else:
                summary_path.parent.joinpath("metadata_summary_alt.txt").write_text(summary_text, encoding="utf-8")

    logger.info("Saved metadata summary to %s", summary_path)
    return df


if __name__ == "__main__":
    df_meta = build_metadata()
    if not df_meta.empty:
        summary_content = METADATA_SUMMARY_PATH.read_text(encoding="utf-8")
        print("\n" + summary_content + "\n")
