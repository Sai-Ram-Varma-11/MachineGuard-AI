"""Dataset acquisition, validation and cleaning.

WHY: A model is only as trustworthy as its data. This module makes sure the
file exists, has the expected structure, and is converted into a clean table
that contains ONLY the allowed features and the target (no identifiers, no
failure-mode leakage columns).
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow: python src/data_loader.py

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    DATA_PROCESSED_PATH, DATA_RAW_PATH, EXPECTED_ROWS, FEATURES, ID_COLUMNS,
    LEAKAGE_COLUMNS, NUM_FEATURES, RAW_COLUMNS, RENAME_MAP, TARGET,
    TARGET_RAW, TYPE_VALUES, UCI_ZIP_URL,
)

logger = logging.getLogger(__name__)

# The UCI CSV names its first column "UDI"; the UCI web page calls it "UID".
# We accept both and standardise to "UID".
COLUMN_ALIASES = {"UDI": "UID"}


class DataValidationError(Exception):
    """Raised when the dataset is missing columns or has invalid content."""


def download_dataset(dest: Path = DATA_RAW_PATH) -> Path:
    """Download AI4I 2020 from UCI (standard library only) into ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith("ai4i2020.csv"))
            dest.write_bytes(zf.read(name))
    except Exception as exc:  # network, zip or lookup problems
        raise RuntimeError(
            "Automatic download failed. Please download the dataset manually "
            "from https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive"
            f"+maintenance+dataset and save the CSV as {dest}. ({exc})"
        ) from exc
    logger.info("Dataset saved to %s", dest)
    return dest


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from headers and map known aliases (UDI -> UID)."""
    df = df.rename(columns=lambda c: str(c).strip())
    return df.rename(columns=COLUMN_ALIASES)


def validate_raw(df: pd.DataFrame) -> dict:
    """Check structure and content; return a small validation report."""
    df = standardise_columns(df)
    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Dataset is missing columns: {missing_cols}. Found columns: {list(df.columns)}"
        )
    if df[TARGET_RAW].isna().any():
        raise DataValidationError("Target column contains missing values.")
    if not set(df[TARGET_RAW].unique()) <= {0, 1}:
        raise DataValidationError("Target must be binary (0/1).")
    bad_types = set(df["Type"].dropna().unique()) - set(TYPE_VALUES)
    if bad_types:
        raise DataValidationError(f"Unexpected Type values: {sorted(bad_types)}")
    raw_numeric = [c for c, clean in RENAME_MAP.items() if clean in NUM_FEATURES]
    for col in raw_numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric.")
    required = raw_numeric + ["Type", TARGET_RAW]
    null_counts = df[required].isna().sum()
    if null_counts.any():
        raise DataValidationError(
            f"Missing values in required columns: {null_counts[null_counts > 0].to_dict()}"
        )
    if len(df) != EXPECTED_ROWS:
        logger.warning("Expected %d rows but found %d.", EXPECTED_ROWS, len(df))
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_ids": int(
            df.drop(columns=ID_COLUMNS).duplicated().sum()
        ),
    }


def load_raw(path: Path = DATA_RAW_PATH, validate: bool = True) -> pd.DataFrame:
    """Load the raw CSV with clear errors if it is missing or malformed."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python src/data_loader.py "
            "--download` or download it manually from the UCI link in the README."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(f"Could not read dataset: {exc}") from exc
    df = standardise_columns(df)
    if validate:
        validate_raw(df)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and keep ONLY allowed features + target.

    Identifiers (UID, Product ID) and failure-mode flags (TWF, HDF, PWF,
    OSF, RNF) are dropped here, so they can never reach the model.
    """
    out = standardise_columns(df).rename(columns=RENAME_MAP)
    out = out[FEATURES + [TARGET]].copy()
    out[TARGET] = out[TARGET].astype(int)
    return out


def load_processed(path: Path = DATA_PROCESSED_PATH) -> pd.DataFrame:
    """Load the cleaned dataset created by train_models.py."""
    if not path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run `python src/train_models.py` first."
        )
    return pd.read_csv(path)


def dropped_columns() -> list[str]:
    """Columns intentionally excluded from the model (for documentation)."""
    return ID_COLUMNS + LEAKAGE_COLUMNS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="MachineGuard AI dataset helper")
    parser.add_argument("--download", action="store_true", help="download from UCI")
    args = parser.parse_args()
    if args.download:
        download_dataset()
    print(validate_raw(load_raw(validate=False)))