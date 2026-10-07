import pandas as pd
import pytest

from src.config import FEATURES, LEAKAGE_COLUMNS, ID_COLUMNS, TARGET
from src.data_loader import DataValidationError, clean_dataset, load_raw, validate_raw


def test_validate_ok(raw_df):
    report = validate_raw(raw_df)
    assert report["missing_values"] == 0 and report["rows"] == len(raw_df)


def test_missing_column_raises(raw_df):
    with pytest.raises(DataValidationError):
        validate_raw(raw_df.drop(columns=["Torque [Nm]"]))


def test_invalid_type_raises(raw_df):
    bad = raw_df.copy(); bad.loc[0, "Type"] = "X"
    with pytest.raises(DataValidationError):
        validate_raw(bad)


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw(tmp_path / "nope.csv")


def test_load_roundtrip(raw_df, tmp_path):
    p = tmp_path / "d.csv"; raw_df.to_csv(p, index=False)
    assert len(load_raw(p)) == len(raw_df)


def test_clean_dataset_drops_ids_and_leakage(raw_df):
    clean = clean_dataset(raw_df)
    assert list(clean.columns) == FEATURES + [TARGET]
    assert not set(ID_COLUMNS + LEAKAGE_COLUMNS) & set(clean.columns)
