import numpy as np
import pytest

from src.config import FEATURES
from src.preprocessing import assert_no_leakage, build_preprocessor, split_data


def test_leakage_guard_blocks_failure_modes():
    with pytest.raises(ValueError):
        assert_no_leakage(FEATURES + ["TWF"])
    with pytest.raises(ValueError):
        assert_no_leakage(FEATURES + ["UID"])
    assert_no_leakage(FEATURES)  # allowed set passes


def test_split_is_80_20_and_stratified(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    assert len(X_test) == pytest.approx(0.2 * len(clean_df), abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


def test_preprocessor_scales_and_encodes(clean_df):
    X_train, *_ = split_data(clean_df)
    out = build_preprocessor(scale=True).fit_transform(X_train)
    assert out.shape[1] == 5 + 3          # 5 numeric + 3 one-hot Type columns
    assert np.allclose(out[:, :5].mean(axis=0), 0, atol=1e-6)


def test_preprocessor_without_scaling_keeps_values(clean_df):
    X_train, *_ = split_data(clean_df)
    out = build_preprocessor(scale=False).fit_transform(X_train)
    assert np.allclose(out[:, 0], X_train["air_temp_k"].values)
