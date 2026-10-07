import sys
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import clean_dataset  # noqa: E402
from src.preprocessing import build_preprocessor, compute_training_stats, split_data  # noqa: E402
from tests.synthetic import make_synthetic_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean_dataset(raw_df)


@pytest.fixture(scope="session")
def trained(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    model = Pipeline([("preprocess", build_preprocessor(False)),
                      ("classifier", RandomForestClassifier(n_estimators=30, random_state=42))])
    model.fit(X_train, y_train)
    return model, compute_training_stats(X_train), (X_train, X_test, y_train, y_test)


@pytest.fixture()
def valid_input():
    return {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0,
            "rot_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}
