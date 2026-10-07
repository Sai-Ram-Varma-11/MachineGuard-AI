"""Central configuration for MachineGuard AI.

Every path, column name and constant lives here so that no other module
hard-codes them. All paths are relative to the repository root, which makes
the project work after a plain ``git clone`` on any computer.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
DATA_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "processed_data.csv"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
KMEANS_PATH = MODELS_DIR / "clustering_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
PCA_PATH = MODELS_DIR / "pca_model.joblib"
RANGES_PATH = MODELS_DIR / "training_ranges.json"

# ------------------------------------------------------------ dataset info
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
EXPECTED_ROWS = 10_000

RAW_COLUMNS = [
    "UID", "Product ID", "Type", "Air temperature [K]",
    "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
    "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
]
RENAME_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}
ID_COLUMNS = ["UID", "Product ID"]            # identifiers: never features
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # failure-mode flags
TARGET_RAW = "Machine failure"
TARGET = "machine_failure"

CAT_FEATURES = ["type"]
NUM_FEATURES = [
    "air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
TYPE_VALUES = ["L", "M", "H"]

FEATURE_LABELS = {
    "type": "Machine Type",
    "air_temp_k": "Air temperature [K]",
    "process_temp_k": "Process temperature [K]",
    "rot_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}

# ------------------------------------------------------------- ML settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
CLUSTER_K_RANGE = range(2, 7)       # K = 2 .. 6
SILHOUETTE_SAMPLE = 5000

# Application-defined, ILLUSTRATIVE risk bands (not empirically validated).
RISK_BAND_LOW = 0.30
RISK_BAND_HIGH = 0.70
# Percentile used by the rule-based recommender to mean "high for this dataset".
HIGH_PERCENTILE = 90

DISCLAIMER = (
    "MachineGuard AI is an educational predictive-maintenance prototype, not a "
    "certified industrial safety or control system. Predictions are model "
    "estimates based on a synthetic training dataset. Actual maintenance "
    "decisions should follow equipment-specific procedures and qualified "
    "personnel."
)
