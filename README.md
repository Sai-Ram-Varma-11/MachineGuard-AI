# MachineGuard AI

**Intelligent Predictive Maintenance and Machine Failure Detection Using Machine Learning**
*Predict. Understand. Maintain.*

> Student project for the AICTE-IBM SkillsBuild-BharatCares Machine Learning & Applied AI Internship 2026.
> Educational prototype - **not** an IBM product, **not** a certified industrial safety or control system.

## Overview
MachineGuard AI estimates machine failure risk from operating parameters (machine type, air/process temperature, rotational speed, torque, tool wear), explains which features influence the model, analyses historical data, discovers operating patterns with K-Means, and gives cautious rule-based maintenance insights - in an interactive Streamlit app.

## Problem statement
Unexpected machine failures can lead to unplanned downtime and maintenance costs. Maintenance may rely on fixed schedules or on reacting after problems occur. A data-driven tool can support understandable, risk-based maintenance decisions.

## Objectives
Analyse data -> preprocess without leakage -> compare four classifiers -> tune and select on evidence -> explain -> cluster -> serve through a dashboard -> document honestly.

## Features
Dashboard - Failure prediction with model-estimated probability and application-defined risk bands - Out-of-distribution warnings - Global feature importance - Analytics (Plotly) - Model comparison, confusion matrix, ROC/PR curves - K-Means + PCA - Rule-based recommendations - Unit tests.

## Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI, DOI 10.24432/C5HS5C): 10,000 records, **synthetic** but designed to reflect industrial data. Target: `Machine failure`. **Excluded from features:** `UID`, `Product ID` (identifiers) and `TWF, HDF, PWF, OSF, RNF` (failure-mode flags = target leakage).

## ML methodology
Stratified 80/20 split (`random_state=42`) -> `ColumnTransformer` (OneHot for Type, StandardScaler for scale-sensitive models) inside one `Pipeline` -> Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (class weighting for imbalance) -> GridSearchCV (5-fold, F1) on RF and GB -> final model chosen by **cross-validated F1 on training data** (test set touched once) -> metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix. See `docs/methodology.md`.

## Architecture
See `docs/architecture.md`. Training (`src/train_models.py`) writes models + metrics; the app only loads them (never retrains).

## Installation
```bash
git clone <repository-url>
cd MachineGuard-AI
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Get the dataset
```bash
python src/data_loader.py --download     # or download manually from UCI and save as data/raw/ai4i2020.csv
```
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Model training (about 3-6 minutes)
```bash
python src/train_models.py
python src/fill_report.py     # optional: fills docs placeholders with real results
```

## Run locally
```bash
streamlit run app.py
```

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Application screenshots
Pending execution - capture from the running app and save in `outputs/figures/screenshots/`:
1 Dashboard - 2 Predict page - 3 Low-risk prediction - 4 High-risk prediction - 5 Analytics - 6 Model comparison - 7 Confusion matrix - 8 Feature importance - 9 Clusters - 10 About.

## Evaluation
Pending execution. After training, results are in `outputs/metrics/` and `outputs/reports/results_summary` (via `fill_report.py`). No number is hard-coded anywhere.

## Limitations
Synthetic dataset - not validated on real sensor streams - depends on training-data quality - prototype - risk scores are model estimates, not guarantees - must not control machinery - real use needs domain validation, monitoring, sensor integration and safety controls.

## Future scope
IoT streaming, time-series forecasting, Remaining Useful Life, anomaly detection, real datasets, edge deployment, alerts, monitoring/drift detection, IBM Watson ML deployment, digital twin, technician assistant (none implemented).

## Deployment (Streamlit Community Cloud)
1. Run training locally and commit `data/`, `models/`, `outputs/` (all small). 2. Push to GitHub. 3. At share.streamlit.io choose the repo, branch `main`, main file `app.py`. 4. Deploy. Live URL: *Pending deployment.*
Use the same scikit-learn version for training and deployment (pin it in `requirements.txt` after training, e.g. `pip freeze | grep scikit-learn`).

## IBM note
IBM SkillsBuild concepts and the machine-learning workflow were incorporated into the project, while the deployable prototype uses open-source Python ML tools. (Edit this line only if you genuinely run an IBM Watson Studio experiment.)

## Dataset attribution
Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C - CC BY 4.0.

## License
Code: MIT (see `LICENSE`). Dataset: CC BY 4.0.

## Author
<"Sairam Varma"> - <"Lovely Professional University"> - AICTE-IBM SkillsBuild-BharatCares Internship 2026
