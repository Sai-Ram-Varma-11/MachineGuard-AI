# Architecture

```
                         TRAINING (offline): python src/train_models.py
 data/raw/ai4i2020.csv
        |
  data_loader.validate_raw  --> clean_dataset (drop UID, Product ID, TWF/HDF/PWF/OSF/RNF)
        |                                   |
        |                      eda.run_eda --> outputs/metrics/eda_summary.json, figures
        v
  preprocessing.split_data (80/20, stratified, rs=42)
        |
  Pipeline[ColumnTransformer -> classifier]  x  LR, DT, RF, GB (+ tuned RF, GB via GridSearchCV)
        |
  5-fold CV F1 on TRAIN  -> select final model -> evaluate ONCE on TEST
        |
  models/best_model.joblib, preprocessing_pipeline.joblib, training_ranges.json
  outputs/metrics/{metrics.json, model_comparison.csv, curves.json, feature_importance.csv}
        |
  clustering.run_clustering: StandardScaler -> K-Means (K=2..6; elbow+silhouette) -> PCA(2)
        |
  models/{scaler, clustering_model, pca_model}.joblib, outputs/metrics/cluster_*.csv

                         INFERENCE (online): streamlit run app.py
 User input -> prediction.validate_inputs -> out_of_range_warnings
            -> best_model.predict / predict_proba -> risk_band
            -> explainability (importance, input vs training) -> recommendations (rules)
            -> app_pages/* render results (loads saved files only; never retrains)
```

| Layer | Files |
|---|---|
| Data | `src/config.py`, `src/data_loader.py`, `data/` |
| ML | `src/preprocessing.py`, `train_models.py`, `evaluate_models.py`, `clustering.py`, `explainability.py` |
| Inference | `src/prediction.py`, `src/recommendations.py` |
| App | `app.py`, `app_pages/` |
| Quality | `tests/` |

Design choices: one saved Pipeline (no duplicated preprocessing); config in one file; relative paths via `pathlib`; app split into small page modules instead of one giant file (`app_pages/` is used instead of `pages/` to avoid Streamlit's automatic multipage behaviour).
