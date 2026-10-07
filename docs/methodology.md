# Methodology

1. **Data collection** - AI4I 2020 (UCI, synthetic, 10,000 rows). Validated for columns, binary target, Type in {L,M,H}, numeric types, missing values; duplicates are reported.
2. **Leakage control** - `TWF/HDF/PWF/OSF/RNF` describe *how* a failure happened and together reproduce the target; they are dropped (EDA reports the measured overlap). `UID` and `Product ID` are identifiers; the machine quality letter in Product ID duplicates `Type`, so Product ID is excluded.
3. **Preprocessing** - `Type` one-hot encoded with fixed categories (L, M, H); the five numeric features standardised for Logistic Regression and passed unchanged to tree models. All fitting happens inside a scikit-learn `Pipeline` on training data only.
4. **Feature engineering** - none in the primary model (only the six allowed raw features). Derived features such as temperature difference or power (torque x angular speed) are a possible extension; they use only allowed inputs, but they would mirror how the synthetic dataset was generated and must be reported separately.
5. **Split** - 80/20, `stratify=y`, `random_state=42`.
6. **Imbalance** - failures are a small minority; handled with `class_weight="balanced"` (LR, DT, RF) and balanced sample weights (Gradient Boosting, which has no `class_weight`). No SMOTE.
7. **Training/evaluation** - four baselines; accuracy, precision, recall, F1, ROC-AUC, PR-AUC, specificity, FPR, confusion matrix on the test set.
8. **Tuning** - GridSearchCV (5-fold stratified, scoring F1) on small grids for RF and GB.
9. **Selection** - highest mean 5-fold CV F1 on the training set (ties: recall). The test set is used once for reporting. Recall is emphasised because a missed failure risk is usually costlier than a false alarm.
10. **Explainability** - global importance (`feature_importances_` or |coefficients|); wording: "strong influence on the model's prediction", never "caused".
11. **Clustering** - scaled numeric features, K-Means for K=2..6, elbow + silhouette, K = highest silhouette unless overridden (`--k`); PCA(2) for plotting; clusters named by number and described from data.
12. **Deployment** - models saved with joblib and loaded by Streamlit with caching.

Risk bands (30% / 70%) are application-defined and illustrative, not empirically validated. Out-of-distribution warnings compare inputs to the training min/max.
