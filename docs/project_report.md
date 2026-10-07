# Project Report - MachineGuard AI

## 1. Title
MachineGuard AI - Intelligent Predictive Maintenance and Machine Failure Detection Using Machine Learning

## 2. Abstract
Unexpected machine failures can cause downtime and maintenance cost. This project builds an educational predictive-maintenance prototype on the synthetic AI4I 2020 dataset ({{N_RECORDS}} records, {{FAILURE_PERCENT}} failures). Four supervised classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) were compared with leakage-safe preprocessing, stratified splitting, class weighting and cross-validated tuning. The selected model, {{FINAL_MODEL}}, achieved F1 = {{FINAL_F1}}, recall = {{FINAL_RECALL}} and ROC-AUC = {{FINAL_ROC_AUC}} on the held-out test set. K-Means (K = {{BEST_K}}) with PCA visualisation reveals operating patterns, and a Streamlit app provides prediction, explainability, analytics and cautious rule-based maintenance insights. Limitations: synthetic data, no real-world validation. *(If values above still show braces, results are: Pending execution.)*

## 3. Introduction
Predictive maintenance uses data to anticipate equipment problems so maintenance can be planned rather than reactive or purely calendar-based.

## 4. Problem statement
Unexpected machine failures can lead to unplanned downtime, maintenance costs and disruption. There is a need for a data-driven solution that identifies potential failure risk from operating conditions and gives understandable maintenance-oriented insights.

## 5. Motivation
Operating data is often available but not turned into an accessible risk indicator; a student prototype can demonstrate the end-to-end ML workflow.

## 6. Objectives
See README: analyse, preprocess, compare, tune, select, explain, cluster, deploy, document.

## 7. SDG alignment
Primary SDG 9; possible SDG 12. The project supports data-driven industrial innovation concepts; no direct or measured impact is claimed.

## 8. Literature / background
The AI4I 2020 dataset (Matzka, 2020) was published to support predictive-maintenance research with a realistic but synthetic structure. Tree ensembles and boosting are common baselines for tabular classification; precision-recall analysis is recommended for imbalanced problems; K-Means and PCA are standard unsupervised tools. (Add 3-5 papers you have actually read.)

## 9. Dataset
UCI dataset 601, 10,000 records, 14 raw columns, synthetic. Target Machine failure ({{N_FAILURES}} failures). Excluded: UID, Product ID, TWF, HDF, PWF, OSF, RNF.

## 10. Data preprocessing
Validation, renaming, one-hot encoding of Type, standardisation for LR, stratified 80/20 split (random_state=42), pipeline fitted on training data only.

## 11. Exploratory data analysis
Figures in `outputs/figures/eda_*.png`; numbers in `outputs/metrics/eda_summary.json` (shape, missing, duplicates, class balance, outliers, correlations, leakage check).

## 12. Feature engineering
None in the primary model; possible extensions (temperature difference, power) are future work and must be reported separately.

## 13. Supervised learning
LR, DT (max_depth 5), RF (200 trees), GB; balanced class weights / sample weights.

## 14. Model evaluation
{{MODEL_TABLE}}

## 15. Hyperparameter tuning
GridSearchCV, 5-fold stratified CV, scoring F1, small grids for RF and GB (see `metrics.json` for the chosen parameters).

## 16. Model selection
{{SELECTION_REASON}}

## 17. Explainability
Global feature importance (`outputs/figures/feature_importance.png`). Wording: features have strong influence on the model's prediction; no causal claim.

## 18. Unsupervised learning
Discovers operating patterns without using the failure label.

## 19. K-Means
K = 2..6 evaluated with elbow and silhouette; chosen K = {{BEST_K}}; clusters described from data (`outputs/metrics/clusters.json`).

## 20. PCA
Two components for visualisation only.

## 21. Application
Streamlit: Dashboard, Predict Failure, Analytics, Model Performance, Machine Clusters, About.

## 22. System architecture
See `docs/architecture.md`.

## 23. Results
Final model {{FINAL_MODEL}}: accuracy {{FINAL_ACCURACY}}, precision {{FINAL_PRECISION}}, recall {{FINAL_RECALL}}, F1 {{FINAL_F1}}, ROC-AUC {{FINAL_ROC_AUC}}, PR-AUC {{FINAL_PR_AUC}}; {{FINAL_FN}} false negatives and {{FINAL_FP}} false positives on the test set.

## 24. Limitations
Synthetic dataset; not validated on real industrial streams; depends on training-data distribution; prototype; scores are estimates; not for machine control; risk bands are illustrative.

## 25. Future scope
IoT, time series, RUL, anomaly detection, real data, edge, alerts, monitoring, drift, IBM Watson ML, digital twin, technician assistant.

## 26. Conclusion
The project delivers a reproducible, explainable, honestly evaluated predictive-maintenance prototype. Conclusions about real industrial benefit require real-world validation.

## 27. References
Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI. doi:10.24432/C5HS5C. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830. Streamlit documentation. United Nations, SDG 9 and SDG 12.
