# Project Report - MachineGuard AI

## 1. Title
MachineGuard AI - Intelligent Predictive Maintenance and Machine Failure Detection Using Machine Learning

## 2. Abstract
Unexpected machine failures can cause downtime and maintenance cost. This project builds an educational predictive-maintenance prototype on the synthetic AI4I 2020 dataset (10,000 records, 3.39% failures). Four supervised classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) were compared with leakage-safe preprocessing, stratified splitting, class weighting and cross-validated tuning. The selected model, Random Forest, achieved F1 = 0.703, recall = 0.662 and ROC-AUC = 0.955 on the held-out test set. K-Means (K = 3) with PCA visualisation reveals operating patterns, and a Streamlit app provides prediction, explainability, analytics and cautious rule-based maintenance insights. Limitations: synthetic data, no real-world validation. *(If values above still show braces, results are: Pending execution.)*

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
UCI dataset 601, 10,000 records, 14 raw columns, synthetic. Target Machine failure (339 failures). Excluded: UID, Product ID, TWF, HDF, PWF, OSF, RNF.

## 10. Data preprocessing
Validation, renaming, one-hot encoding of Type, standardisation for LR, stratified 80/20 split (random_state=42), pipeline fitted on training data only.

## 11. Exploratory data analysis
Figures in `outputs/figures/eda_*.png`; numbers in `outputs/metrics/eda_summary.json` (shape, missing, duplicates, class balance, outliers, correlations, leakage check).

## 12. Feature engineering
None in the primary model; possible extensions (temperature difference, power) are future work and must be reported separately.

## 13. Supervised learning
LR, DT (max_depth 5), RF (200 trees), GB; balanced class weights / sample weights.

## 14. Model evaluation
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.825 | 0.142 | 0.824 | 0.242 | 0.907 |
| Decision Tree | 0.929 | 0.311 | 0.882 | 0.460 | 0.908 |
| Random Forest | 0.981 | 0.750 | 0.662 | 0.703 | 0.955 |
| Random Forest (tuned) | 0.981 | 0.750 | 0.662 | 0.703 | 0.955 |
| Gradient Boosting | 0.934 | 0.324 | 0.868 | 0.472 | 0.966 |
| Gradient Boosting (tuned) | 0.950 | 0.391 | 0.868 | 0.539 | 0.966 |

## 15. Hyperparameter tuning
GridSearchCV, 5-fold stratified CV, scoring F1, small grids for RF and GB (see `metrics.json` for the chosen parameters).

## 16. Model selection
Random Forest was selected because it had the highest mean 5-fold cross-validated F1-score on the training data (CV F1 = 0.679; runner-up: Random Forest (tuned) at 0.679). Selection used training-set cross-validation only, so the test set stayed unseen. On the test set it reached recall = 0.662, precision = 0.750, F1 = 0.703, ROC-AUC = 0.955, with 23 missed failures (false negatives) and 15 false alarms (false positives). In predictive maintenance a false negative (a missed failure risk) is usually costlier than a false positive (an unnecessary inspection), so recall and F1 were prioritised over accuracy.

## 17. Explainability
Global feature importance (`outputs/figures/feature_importance.png`). Wording: features have strong influence on the model's prediction; no causal claim.

## 18. Unsupervised learning
Discovers operating patterns without using the failure label.

## 19. K-Means
K = 2..6 evaluated with elbow and silhouette; chosen K = 3; clusters described from data (`outputs/metrics/clusters.json`).

## 20. PCA
Two components for visualisation only.

## 21. Application
Streamlit: Dashboard, Predict Failure, Analytics, Model Performance, Machine Clusters, About.

## 22. System architecture
See `docs/architecture.md`.

## 23. Results
Final model Random Forest: accuracy 0.981, precision 0.750, recall 0.662, F1 0.703, ROC-AUC 0.955, PR-AUC 0.753; 23 false negatives and 15 false positives on the test set.

## 24. Limitations
Synthetic dataset; not validated on real industrial streams; depends on training-data distribution; prototype; scores are estimates; not for machine control; risk bands are illustrative.

## 25. Future scope
IoT, time series, RUL, anomaly detection, real data, edge, alerts, monitoring, drift, IBM Watson ML, digital twin, technician assistant.

## 26. Conclusion
The project delivers a reproducible, explainable, honestly evaluated predictive-maintenance prototype. Conclusions about real industrial benefit require real-world validation.

## 27. References
Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI. doi:10.24432/C5HS5C. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830. Streamlit documentation. United Nations, SDG 9 and SDG 12.
