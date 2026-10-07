# PPT content (12 slides) - fill from real results; use real screenshots only

**1. Title** - MachineGuard AI: Predict. Understand. Maintain. | <Name> | <College> | AICTE-IBM SkillsBuild-BharatCares Internship 2026

**2. Problem statement** - Unexpected failures can cause downtime and cost; maintenance may rely on schedules or reaction; need an accessible failure-risk indicator.

**3. Motivation + SDG 9** - Data-driven industrial innovation; prototype supports SDG 9 concepts (no measured impact claimed).

**4. Objectives** - Analyse - preprocess without leakage - compare 4 models - tune/select by evidence - explain - cluster - dashboard.

**5. Dataset** - AI4I 2020 (UCI, synthetic), 10,000 records, 3.39% failures. Dropped: IDs + 5 failure-mode flags (leakage).

**6. System architecture** - Paste diagram from `docs/architecture.md` (training pipeline -> saved model -> Streamlit app).

**7. ML methodology** - Stratified 80/20 - ColumnTransformer - class weighting - GridSearchCV (5-fold F1) - selection on CV F1 - test set used once - K-Means + PCA.

**8. Model comparison** - Table: | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.825 | 0.142 | 0.824 | 0.242 | 0.907 |
| Decision Tree | 0.929 | 0.311 | 0.882 | 0.460 | 0.908 |
| Random Forest | 0.981 | 0.750 | 0.662 | 0.703 | 0.955 |
| Random Forest (tuned) | 0.981 | 0.750 | 0.662 | 0.703 | 0.955 |
| Gradient Boosting | 0.934 | 0.324 | 0.868 | 0.472 | 0.966 |
| Gradient Boosting (tuned) | 0.950 | 0.391 | 0.868 | 0.539 | 0.966 |  Selected: Random Forest. Why recall/F1 matter (missed failure vs false alarm).

**9. Application screenshots** - Dashboard, Predict (low and high risk), Analytics, Model Performance, Clusters (capture from the running app).

**10. Prediction + explainability + clustering** - Model-estimated probability and illustrative risk bands; feature importance; OOD warning; K = 3.

**11. Results / limitations / future scope** - F1 0.703, recall 0.662, ROC-AUC 0.955; limitations (synthetic, prototype); future (IoT, RUL, drift, IBM Watson ML).

**12. Conclusion** - End-to-end, explainable, honest predictive-maintenance prototype; thank you + repo/live link (once deployed).
