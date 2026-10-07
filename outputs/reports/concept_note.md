# Concept Note - MachineGuard AI

**1. Project title:** MachineGuard AI - Intelligent Predictive Maintenance and Machine Failure Detection Using Machine Learning

**2. SDG:** Primary SDG 9 (Industry, Innovation and Infrastructure); possible secondary SDG 12. The project supports data-driven industrial innovation concepts; it does not claim to directly solve an SDG.

**3. Background:** Industrial machines produce operating data (temperature, speed, torque, tool wear). Maintenance may rely on schedules or on reacting after problems occur.

**4. Problem statement:** Unexpected machine failures can lead to unplanned downtime, maintenance costs and production disruption. There is a need for a data-driven tool that can identify potential failure risk from operating conditions and present understandable maintenance-oriented insights.

**5. Proposed solution:** An educational predictive-maintenance prototype that accepts operating parameters, estimates failure risk, explains influential model features, analyses historical data, clusters operating patterns and warns when inputs fall outside the training distribution.

**6. Objectives:** Analyse the data; preprocess without leakage; compare four classifiers; tune and select by evidence; explain; cluster; deliver an interactive app; document limitations.

**7. Target users (potential):** Maintenance teams, production managers, small/medium manufacturers, industrial analytics teams (proposed, not confirmed).

**8. Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI; synthetic; 10,000 records; failure share 3.39%).

**9. Methodology:** Validation -> EDA -> leakage-safe preprocessing -> stratified split -> LR/DT/RF/GB -> cross-validated tuning -> evidence-based selection -> explainability -> K-Means + PCA -> Streamlit app.

**10. Technologies:** Python, pandas, NumPy, scikit-learn, Plotly, Matplotlib, Streamlit, joblib.

**11. Expected outcomes:** A reproducible pipeline, a saved model, an interactive dashboard and honest evaluation. Actual performance: final model Random Forest, F1 0.703, ROC-AUC 0.955 (placeholders are replaced by `python src/fill_report.py`; until then: *Pending execution*).

**12. Innovation:** One integrated prototype combining supervised prediction, unsupervised pattern discovery, explainability, analytics, reliability warnings and a transparent rule layer.

**13. Limitations:** Synthetic data; not validated on real sensor streams; prototype; risk scores are estimates; not for controlling machinery.

**14. Future scope:** IoT integration, time-series forecasting, RUL, anomaly detection, real datasets, edge deployment, alerts, drift monitoring, IBM Watson ML deployment, digital twin.

**15. SDG alignment:** Supports industrial innovation and resilient infrastructure thinking (SDG 9) and, potentially, more efficient resource use through better maintenance planning (SDG 12). No measurable impact has been evaluated.

**16. References:** Matzka, S. (2020) AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository, doi:10.24432/C5HS5C; Pedregosa et al. (2011) Scikit-learn, JMLR 12; Streamlit documentation; UN SDG 9 and 12 documentation.
