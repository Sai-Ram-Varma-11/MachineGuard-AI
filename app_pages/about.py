import streamlit as st

from src.config import DISCLAIMER


def render() -> None:
    st.title("About MachineGuard AI")
    st.caption("Predict. Understand. Maintain.")
    st.warning(DISCLAIMER)
    st.markdown("""
**Project:** MachineGuard AI - Intelligent Predictive Maintenance and Machine Failure Detection Using Machine Learning
*Developed as a student project for the AICTE-IBM SkillsBuild-BharatCares internship. This is not an IBM product and is not endorsed by IBM.*

**Problem.** Unexpected machine failures can lead to unplanned downtime and maintenance costs. Maintenance may rely on fixed schedules or on reacting after problems occur.

**Solution.** Estimate failure risk from operating parameters, explain influential model features, analyse historical behaviour and discover operating patterns.

**Objectives.** Analyse data - preprocess without leakage - compare four classifiers - tune and select by evidence - explain - cluster - serve through an interactive app.

**Dataset.** AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository (CC BY 4.0), DOI 10.24432/C5HS5C. It is **synthetic**, designed to reflect predictive-maintenance data found in industry.

**Technologies.** Python, pandas, NumPy, scikit-learn, Plotly, Matplotlib, Streamlit, joblib.

**ML techniques.** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, stratified split, class weighting, GridSearchCV, K-Means, PCA, rule-based recommendations.

**SDG alignment.** Supports data-driven industrial innovation related to UN SDG 9 (Industry, Innovation and Infrastructure); possible secondary link to SDG 12. No measured SDG impact is claimed.

**IBM connection.** IBM SkillsBuild concepts and the machine-learning workflow were incorporated into the project, while the deployable prototype uses open-source Python ML tools.
""")
    st.subheader("Limitations")
    for item in ["The dataset is synthetic.", "The model is not validated on real industrial sensor streams.",
                 "Predictions depend on the quality and distribution of the training data.", "The application is a prototype.",
                 "Risk scores are model estimates, not guarantees.", "It must not directly control industrial machinery.",
                 "Real deployment would need domain validation, monitoring, sensor integration and safety controls."]:
        st.write("- " + item)
    st.subheader("Future scope")
    st.write("Real-time IoT sensor integration, time-series forecasting, Remaining Useful Life estimation, anomaly detection, "
             "real industrial datasets, edge deployment, automated alerts, model monitoring and drift detection, "
             "IBM Watson Machine Learning deployment, digital twins, technician-facing assistant. (Not implemented.)")
    st.subheader("Dataset attribution")
    st.write("Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. "
             "https://doi.org/10.24432/C5HS5C - licensed CC BY 4.0.")
