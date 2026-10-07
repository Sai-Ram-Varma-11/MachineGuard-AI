# Viva Questions and Answers (65)

*Tokens such as 0.703 are filled with real results by `python src/fill_report.py`. Until you train, treat them as: Pending execution.*

## Q1. Why did you choose predictive maintenance?
**Short answer:** It targets a real industrial problem: unplanned downtime.

**Detailed explanation:** Maintenance may rely on fixed schedules or on reacting after a problem. Using operating data to estimate risk can support better-planned maintenance.

**In MachineGuard AI:** MachineGuard AI turns temperature, speed, torque, tool wear and type into a model-estimated failure risk.

## Q2. What is your problem statement?
**Short answer:** Unexpected machine failures can cause downtime and cost; an accessible risk indicator is needed.

**Detailed explanation:** I avoid saying all industries are reactive; I say maintenance 'may rely on' schedules or reaction.

**In MachineGuard AI:** The app returns status, probability, explanation and cautious recommendations.

## Q3. How does your project relate to SDG 9?
**Short answer:** It supports data-driven industrial innovation concepts.

**Detailed explanation:** SDG 9 covers industry, innovation and infrastructure. I do not claim measured impact because nothing was evaluated in real industry.

**In MachineGuard AI:** A prototype showing how ML can support maintenance planning.

## Q4. Which dataset did you use and why?
**Short answer:** AI4I 2020 Predictive Maintenance (UCI).

**Detailed explanation:** It is public, citable (DOI 10.24432/C5HS5C), CC BY 4.0, has 10,000 rows, no missing values and a binary failure target.

**In MachineGuard AI:** Loaded by src/data_loader.py and validated before use.

## Q5. The dataset is synthetic - is that a problem?
**Short answer:** It limits real-world claims, not the learning workflow.

**Detailed explanation:** Synthetic data was designed to reflect industrial data, but patterns come from a generator, so results may not transfer to real machines.

**In MachineGuard AI:** Stated in the app's About page and in every limitation list.

## Q6. Why drop UID and Product ID?
**Short answer:** They are identifiers, not operating conditions.

**Detailed explanation:** IDs carry no physical meaning and can let a model memorise records. The quality letter inside Product ID duplicates Type.

**In MachineGuard AI:** Config lists them in ID_COLUMNS; assert_no_leakage blocks them.

## Q7. What preprocessing did you do?
**Short answer:** Validation, renaming, one-hot encoding, scaling where needed, stratified split.

**Detailed explanation:** All steps sit inside a scikit-learn Pipeline so training and prediction use identical transformations.

**In MachineGuard AI:** preprocessing.build_preprocessor().

## Q8. What is one-hot encoding and why for Type?
**Short answer:** It turns a category into separate 0/1 columns.

**Detailed explanation:** Type (L/M/H) has no numeric order; label-encoding 0/1/2 would imply one. One-hot avoids that.

**In MachineGuard AI:** Columns type_L, type_M, type_H.

## Q9. Why scale features?
**Short answer:** So no feature dominates just because of its units.

**Detailed explanation:** Logistic Regression and K-Means are distance/gradient sensitive; rpm (~1500) would dwarf torque (~40).

**In MachineGuard AI:** StandardScaler for LR and for K-Means.

## Q10. Why not scale for tree models?
**Short answer:** Trees split on thresholds, which are unaffected by monotonic scaling.

**Detailed explanation:** Scaling is harmless but unnecessary, so tree models receive values unchanged via 'passthrough'.

**In MachineGuard AI:** build_preprocessor(scale=False) for DT, RF, GB.

## Q11. Why an 80/20 split?
**Short answer:** Most data to learn from, a held-out set to test honestly.

**Detailed explanation:** The test set simulates unseen machines and is used once, after selection.

**In MachineGuard AI:** random_state=42 for reproducibility.

## Q12. What is stratification and why use it?
**Short answer:** It keeps the failure proportion equal in train and test.

**Detailed explanation:** With rare failures a random split could put too few failures in one set.

**In MachineGuard AI:** stratify=y in train_test_split.

## Q13. What does random_state=42 do?
**Short answer:** Fixes randomness so results are reproducible.

**Detailed explanation:** The number is arbitrary; consistency matters.

**In MachineGuard AI:** Used for split, models, CV and K-Means.

## Q14. How did you handle class imbalance?
**Short answer:** Class weighting plus metrics beyond accuracy.

**Detailed explanation:** Failures are a small minority (3.39%). Weighting makes mistakes on failures cost more.

**In MachineGuard AI:** class_weight='balanced'; sample weights for Gradient Boosting.

## Q15. What does class_weight='balanced' do?
**Short answer:** Weights classes inversely to their frequency.

**Detailed explanation:** The rare failure class gets a larger weight in the loss, so the model does not ignore it.

**In MachineGuard AI:** LR, DT, RF use it directly.

## Q16. Why not SMOTE?
**Short answer:** Not needed, and it creates synthetic rows that complicate explanation.

**Detailed explanation:** Weighting is simpler and has no leakage risk. If SMOTE were used it must be applied only to training data after splitting.

**In MachineGuard AI:** The brief asked to avoid blind oversampling.

## Q17. Explain Logistic Regression.
**Short answer:** A linear model that outputs a probability via the sigmoid function.

**Detailed explanation:** It combines inputs with weights, then squashes to 0-1. Coefficients are interpretable on scaled inputs.

**In MachineGuard AI:** Baseline and the only model needing scaling.

## Q18. Explain a Decision Tree.
**Short answer:** A flowchart of yes/no splits.

**Detailed explanation:** It picks splits that best separate classes. Unlimited depth overfits, so I limit max_depth to 5.

**In MachineGuard AI:** Easy to explain to non-experts.

## Q19. Explain Random Forest.
**Short answer:** Many decision trees voting together.

**Detailed explanation:** Each tree sees a random sample of rows and features; averaging reduces overfitting.

**In MachineGuard AI:** Tuned on estimators, depth, leaf size and class weight.

## Q20. Explain Gradient Boosting.
**Short answer:** Trees built one after another, each fixing the previous errors.

**Detailed explanation:** It adds small trees that correct residual mistakes; learning_rate controls step size.

**In MachineGuard AI:** Tuned on estimators, learning rate, depth, min_samples_split.

## Q21. Why sample weights for Gradient Boosting?
**Short answer:** The scikit-learn GradientBoostingClassifier has no class_weight parameter.

**Detailed explanation:** compute_sample_weight('balanced') gives the same effect.

**In MachineGuard AI:** Passed as classifier__sample_weight.

## Q22. What is hyperparameter tuning?
**Short answer:** Searching for the best model settings.

**Detailed explanation:** Hyperparameters are set before training (depth, learning rate). Choosing them with the test set would leak information.

**In MachineGuard AI:** GridSearchCV on training data only.

## Q23. What is cross-validation?
**Short answer:** Splitting training data into folds, training on some, validating on the rest, and averaging.

**Detailed explanation:** It gives a more stable estimate than one split.

**In MachineGuard AI:** 5-fold stratified CV.

## Q24. Why GridSearchCV and why small grids?
**Short answer:** It tries every combination in a grid.

**Detailed explanation:** Large grids cost time; small grids are enough for a student project.

**In MachineGuard AI:** 16 combinations each for RF and GB.

## Q25. Define precision.
**Short answer:** Of predicted failures, how many were real.

**Detailed explanation:** TP/(TP+FP). Low precision means many false alarms.

**In MachineGuard AI:** Reported as 0.750 for the final model.

## Q26. Define recall.
**Short answer:** Of real failures, how many were caught.

**Detailed explanation:** TP/(TP+FN). High recall means few missed failures.

**In MachineGuard AI:** Final recall: 0.662.

## Q27. Define F1.
**Short answer:** Harmonic mean of precision and recall.

**Detailed explanation:** It is low if either is low, so it balances both.

**In MachineGuard AI:** Selection criterion (cross-validated).

## Q28. What is ROC-AUC?
**Short answer:** Area under the ROC curve; ranking quality across thresholds.

**Detailed explanation:** 0.5 is chance, 1.0 is perfect separation. It can look optimistic for rare classes.

**In MachineGuard AI:** Final ROC-AUC: 0.955.

## Q29. What is PR-AUC and when is it better?
**Short answer:** Area under the precision-recall curve.

**Detailed explanation:** It focuses on the rare positive class, so it is more informative than ROC-AUC when positives are scarce.

**In MachineGuard AI:** Shown on the Performance page.

## Q30. Explain the confusion matrix.
**Short answer:** A 2x2 table of TN, FP, FN, TP.

**Detailed explanation:** Rows are actual classes, columns predicted. It shows the types of mistakes.

**In MachineGuard AI:** Final model: 23 false negatives, 15 false positives.

## Q31. What is a false positive here?
**Short answer:** An alarm when no failure would occur.

**Detailed explanation:** It costs an unnecessary inspection.

**In MachineGuard AI:** Counted in the confusion matrix.

## Q32. What is a false negative here?
**Short answer:** A missed failure risk.

**Detailed explanation:** Usually costlier (downtime, damage), which is why recall matters.

**In MachineGuard AI:** Shown on the Performance page.

## Q33. Why isn't accuracy enough?
**Short answer:** With few failures, 'always no failure' looks accurate.

**Detailed explanation:** Accuracy hides that the model detects nothing.

**In MachineGuard AI:** See the Performance page expander.

## Q34. What is data leakage?
**Short answer:** Using information that would not be available at prediction time or that reveals the target.

**Detailed explanation:** It inflates scores and fails in real use.

**In MachineGuard AI:** TWF/HDF/PWF/OSF/RNF are excluded.

## Q35. How did you prove leakage in the failure-mode columns?
**Short answer:** I measured how often the target equals 'any failure-mode flag'.

**Detailed explanation:** A value near 1 shows the flags encode the target.

**In MachineGuard AI:** eda_summary.json -> leakage_check.

## Q36. How else did you prevent leakage?
**Short answer:** Fit preprocessing and tuning on training data only.

**Detailed explanation:** Everything lives inside a Pipeline; selection uses training CV; the test set is used once.

**In MachineGuard AI:** train_models.py.

## Q37. What is feature importance?
**Short answer:** A score of how much the model relies on each feature.

**Detailed explanation:** For trees it is based on impurity reduction; for LR on coefficient size.

**In MachineGuard AI:** Dashboard chart 'Features influencing the model'.

## Q38. Does importance mean causation?
**Short answer:** No.

**Detailed explanation:** It shows association learned by the model, not physical cause. I word it 'strong influence on the model's prediction'.

**In MachineGuard AI:** UI wording.

## Q39. How do you interpret Logistic Regression coefficients?
**Short answer:** Positive raises predicted failure odds, negative lowers them (on scaled inputs).

**Detailed explanation:** Magnitude indicates strength when features are standardised.

**In MachineGuard AI:** Used if LR is the final model.

## Q40. How did you pick the final model?
**Short answer:** Highest cross-validated F1 on training data.

**Detailed explanation:** The test set stayed unseen; recall trade-offs are discussed.

**In MachineGuard AI:** Random Forest was selected because it had the highest mean 5-fold cross-validated F1-score on the training data (CV F1 = 0.679; runner-up: Random Forest (tuned) at 0.679). Selection used training-set cross-validation only, so the test set stayed unseen. On the test set it reached recall = 0.662, precision = 0.750, F1 = 0.703, ROC-AUC = 0.955, with 23 missed failures (false negatives) and 15 false alarms (false positives). In predictive maintenance a false negative (a missed failure risk) is usually costlier than a false positive (an unnecessary inspection), so recall and F1 were prioritised over accuracy.

## Q41. What if a model has higher accuracy but lower recall?
**Short answer:** I prefer higher recall if missed failures are costly.

**Detailed explanation:** The trade-off is false alarms vs missed failures; F1 helps balance.

**In MachineGuard AI:** Documented in the selection reason.

## Q42. What is the decision threshold?
**Short answer:** The probability above which we predict failure (default 0.5).

**Detailed explanation:** Lowering it raises recall and lowers precision.

**In MachineGuard AI:** Threshold tuning is future work (would need CV on train).

## Q43. What does predict_proba give?
**Short answer:** The model-estimated probability of each class.

**Detailed explanation:** It is a model output, not a guaranteed real-world probability.

**In MachineGuard AI:** UI label: 'Model-estimated failure probability'.

## Q44. Are the probabilities calibrated?
**Short answer:** Not necessarily.

**Detailed explanation:** Class weighting shifts probabilities upward; calibration would need separate analysis.

**In MachineGuard AI:** One reason I call the bands illustrative.

## Q45. What are your risk bands?
**Short answer:** Low <30%, Medium 30-70%, High >=70%, application-defined.

**Detailed explanation:** They are not scientifically validated or industrial standards.

**In MachineGuard AI:** risk_band() in prediction.py.

## Q46. What is the out-of-distribution warning?
**Short answer:** A warning when an input is outside the training min/max.

**Detailed explanation:** Models extrapolate unreliably beyond seen data.

**In MachineGuard AI:** Shown in the app; prediction still runs.

## Q47. Explain K-Means.
**Short answer:** It groups points around K centres.

**Detailed explanation:** Assign points to the nearest centre, move centres to the group means, repeat.

**In MachineGuard AI:** Finds operating patterns without the failure label.

## Q48. What is the elbow method?
**Short answer:** Plot inertia vs K and look for the bend.

**Detailed explanation:** Inertia always falls with K; the elbow shows diminishing returns.

**In MachineGuard AI:** Elbow curve on the Clusters page.

## Q49. What is the silhouette score?
**Short answer:** How well points fit their own cluster vs others (-1 to 1).

**Detailed explanation:** Higher is better separated.

**In MachineGuard AI:** K was chosen at the highest silhouette in K=2..6 (K = 3).

## Q50. What is PCA and why use it?
**Short answer:** It projects data to fewer dimensions keeping most variance.

**Detailed explanation:** Used only to draw clusters in 2-D.

**In MachineGuard AI:** PCA(2) scatter.

## Q51. Why scale before K-Means?
**Short answer:** K-Means uses distance, so large-unit features would dominate.

**Detailed explanation:** StandardScaler equalises features.

**In MachineGuard AI:** scaler.joblib.

## Q52. Can you call a cluster 'healthy'?
**Short answer:** No, unless analysis supports it.

**Detailed explanation:** Clusters are patterns; a higher failure share is an association, not a cause.

**In MachineGuard AI:** Named Cluster 1, 2, ...

## Q53. What is a Pipeline and ColumnTransformer?
**Short answer:** Pipeline chains steps; ColumnTransformer applies different steps to different columns.

**Detailed explanation:** Type is encoded while numeric features are scaled, and the same logic runs at training and prediction.

**In MachineGuard AI:** best_model.joblib is the full pipeline.

## Q54. What is joblib?
**Short answer:** A library to save and load Python objects efficiently.

**Detailed explanation:** The trained pipeline is serialised so the app loads it instead of retraining.

**In MachineGuard AI:** models/*.joblib.

## Q55. Why Streamlit?
**Short answer:** Fast Python-only way to build data apps.

**Detailed explanation:** It supports widgets, Plotly, caching and free cloud hosting.

**In MachineGuard AI:** app.py + app_pages/.

## Q56. How does caching help?
**Short answer:** It avoids reloading models and data on each interaction.

**Detailed explanation:** st.cache_resource for models, st.cache_data for datasets.

**In MachineGuard AI:** app_pages/common.py.

## Q57. How would you deploy it?
**Short answer:** Streamlit Community Cloud from GitHub.

**Detailed explanation:** Commit code and small model files, set main file app.py, match the scikit-learn version.

**In MachineGuard AI:** Live URL: pending deployment.

## Q58. Is the recommendation engine ML?
**Short answer:** No, it is transparent rules.

**Detailed explanation:** Rules compare inputs to training percentiles and the model risk, with cautious wording.

**In MachineGuard AI:** recommendations.py.

## Q59. How did you test?
**Short answer:** pytest unit tests.

**Detailed explanation:** Cover loading, preprocessing, leakage guard, prediction, invalid/missing/extreme inputs and recommendations, using a tiny synthetic fixture (tests only).

**In MachineGuard AI:** tests/ folder.

## Q60. What are the limitations?
**Short answer:** Synthetic data, no real validation, prototype, estimates not guarantees.

**Detailed explanation:** It must not control machinery; real use needs domain validation and safety controls.

**In MachineGuard AI:** Listed in About and README.

## Q61. What is your future scope?
**Short answer:** IoT, time series, RUL, anomaly detection, real data, drift monitoring, IBM Watson ML.

**Detailed explanation:** None implemented; they are directions.

**In MachineGuard AI:** About page.

## Q62. How is IBM SkillsBuild connected?
**Short answer:** The internship's ML workflow guided the project.

**Detailed explanation:** The deployable prototype uses open-source tools; I do not claim IBM Cloud training or deployment unless I really did it.

**In MachineGuard AI:** README IBM note.

## Q63. What is your Lean Canvas value proposition?
**Short answer:** 'Predict. Understand. Maintain.'

**Detailed explanation:** An early-warning prototype; customers listed are potential segments, not confirmed customers.

**In MachineGuard AI:** docs/lean_canvas.md.

## Q64. What is overfitting and how did you control it?
**Short answer:** Learning noise instead of patterns.

**Detailed explanation:** Depth limits, ensembles, cross-validation and a held-out test set check generalisation.

**In MachineGuard AI:** Compare CV F1 with test F1.

## Q65. Why not automatically choose Random Forest?
**Short answer:** Because the best model depends on evidence.

**Detailed explanation:** Selection is by cross-validated F1 and recall considerations.

**In MachineGuard AI:** Any of the six candidates could win.

---
# Common viva mistakes
1. Quoting any number you did not run yourself - always read it from the app or `outputs/metrics`.
2. Saying "the model guarantees/predicts exactly when a machine will fail" - say "model-estimated risk".
3. Saying a feature "causes" failure - say "had strong influence on the model's prediction".
4. Calling the data real industrial data - it is synthetic.
5. Claiming IBM Cloud training/deployment if you did not do it.
6. Forgetting why TWF/HDF/PWF/OSF/RNF were excluded (leakage).
7. Defending accuracy as the main metric on imbalanced data.
8. Calling clusters "healthy/dangerous" or the risk bands "confidence levels".
9. Saying the test set was used to choose the model (it was not).
10. Claiming SDG impact you did not measure.
