"""End-to-end training script:  python src/train_models.py

load -> validate -> clean -> EDA -> stratified split -> train 4 baselines ->
cross-validated tuning of RF and GB -> choose final model by CROSS-VALIDATED F1
on the TRAINING data -> evaluate ONCE on the untouched test set -> save
model/metrics/figures -> explainability -> K-Means + PCA.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from src import clustering, eda  # noqa: E402
from src.config import (  # noqa: E402
    BEST_MODEL_PATH, DATA_PROCESSED_PATH, FIGURES_DIR, METRICS_DIR, MODELS_DIR,
    N_CV_FOLDS, PREPROCESSOR_PATH, RANDOM_STATE, RANGES_PATH, REPORTS_DIR, TARGET,
)
from src.data_loader import clean_dataset, load_raw, validate_raw  # noqa: E402
from src.evaluate_models import (  # noqa: E402
    curve_data, evaluate, plot_confusion_matrix, plot_curves, plot_feature_importance,
)
from src.explainability import IMPORTANCE_PATH, global_importance  # noqa: E402
from src.preprocessing import build_preprocessor, compute_training_stats, split_data  # noqa: E402
from src.utils import save_json, setup_logging  # noqa: E402

logger = logging.getLogger("train")


@dataclass
class Spec:
    estimator: object
    scale: bool
    sample_weight: bool = False      # GB has no class_weight -> use sample weights
    grid: dict = field(default_factory=dict)


def candidates() -> dict[str, Spec]:
    """Four required models; RF and GB also get a small tuning grid."""
    rs = RANDOM_STATE
    return {
        "Logistic Regression": Spec(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=rs), scale=True),
        "Decision Tree": Spec(
            DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=rs), scale=False),
        "Random Forest": Spec(
            RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=rs),
            scale=False,
            grid={"classifier__n_estimators": [100, 200],
                  "classifier__max_depth": [8, None],
                  "classifier__min_samples_leaf": [1, 3],
                  "classifier__class_weight": ["balanced", "balanced_subsample"]}),
        "Gradient Boosting": Spec(
            GradientBoostingClassifier(random_state=rs), scale=False, sample_weight=True,
            grid={"classifier__n_estimators": [100, 200],
                  "classifier__learning_rate": [0.05, 0.1],
                  "classifier__max_depth": [2, 3],
                  "classifier__min_samples_split": [2, 10]}),
    }


def make_pipe(spec: Spec, params: dict | None = None) -> Pipeline:
    pipe = Pipeline([("preprocess", build_preprocessor(spec.scale)),
                     ("classifier", clone(spec.estimator))])
    if params:
        pipe.set_params(**params)
    return pipe


def fit(pipe: Pipeline, spec: Spec, X, y) -> Pipeline:
    kwargs = {"classifier__sample_weight": compute_sample_weight("balanced", y)} if spec.sample_weight else {}
    return pipe.fit(X, y, **kwargs)


def cv_scores(spec: Spec, params: dict | None, X, y) -> dict:
    """Mean 5-fold stratified CV scores on TRAINING data only."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for tr, va in skf.split(X, y):
        pipe = fit(make_pipe(spec, params), spec, X.iloc[tr], y.iloc[tr])
        pred, prob = pipe.predict(X.iloc[va]), pipe.predict_proba(X.iloc[va])[:, 1]
        rows.append((f1_score(y.iloc[va], pred, zero_division=0),
                     recall_score(y.iloc[va], pred, zero_division=0),
                     precision_score(y.iloc[va], pred, zero_division=0),
                     roc_auc_score(y.iloc[va], prob)))
    m = np.mean(rows, axis=0)
    return {"cv_f1": float(m[0]), "cv_recall": float(m[1]),
            "cv_precision": float(m[2]), "cv_roc_auc": float(m[3])}


def explain_selection(best: dict, table: pd.DataFrame) -> str:
    """Evidence-based text built from the real numbers."""
    others = table[table["model"] != best["model"]]
    runner = others.sort_values("cv_f1", ascending=False).iloc[0]
    fn = best["confusion_matrix"][1][0]
    fp = best["confusion_matrix"][0][1]
    return (
        f"{best['model']} was selected because it had the highest mean {N_CV_FOLDS}-fold "
        f"cross-validated F1-score on the training data (CV F1 = {best['cv_f1']:.3f}; "
        f"runner-up: {runner['model']} at {runner['cv_f1']:.3f}). Selection used training-set "
        f"cross-validation only, so the test set stayed unseen. On the test set it reached "
        f"recall = {best['recall']:.3f}, precision = {best['precision']:.3f}, "
        f"F1 = {best['f1']:.3f}, ROC-AUC = {best['roc_auc']:.3f}, with {fn} missed failures "
        f"(false negatives) and {fp} false alarms (false positives). In predictive maintenance a "
        "false negative (a missed failure risk) is usually costlier than a false positive "
        "(an unnecessary inspection), so recall and F1 were prioritised over accuracy."
    )


def main(k: int | None = None) -> None:
    setup_logging()
    for d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR, REPORTS_DIR, DATA_PROCESSED_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    validation = validate_raw(raw)
    clean = clean_dataset(raw)
    clean.to_csv(DATA_PROCESSED_PATH, index=False)
    summary = eda.run_eda(raw, clean)
    logger.info("Class distribution: %s", summary["class_distribution"])

    X_train, X_test, y_train, y_test = split_data(clean)
    save_json(compute_training_stats(X_train), RANGES_PATH)
    logger.info("Split: train=%d test=%d (stratified, rs=%d)", len(X_train), len(X_test), RANDOM_STATE)

    fitted: dict[str, Pipeline] = {}
    params_used: dict[str, dict] = {}
    for name, spec in candidates().items():
        logger.info("Training baseline: %s", name)
        fitted[name] = fit(make_pipe(spec), spec, X_train, y_train)
        params_used[name] = {}
        if spec.grid:
            logger.info("Tuning (GridSearchCV, %d-fold, scoring=F1): %s", N_CV_FOLDS, name)
            gs = GridSearchCV(
                make_pipe(spec), spec.grid, scoring="f1", refit=True, n_jobs=1,
                cv=StratifiedKFold(N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))
            kw = {"classifier__sample_weight": compute_sample_weight("balanced", y_train)} if spec.sample_weight else {}
            gs.fit(X_train, y_train, **kw)
            fitted[f"{name} (tuned)"] = gs.best_estimator_
            params_used[f"{name} (tuned)"] = {k_: v for k_, v in gs.best_params_.items()}
            logger.info("Best params for %s: %s", name, gs.best_params_)

    spec_for = {n.replace(" (tuned)", ""): s for n, s in candidates().items()}
    rows, curves = [], {}
    for name, pipe in fitted.items():
        spec = spec_for[name.replace(" (tuned)", "")]
        cv = cv_scores(spec, params_used[name] or None, X_train, y_train)
        test = evaluate(pipe, X_test, y_test)
        rows.append({"model": name, **cv, **test, "params": params_used[name]})
        curves[name] = curve_data(pipe, X_test, y_test)

    table = pd.DataFrame(rows)
    order = table.sort_values(["cv_f1", "cv_recall"], ascending=False)
    best_name = order.iloc[0]["model"]
    best_row = next(r for r in rows if r["model"] == best_name)
    reason = explain_selection(best_row, table)
    logger.info(reason)

    flat = table.drop(columns=["confusion_matrix", "params"])
    flat.round(4).to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    save_json({
        "dataset": {**summary["class_distribution"], "rows": validation["rows"],
                    "train_rows": len(X_train), "test_rows": len(X_test)},
        "random_state": RANDOM_STATE, "cv_folds": N_CV_FOLDS,
        "models": rows, "selected": {"model": best_name, "params": params_used[best_name],
                                     "reason": reason, **{k_: best_row[k_] for k_ in
                                     ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                      "specificity", "false_positive_rate", "confusion_matrix", "cv_f1")}},
    }, METRICS_DIR / "metrics.json")
    save_json(curves, METRICS_DIR / "curves.json")

    final = fitted[best_name]
    joblib.dump(final, BEST_MODEL_PATH)
    joblib.dump(final.named_steps["preprocess"], PREPROCESSOR_PATH)
    plot_confusion_matrix(best_row["confusion_matrix"], FIGURES_DIR / "confusion_matrix.png", best_name)
    plot_curves(curves, "roc", FIGURES_DIR / "roc_curves.png")
    plot_curves(curves, "pr", FIGURES_DIR / "pr_curves.png")

    imp = global_importance(final)
    imp.to_csv(IMPORTANCE_PATH, index=False)
    plot_feature_importance(imp, FIGURES_DIR / "feature_importance.png")

    clustering.run_clustering(clean, k=k)
    print("\n=== MODEL COMPARISON (test set) ===")
    print(flat[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1"]].round(3).to_string(index=False))
    print(f"\nFinal model: {best_name}\n{reason}\n\nNext: streamlit run app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="override number of K-Means clusters")
    main(ap.parse_args().k)
