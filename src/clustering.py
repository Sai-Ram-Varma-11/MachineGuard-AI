"""K-Means operating-pattern discovery + PCA for 2-D visualisation.

Unsupervised learning finds groups WITHOUT using the failure label. Clusters
are named Cluster 1, 2, 3... - we never assume one is "healthy".
PCA is used only to draw the clusters in 2-D, not as the predictive model.
"""
from __future__ import annotations

import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import (
    CLUSTER_K_RANGE, FIGURES_DIR, FEATURE_LABELS, KMEANS_PATH, METRICS_DIR,
    NUM_FEATURES, PCA_PATH, RANDOM_STATE, SCALER_PATH, SILHOUETTE_SAMPLE, TARGET,
)
from src.utils import save_json

logger = logging.getLogger(__name__)


def evaluate_k(X_scaled: np.ndarray) -> pd.DataFrame:
    """Inertia (elbow) and silhouette score for each K."""
    rows = []
    for k in CLUSTER_K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        sil = silhouette_score(
            X_scaled, km.labels_, sample_size=min(SILHOUETTE_SAMPLE, len(X_scaled)),
            random_state=RANDOM_STATE,
        )
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def describe_clusters(profile: pd.DataFrame, centers_z: np.ndarray, overall_rate: float) -> list[str]:
    """Plain-language, data-driven cluster descriptions."""
    texts = []
    for i, row in profile.reset_index(drop=True).iterrows():
        traits = []
        for j, col in enumerate(NUM_FEATURES):
            z = centers_z[i][j]
            if abs(z) >= 0.5:
                traits.append(f"{'higher' if z > 0 else 'lower'} {FEATURE_LABELS[col].split(' [')[0].lower()}")
        trait_txt = ", ".join(traits) if traits else "values close to the dataset average"
        rate = row["failure_rate"]
        rel = "higher" if rate > overall_rate * 1.2 else ("lower" if rate < overall_rate * 0.8 else "similar")
        texts.append(
            f"Cluster {int(row['cluster'])}: {trait_txt}. Observed {rel} failure "
            f"proportion in this dataset ({rate:.1%} vs {overall_rate:.1%} overall). "
            "This describes association only, not cause."
        )
    return texts


def run_clustering(df: pd.DataFrame, k: int | None = None) -> dict:
    """Fit scaler, K-Means and PCA; save models, tables and figures."""
    X = df[NUM_FEATURES]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    k_table = evaluate_k(X_scaled)
    chosen_k = int(k) if k else int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=chosen_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_scaled)
    coords = pca.transform(X_scaled)

    labels = km.labels_ + 1  # human-friendly: Cluster 1..K
    work = df.copy()
    work["cluster"] = labels
    profile = work.groupby("cluster").agg(
        count=(TARGET, "size"),
        **{c: (c, "mean") for c in NUM_FEATURES},
        failure_rate=(TARGET, "mean"),
    ).reset_index()
    overall = float(df[TARGET].mean())
    texts = describe_clusters(profile, km.cluster_centers_, overall)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(km, KMEANS_PATH)
    joblib.dump(pca, PCA_PATH)
    k_table.to_csv(METRICS_DIR / "cluster_k_selection.csv", index=False)
    profile.to_csv(METRICS_DIR / "cluster_profile.csv", index=False)
    pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": labels,
                  TARGET: df[TARGET].values}).to_csv(METRICS_DIR / "cluster_points.csv", index=False)
    info = {
        "chosen_k": chosen_k,
        "selection_rule": "highest silhouette score in K=2..6" if not k else "user-specified K",
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "descriptions": texts,
    }
    save_json(info, METRICS_DIR / "clusters.json")
    _plot(k_table, coords, labels, chosen_k, pca)
    logger.info("K-Means done: K=%d (%s)", chosen_k, info["selection_rule"])
    return info


def _plot(k_table, coords, labels, chosen_k, pca) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(k_table["k"], k_table["inertia"], "o-")
    axes[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
    axes[1].plot(k_table["k"], k_table["silhouette"], "o-", color="#e76f51")
    axes[1].set(title="Silhouette score", xlabel="K", ylabel="Score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_k_selection.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=6, cmap="viridis")
    ev = pca.explained_variance_ratio_
    ax.set(title=f"K-Means clusters (K={chosen_k}) in PCA space",
           xlabel=f"PCA Component 1 ({ev[0]:.0%} var.)",
           ylabel=f"PCA Component 2 ({ev[1]:.0%} var.)")
    ax.legend(*sc.legend_elements(), title="Cluster", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_pca.png", dpi=150)
    plt.close(fig)
