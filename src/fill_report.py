"""Fill documentation placeholders with REAL results:  python src/fill_report.py

Templates in docs/ contain tokens such as {{FINAL_F1}}. This script reads
outputs/metrics/metrics.json (created by train_models.py) and writes completed
copies to outputs/reports/. If training has not been run it refuses, so no
number can ever be invented.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import DOCS_DIR, METRICS_DIR, REPORTS_DIR  # noqa: E402
from src.utils import load_json  # noqa: E402

TEMPLATES = ["concept_note.md", "project_report.md", "ppt_content.md", "viva_questions.md"]


def build_tokens() -> dict[str, str]:
    m = load_json(METRICS_DIR / "metrics.json")
    c = load_json(METRICS_DIR / "clusters.json")
    s, d = m["selected"], m["dataset"]
    cm = s["confusion_matrix"]
    return {
        "{{FINAL_MODEL}}": s["model"], "{{FINAL_ACCURACY}}": f"{s['accuracy']:.3f}",
        "{{FINAL_PRECISION}}": f"{s['precision']:.3f}", "{{FINAL_RECALL}}": f"{s['recall']:.3f}",
        "{{FINAL_F1}}": f"{s['f1']:.3f}", "{{FINAL_ROC_AUC}}": f"{s['roc_auc']:.3f}",
        "{{FINAL_PR_AUC}}": f"{s['pr_auc']:.3f}", "{{FINAL_CV_F1}}": f"{s['cv_f1']:.3f}",
        "{{FINAL_FN}}": str(cm[1][0]), "{{FINAL_FP}}": str(cm[0][1]),
        "{{N_RECORDS}}": f"{d['rows']:,}", "{{N_FAILURES}}": f"{d['failure']:,}",
        "{{FAILURE_PERCENT}}": f"{d['failure_percent']:.2f}%", "{{BEST_K}}": str(c["chosen_k"]),
        "{{SELECTION_REASON}}": s["reason"],
    }


def results_table() -> str:
    t = pd.read_csv(METRICS_DIR / "model_comparison.csv")[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    t.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    for _, r in t.iterrows():
        lines.append(f"| {r['Model']} | " + " | ".join(f"{r[c]:.3f}" for c in t.columns[1:]) + " |")
    return "\n".join(lines)


def main() -> None:
    try:
        tokens = build_tokens()
        tokens["{{MODEL_TABLE}}"] = results_table()
    except FileNotFoundError:
        sys.exit("No results found. Run `python src/train_models.py` first (placeholders stay 'Pending execution').")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in TEMPLATES:
        text = (DOCS_DIR / name).read_text(encoding="utf-8")
        for key, val in tokens.items():
            text = text.replace(key, val)
        (REPORTS_DIR / name).write_text(text, encoding="utf-8")
        print("written", REPORTS_DIR / name)


if __name__ == "__main__":
    main()
