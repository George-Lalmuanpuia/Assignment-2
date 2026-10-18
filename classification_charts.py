"""
make_model_charts.py
Draws the classification result charts.

Input files:
  logreg_test_predictions.csv      - one row per test crash: y_true, y_pred, y_prob
                                     (1 = Casualty, 0 = Non-casualty; y_prob = predicted chance of casualty)
  logreg_permutation_importance.csv - feature, importance_mean, importance_std

Charts created:
  chart_confusion_matrix.png         - counts and row percentages
  chart_class_metrics.png            - model vs "do nothing" baseline for each metric
  chart_roc_pr_curves.png            - ROC curve and precision-recall curve
  chart_probability_distribution.png - predicted chance of Casualty, split by the true class
  chart_feature_importance.png       - permutation importance of each feature

To run:
  python make_model_charts.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, ConfusionMatrixDisplay)

PRED_FILE = "data/processed/logreg_test_predictions.csv"

IMPORTANCE_FILE = "data/processed/logreg_permutation_importance.csv"

CLASS_NAMES = {0: "Non-casualty", 1: "Casualty"}  # what 0 and 1 mean in y_true / y_pred
POSITIVE = 0                                   # class treated as "positive" (Non-casualty)


def compute_metrics(y_true, y_pred, y_prob):
    # Use non-casualty as the positive class for these metrics
    actual = (y_true == POSITIVE).to_numpy()
    predicted = (y_pred == POSITIVE).to_numpy()

    # model's chance of the positive class (y_prob is the chance of class 1)
    score_pos = (1 - y_prob) if POSITIVE == 0 else y_prob
    prevalence = actual.mean()
    names = CLASS_NAMES[POSITIVE]

    model = {
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced\naccuracy": balanced_accuracy_score(actual, predicted),
        f"F1\n({names})": f1_score(actual, predicted),
        "ROC-AUC": roc_auc_score(actual, score_pos),
        f"PR-AUC\n({names})": average_precision_score(actual, score_pos),
    }

    baseline = {
        "Accuracy": max(prevalence, 1 - prevalence),    # always guess the bigger class
        "Balanced\naccuracy": 0.5,
        f"F1\n({names})": 2 * prevalence / (1 + prevalence),  # always guess the positive class
        "ROC-AUC": 0.5,
        f"PR-AUC\n({names})": prevalence,
    }

    extra = {"Precision": precision_score(actual, predicted), "Recall": recall_score(actual, predicted)}
    
    return model, baseline, extra, actual, score_pos


def save(fig, filename):
    """Save at 200 dpi so the chart stays readable in the PDF report."""
    plt.tight_layout()
    fig.savefig(f"figures/{filename}", dpi=200)
    plt.close(fig)
    print("saved", filename)


def confusion_chart(y_true, y_pred):
    """Confusion matrix twice: counts, and the share of each true class."""
    labels = [0, 1]
    names = [CLASS_NAMES[i] for i in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, display_labels=names,
                                            cmap="Blues", colorbar=False, ax=axes[0])
    axes[0].set_title("Logistic Regression: Confusion Matrix (Counts)", fontsize=9)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, display_labels=names,
                                            normalize="true", values_format=".0%",
                                            cmap="Blues", colorbar=False, ax=axes[1])
    axes[1].set_title("Logistic Regression: Confusion Matrix (Row Percentages)", fontsize=9)
    save(fig, "chart_confusion_matrix.png")


def metrics_chart(model, baseline):
    """Grouped bars: the model next to a model that does nothing useful."""
    x = np.arange(len(model))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - 0.2, list(model.values()), 0.4, label="Model", color="#1f77b4")
    b2 = ax.bar(x + 0.2, list(baseline.values()), 0.4, label="Baseline (no skill)", color="#bbbbbb")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(model.keys()))
    ax.set_ylim(0, 1.05)
    ax.set_title("Logistic Regression: Crash Casualty Classification — Model vs Baseline")
    ax.legend()
    save(fig, "chart_class_metrics.png")


def curves_chart(pos, score_pos):
    """ROC curve and precision-recall curve for the positive (Non-severe) class."""
    fpr, tpr, _ = roc_curve(pos, score_pos)
    prec, rec, _ = precision_recall_curve(pos, score_pos)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.3))
    a1.plot(fpr, tpr, label=f"Model (AUC = {roc_auc_score(pos, score_pos):.2f})")
    a1.plot([0, 1], [0, 1], "--", color="grey", label="Chance (AUC = 0.50)")
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    a1.set_title("Logistic Regression: ROC Curve (Casualty Classification)", fontsize=9)
    a1.legend()
    a2.plot(rec[:-1], prec[:-1], label=f"Model (AP = {average_precision_score(pos, score_pos):.2f})")
    a2.axhline(pos.mean(), ls="--", color="grey", label=f"Chance (= {pos.mean():.2f})")
    a2.set_xlabel("Recall")
    a2.set_ylabel("Precision")
    a2.set_title(
    f"Logistic Regression: Precision-Recall Curve ({CLASS_NAMES[POSITIVE]})", fontsize=9)
    a2.legend()
    save(fig, "chart_roc_pr_curves.png")


def distribution_chart(y_true, y_prob):
    """Histogram of the predicted chance of Severe, split by the true class."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls, colour in [(1, "#d62728"), (0, "#2ca02c")]:
        ax.hist(y_prob[y_true == cls], bins=40, alpha=0.55, color=colour, density=True,
                label=f"Actually {CLASS_NAMES[cls]}")
    ax.set_xlabel(f"Predicted chance of {CLASS_NAMES[1]}")
    ax.set_ylabel("Density")
    ax.set_title("Logistic Regression: Predicted Casualty Probabilities")
    ax.legend()
    save(fig, "chart_probability_distribution.png")


def importance_chart(path):
    """Horizontal bars: how much the ROC-AUC drops when each feature is shuffled."""
    df = pd.read_csv(path).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(df["feature"], df["importance_mean"], color="#1f77b4")
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("Logistic Regression: Feature Importance for Crash Casualty Prediction", fontsize=9)
    save(fig, "chart_feature_importance.png")


def main():
    if os.path.exists(PRED_FILE):
        df = pd.read_csv(PRED_FILE)
        y_true, y_pred, y_prob = df["y_true"], df["y_pred"], df["y_prob"]
        model, baseline, extra, pos, score_pos = compute_metrics(y_true, y_pred, y_prob)
        print(f"{len(df)} test rows. Model vs baseline:")
        for k in model:
            print(f"  {k.replace(chr(10), ' '):28s} {model[k]:.3f}  (baseline {baseline[k]:.3f})")
        print("  " + ", ".join(f"{k} ({CLASS_NAMES[POSITIVE]}) {v:.3f}" for k, v in extra.items()))
        confusion_chart(y_true, y_pred)
        metrics_chart(model, baseline)
        curves_chart(pos, score_pos)
        distribution_chart(y_true, y_prob)
    else:
        print(f"{PRED_FILE} not found - skipped the prediction charts")
    if os.path.exists(IMPORTANCE_FILE):
        importance_chart(IMPORTANCE_FILE)
    else:
        print(f"{IMPORTANCE_FILE} not found - skipped the importance chart")


if __name__ == "__main__":
    main()
