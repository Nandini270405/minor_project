"""
Generate evaluation visuals for the expense classifier:
- model metric comparison bar chart
- confusion matrix for Logistic Regression
- macro + micro ROC curves for Logistic Regression

Outputs are saved under the ./figures directory as PNG files.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    RocCurveDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

from model_utils import _prepare_features, _train_test_split, compare_models


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def plot_metric_comparison() -> Path:
    """Train all candidate models and plot their weighted metrics."""
    summaries = compare_models(return_fitted=False)
    metrics_df = pd.DataFrame(summaries)[
        ["model", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    ]
    metrics_long = metrics_df.melt(id_vars="model", var_name="metric", value_name="score")

    plt.figure(figsize=(9, 5))
    for idx, metric in enumerate(sorted(metrics_long["metric"].unique())):
        subset = metrics_long[metrics_long["metric"] == metric]
        plt.bar(
            np.arange(len(subset)) + idx * 0.2,
            subset["score"],
            width=0.2,
            label=metric.replace("_", " ").title(),
        )
    plt.xticks(
        ticks=np.arange(len(metrics_df)) + 0.3,
        labels=metrics_df["model"],
        rotation=20,
        ha="right",
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Score (0-1)")
    plt.title("Model Metric Comparison (weighted averages)")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "model_metric_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_metric_table() -> Path:
    """Create a compact table image comparing all model metrics."""
    summaries = compare_models(return_fitted=False)
    df = pd.DataFrame(summaries)[
        ["model", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    ]
    df.columns = ["Model", "Accuracy", "Precision (w)", "Recall (w)", "F1 (w)"]
    df = df.round(3)

    fig_width = 8.0
    fig_height = max(2.5, 0.55 * len(df) + 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        colColours=["#e3f2fd"] * len(df.columns),
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.25)

    ax.set_title("Model Comparison (weighted metrics)", pad=12, fontsize=11)
    out_path = FIG_DIR / "model_metric_table.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def _train_logreg_split():
    """Train Logistic Regression using the same preprocessing as the app."""
    x, y, tfidf, scaler, encoder, feature_cols = _prepare_features(
        dataset_path=Path("spending_patterns_detailed.csv"), min_samples=3, max_features=3000
    )
    x_train, x_test, y_train, y_test = _train_test_split(x, y)
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(x_train, y_train)
    artifacts = {
        "tfidf": tfidf,
        "scaler": scaler,
        "encoder": encoder,
        "feature_cols": feature_cols,
    }
    return model, x_train, x_test, y_train, y_test, artifacts


def plot_confusion_matrix(model, x_test, y_test) -> Path:
    """Plot a normalized confusion matrix for the given model and data."""
    y_pred = model.predict(x_test)
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format=".2f")
    ax.set_title("Normalized Confusion Matrix — Logistic Regression")
    plt.tight_layout()

    out_path = FIG_DIR / "confusion_matrix_logreg.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_class_distribution() -> Path:
    df = pd.read_csv("spending_patterns_detailed.csv")
    counts = df["Category"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")
    plt.ylabel("Count")
    plt.title("Class Distribution (Category)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = FIG_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_precision_recall(model, x_test, y_test) -> Path:
    """Micro and macro PR curves for multiclass Logistic Regression."""
    y_score = model.predict_proba(x_test)
    classes = model.classes_
    y_bin = label_binarize(y_test, classes=classes)

    # Micro
    precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_bin, y_score, average="micro")

    # Macro
    precision_dict, recall_dict, ap_dict = {}, {}, {}
    for i, cls in enumerate(classes):
        precision_dict[cls], recall_dict[cls], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap_dict[cls] = average_precision_score(y_bin[:, i], y_score[:, i])

    all_recall = np.unique(np.concatenate([recall_dict[c] for c in classes]))
    mean_precision = np.zeros_like(all_recall)
    for c in classes:
        mean_precision += np.interp(all_recall, recall_dict[c][::-1], precision_dict[c][::-1])
    mean_precision /= len(classes)
    ap_macro = average_precision_score(y_bin, y_score, average="macro")

    plt.figure(figsize=(7, 6))
    plt.plot(recall_micro, precision_micro, label=f"Micro-average (AP = {ap_micro:.3f})", linewidth=2)
    plt.plot(all_recall, mean_precision, label=f"Macro-average (AP = {ap_macro:.3f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves — Logistic Regression (multiclass)")
    plt.legend(loc="lower left")
    plt.ylim([0.0, 1.02])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()

    out_path = FIG_DIR / "precision_recall_logreg.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_roc_curves(model, x_test, y_test) -> Path:
    """Plot micro and macro ROC curves for the multiclass Logistic Regression."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("ROC curves require predict_proba support.")

    y_score = model.predict_proba(x_test)
    classes = model.classes_
    y_test_bin = label_binarize(y_test, classes=classes)

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    auc_micro = roc_auc_score(y_test_bin, y_score, average="micro")

    # Macro-average
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i, cls in enumerate(classes):
        fpr_dict[cls], tpr_dict[cls], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc_dict[cls] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

    all_fpr = np.unique(np.concatenate([fpr_dict[cls] for cls in classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls in classes:
        mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
    mean_tpr /= len(classes)
    auc_macro = roc_auc_score(y_test_bin, y_score, average="macro")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_micro, tpr_micro, label=f"Micro-average (AUC = {auc_micro:.3f})", linewidth=2)
    plt.plot(all_fpr, mean_tpr, label=f"Macro-average (AUC = {auc_macro:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Logistic Regression (multiclass)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = FIG_DIR / "roc_curve_logreg.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_multi_model_roc() -> Path:
    """
    Overlay micro-averaged ROC curves for all implemented models.
    Uses the same train/test split and preprocessing as compare_models.
    """
    # Prepare data once
    x, y, *_ = _prepare_features(
        dataset_path=Path("spending_patterns_detailed.csv"), min_samples=3, max_features=3000
    )
    x_train, x_test, y_train, y_test = _train_test_split(x, y)
    classes = np.unique(y)
    y_bin = label_binarize(y_test, classes=classes)

    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=3000, class_weight="balanced"), False),
        ("Linear SVM (LinearSVC)", LinearSVC(class_weight="balanced", max_iter=5000), False),
        ("Complement Naive Bayes", ComplementNB(), False),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                n_jobs=1,
                class_weight="balanced_subsample",
                random_state=42,
            ),
            True,
        ),
    ]

    plt.figure(figsize=(7.5, 6))
    for name, model, needs_dense in candidates:
        x_train_use = x_train.toarray() if needs_dense else x_train
        x_test_use = x_test.toarray() if needs_dense else x_test
        model.fit(x_train_use, y_train)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test_use)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(x_test_use)
            # For binary case, decision_function returns 1d; expand to 2d
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
        else:
            continue  # skip models without a scoring method

        # Align columns to classes if needed
        if scores.shape[1] != len(classes):
            # This can happen if classes order differs; reorder using model.classes_
            order = [list(model.classes_).index(c) for c in classes]
            scores = scores[:, order]

        fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
        auc_micro = roc_auc_score(y_bin, scores, average="micro")
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_micro:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = FIG_DIR / "roc_curve_all_models.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_feature_importance(model, artifacts) -> Path:
    """Top features by absolute coefficient magnitude (LogReg one-vs-rest)."""
    tfidf = artifacts["tfidf"]
    encoder = artifacts["encoder"]

    vocab = np.array(tfidf.get_feature_names_out())
    cat_names = encoder.get_feature_names_out(["Payment Method", "Location"])
    feature_names = np.concatenate([vocab, np.array(["Total Spent"]), cat_names])

    coefs = model.coef_
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)
    mean_abs = np.mean(np.abs(coefs), axis=0)

    top_idx = np.argsort(mean_abs)[-20:][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_idx)), mean_abs[top_idx][::-1])
    plt.yticks(range(len(top_idx)), feature_names[top_idx][::-1])
    plt.xlabel("Mean |Coefficient| across classes")
    plt.title("Top Features — Logistic Regression")
    plt.tight_layout()

    out_path = FIG_DIR / "feature_importance_logreg.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_learning_curve(model, x, y) -> Path:
    train_sizes, train_scores, test_scores = learning_curve(
        model, x, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=1
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, "o-", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", label="CV score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve — Logistic Regression")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = FIG_DIR / "learning_curve_logreg.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_top_confusions(model, x_test, y_test) -> Path:
    """Show the most common misclassifications."""
    y_pred = model.predict(x_test)
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)

    records = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if cm[i, j] > 0:
                records.append((true_label, pred_label, cm[i, j]))
    records = sorted(records, key=lambda r: r[2], reverse=True)[:10]
    if not records:
        return None

    df = pd.DataFrame(records, columns=["True", "Predicted", "Count"])
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(df)), df["Count"][::-1])
    plt.yticks(range(len(df)), (df["True"] + " → " + df["Predicted"])[::-1])
    plt.xlabel("Count")
    plt.title("Top Misclassifications — Logistic Regression")
    plt.tight_layout()

    out_path = FIG_DIR / "top_confusions_logreg.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_architecture() -> Path:
    """High-level project architecture diagram (clean layout)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1.8, 3.1)

    palette = {
        "user": "#f9f0e6",
        "ui": "#dbeafe",
        "api": "#e0f2fe",
        "store": "#e8f5e9",
        "data": "#f3e8ff",
        "model": "#e5f3ff",
    }

    def add_box(x, y, text, width=2.2, height=0.95, color="#e0f0ff", edge="#1f4b99"):
        rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9, wrap=True)
        return (x + width / 2, y + height / 2)

    def arrow(p1, p2, text=None, style="-|>", dashed=False):
        ax.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops=dict(
                arrowstyle=style,
                linewidth=1.5,
                color="#2f3b52",
                linestyle="--" if dashed else "-",
                shrinkA=6,
                shrinkB=6,
            ),
        )
        if text:
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            ax.text(mid[0], mid[1] + 0.15, text, ha="center", va="center", fontsize=8, color="#2f3b52")

    # Layered nodes (left to right)
    user = add_box(0.0, 1.7, "User\nBrowser / Streamlit UI", color=palette["user"], edge="#b77814")
    ui = add_box(2.5, 1.7, "Streamlit Frontend\n(expense_ml.py)", color=palette["ui"])
    flask = add_box(5.0, 1.7, "Flask API\n(backend.py)", color=palette["api"])
    bundle = add_box(7.5, 1.7, "Model Bundle\nlr_expense_bundle.pkl", color=palette["model"])

    db = add_box(5.0, 0.2, "SQLite DB\ntransactions", color=palette["store"], edge="#2e7d32")
    dataset = add_box(2.5, -1.2, "Historical Dataset\nspending_patterns_detailed.csv", color=palette["data"], edge="#7b1fa2")
    training = add_box(5.0, -1.2, "Training & Comparison\n(model_utils.py)", color=palette["model"])

    # Flows
    arrow(user, ui, "UI interactions")
    arrow(ui, flask, "API calls (optional)", dashed=True)
    arrow(ui, db, "save/load transactions")
    arrow(ui, bundle, "predict_category\n(load model)")
    arrow(dataset, ui, "history preview", dashed=True)

    arrow(dataset, training, "train/test split")
    arrow(training, bundle, "save bundle")

    arrow(flask, bundle, "predict / train")
    arrow(flask, db, "persist transactions", dashed=True)

    ax.set_title("Expense Tracker ML — Architecture", fontsize=12, pad=10, color="#1f2937")
    plt.tight_layout()
    out_path = FIG_DIR / "architecture_diagram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main():
    metric_path = plot_metric_comparison()
    metric_table_path = plot_metric_table()
    class_dist_path = plot_class_distribution()
    model, x_train, x_test, y_train, y_test, artifacts = _train_logreg_split()
    cm_path = plot_confusion_matrix(model, x_test, y_test)
    pr_path = plot_precision_recall(model, x_test, y_test)
    roc_path = plot_roc_curves(model, x_test, y_test)
    fi_path = plot_feature_importance(model, artifacts)
    lc_path = plot_learning_curve(
        LogisticRegression(max_iter=3000, class_weight="balanced"), x_train, y_train
    )
    tc_path = plot_top_confusions(model, x_test, y_test)
    arch_path = plot_architecture()
    roc_all_path = plot_multi_model_roc()

    print("Saved figures:")
    print(f"- {metric_path}")
    print(f"- {metric_table_path}")
    print(f"- {class_dist_path}")
    print(f"- {cm_path}")
    print(f"- {pr_path}")
    print(f"- {roc_path}")
    print(f"- {fi_path}")
    print(f"- {lc_path}")
    if tc_path:
        print(f"- {tc_path}")
    print(f"- {arch_path}")
    print(f"- {roc_all_path}")


if __name__ == "__main__":
    main()
