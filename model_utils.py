from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


DEFAULT_DATASET_PATH = Path("spending_patterns_detailed.csv")
DEFAULT_BUNDLE_PATH = Path("lr_expense_bundle.pkl")


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"Category", "Item", "Total Spent"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def _prepare_features(
    dataset_path: Path,
    min_samples: int,
    max_features: int,
) -> Tuple[Any, pd.Series, TfidfVectorizer, StandardScaler, OneHotEncoder, List[str]]:
    df = pd.read_csv(dataset_path)
    _validate_columns(df)

    feature_cols = ["Item", "Total Spent", "Payment Method", "Location"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[["Category"] + feature_cols].dropna(subset=["Category", "Item", "Total Spent"])
    df["Item"] = df["Item"].astype(str)
    df["Total Spent"] = pd.to_numeric(df["Total Spent"], errors="coerce")
    df = df.dropna(subset=["Total Spent"])

    category_counts = df["Category"].value_counts()
    rare_categories = category_counts[category_counts < min_samples].index
    if len(rare_categories) > 0:
        df["Category"] = df["Category"].replace(rare_categories, "Other")

    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    scaler = StandardScaler(with_mean=False)
    encoder = OneHotEncoder(handle_unknown="ignore")

    x_text = tfidf.fit_transform(df["Item"])
    x_amount = scaler.fit_transform(df[["Total Spent"]].values)
    x_cat = encoder.fit_transform(df[["Payment Method", "Location"]].fillna(""))
    x = hstack([x_text, x_amount, x_cat])
    y = df["Category"]

    return x, y, tfidf, scaler, encoder, feature_cols


def _train_test_split(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


def _compute_metrics(y_true, y_pred, train_size: int, test_size: int, num_classes: int) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "train_size": int(train_size),
        "test_size": int(test_size),
        "num_classes": int(num_classes),
    }


def _save_bundle(
    model,
    tfidf: TfidfVectorizer,
    scaler: StandardScaler,
    encoder: OneHotEncoder,
    feature_cols: List[str],
    dataset_path: Path,
    bundle_path: Path,
) -> None:
    bundle = {
        "model": model,
        "tfidf": tfidf,
        "scaler": scaler,
        "encoder": encoder,
        "classes": list(model.classes_),
        "feature_columns": feature_cols,
        "dataset_path": str(dataset_path),
    }
    joblib.dump(bundle, bundle_path)


def _train_and_score(model, x_train, x_test, y_train, y_test, require_dense: bool = False):
    x_train_use = x_train.toarray() if require_dense else x_train
    x_test_use = x_test.toarray() if require_dense else x_test
    model.fit(x_train_use, y_train)
    y_pred = model.predict(x_test_use)
    metrics = _compute_metrics(y_test, y_pred, x_train.shape[0], x_test.shape[0], y_train.nunique())
    return metrics, model


def train_logistic_regression(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    min_samples: int = 3,
    max_features: int = 3000,
) -> Dict[str, Any]:
    x, y, tfidf, scaler, encoder, feature_cols = _prepare_features(
        dataset_path=dataset_path, min_samples=min_samples, max_features=max_features
    )
    x_train, x_test, y_train, y_test = _train_test_split(x, y)

    metrics, model = _train_and_score(
        LogisticRegression(max_iter=3000, class_weight="balanced"),
        x_train,
        x_test,
        y_train,
        y_test,
    )
    _save_bundle(model, tfidf, scaler, encoder, feature_cols, dataset_path, bundle_path)
    return metrics


def compare_models(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    min_samples: int = 3,
    max_features: int = 3000,
    save_logreg_bundle: bool = True,
    return_fitted: bool = False,
) -> List[Dict[str, Any]]:
    """
    Train several algorithms on the same split and return sorted metrics.
    Logistic Regression bundle is saved for downstream use when save_logreg_bundle is True.
    """
    x, y, tfidf, scaler, encoder, feature_cols = _prepare_features(
        dataset_path=dataset_path, min_samples=min_samples, max_features=max_features
    )
    x_train, x_test, y_train, y_test = _train_test_split(x, y)

    candidates = [
        ("logreg", "Logistic Regression", LogisticRegression(max_iter=3000, class_weight="balanced"), False, True),
        ("linearsvc", "Linear SVM (LinearSVC)", LinearSVC(class_weight="balanced", max_iter=5000), False, False),
        ("cnb", "Complement Naive Bayes", ComplementNB(), False, False),
        (
            "rf",
            "Random Forest",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                n_jobs=1,
                class_weight="balanced_subsample",
                random_state=42,
            ),
            True,
            False,
        ),
    ]

    results: List[Dict[str, Any]] = []
    for key, name, model, needs_dense, saves_bundle in candidates:
        metrics, fitted = _train_and_score(model, x_train, x_test, y_train, y_test, require_dense=needs_dense)
        metrics["model"] = name
        metrics["key"] = key
        metrics["supports_proba"] = hasattr(fitted, "predict_proba")
        if return_fitted:
            metrics["_fitted"] = fitted
            metrics["_needs_dense"] = needs_dense
            metrics["_artifacts"] = (tfidf, scaler, encoder, feature_cols)
        results.append(metrics)

        if name == "Logistic Regression" and save_logreg_bundle:
            _save_bundle(fitted, tfidf, scaler, encoder, feature_cols, dataset_path, bundle_path)

    results.sort(key=lambda m: m["f1_weighted"], reverse=True)
    return results


def train_and_save_model(
    model_key: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    min_samples: int = 3,
    max_features: int = 3000,
) -> Dict[str, Any]:
    """
    Train a chosen model, report metrics on a holdout split, then refit on full data and save bundle.
    Only models that support predict_proba should be used in the UI to ensure confidence scores.
    """
    model_options = {
        "logreg": (LogisticRegression(max_iter=3000, class_weight="balanced"), False, "Logistic Regression"),
        "cnb": (ComplementNB(), False, "Complement Naive Bayes"),
        "rf": (
            RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                n_jobs=1,
                class_weight="balanced_subsample",
                random_state=42,
            ),
            True,
            "Random Forest",
        ),
    }
    if model_key not in model_options:
        raise ValueError(f"Model '{model_key}' is not supported for saving (needs predict_proba).")

    base_model, needs_dense, name = model_options[model_key]
    x, y, tfidf, scaler, encoder, feature_cols = _prepare_features(
        dataset_path=dataset_path, min_samples=min_samples, max_features=max_features
    )

    # Evaluate on a split
    x_train, x_test, y_train, y_test = _train_test_split(x, y)
    metrics, fitted = _train_and_score(
        base_model, x_train, x_test, y_train, y_test, require_dense=needs_dense
    )
    metrics["model"] = name
    metrics["key"] = model_key

    # Refit on full data for best inference performance
    x_full = x.toarray() if needs_dense else x
    fitted_full = base_model.__class__(**base_model.get_params())
    fitted_full.fit(x_full, y)

    _save_bundle(fitted_full, tfidf, scaler, encoder, feature_cols, dataset_path, bundle_path)
    return metrics


def load_bundle(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> Dict[str, Any]:
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at '{bundle_path}'. Train the model first."
        )
    return joblib.load(bundle_path)


def predict_category(
    item: str,
    total_spent: float,
    payment_method: str = "",
    location: str = "",
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
) -> Dict[str, Any]:
    bundle = load_bundle(bundle_path)

    row = pd.DataFrame(
        [
            {
                "Item": str(item),
                "Total Spent": float(total_spent),
                "Payment Method": str(payment_method),
                "Location": str(location),
            }
        ]
    )

    x_text = bundle["tfidf"].transform(row["Item"])
    x_amount = bundle["scaler"].transform(row[["Total Spent"]].values)
    x_cat = bundle["encoder"].transform(row[["Payment Method", "Location"]])
    x = hstack([x_text, x_amount, x_cat])

    model = bundle["model"]
    predicted = model.predict(x)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x).max())
    elif hasattr(model, "decision_function"):
        # Convert decision scores to pseudo-probabilities via softmax for display
        import numpy as np

        scores = model.decision_function(x)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        probability = float(probs.max())

    return {
        "predicted_category": str(predicted),
        "confidence": round(probability * 100, 2) if probability is not None else None,
    }


def get_categories(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> List[str]:
    bundle = load_bundle(bundle_path)
    return list(bundle["classes"])
