from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATASET_PATH = Path("spending_patterns_detailed.csv")
DEFAULT_BUNDLE_PATH = Path("lr_expense_bundle.pkl")


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"Category", "Item", "Total Spent"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def train_logistic_regression(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    min_samples: int = 3,
    max_features: int = 3000,
) -> Dict[str, Any]:
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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "num_classes": int(y.nunique()),
    }

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

    predicted = bundle["model"].predict(x)[0]
    probability = float(bundle["model"].predict_proba(x).max())

    return {
        "predicted_category": str(predicted),
        "confidence": round(probability * 100, 2),
    }


def get_categories(bundle_path: Path = DEFAULT_BUNDLE_PATH) -> List[str]:
    bundle = load_bundle(bundle_path)
    return list(bundle["classes"])
