from __future__ import annotations

from pathlib import Path

from model_utils import train_logistic_regression


if __name__ == "__main__":
    metrics = train_logistic_regression(
        dataset_path=Path("spending_patterns_detailed.csv"),
        bundle_path=Path("lr_expense_bundle.pkl"),
        min_samples=3,
        max_features=3000,
    )

    print("Model training completed")
    for key, value in metrics.items():
        print(f"{key}: {value}")
