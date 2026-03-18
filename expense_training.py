from __future__ import annotations

from pathlib import Path

from model_utils import compare_models


if __name__ == "__main__":
    dataset = Path("spending_patterns_detailed.csv")
    bundle = Path("lr_expense_bundle.pkl")

    print("Training and comparing models...")
    summaries = compare_models(
        dataset_path=dataset,
        bundle_path=bundle,
        min_samples=3,
        max_features=3000,
        save_logreg_bundle=True,
    )

    print("\nMetrics (sorted by weighted F1):")
    print(f"{'Model':26} {'Acc':>6} {'Prec_w':>8} {'Recall_w':>8} {'F1_w':>8}")
    for row in summaries:
        print(
            f"{row['model']:26} "
            f"{row['accuracy']:.3f} "
            f"{row['precision_weighted']:.3f} "
            f"{row['recall_weighted']:.3f} "
            f"{row['f1_weighted']:.3f}"
        )

    best = summaries[0]
    print(f"\nBest model by weighted F1: {best['model']} (F1={best['f1_weighted']:.3f})")
    print(f"Logistic Regression bundle saved to {bundle}")
