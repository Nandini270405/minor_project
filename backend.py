from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request

from model_utils import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_DATASET_PATH,
    get_categories,
    predict_category,
    train_logistic_regression,
)


app = Flask(__name__)


@app.get("/")
def index():
    return jsonify(
        {
            "message": "Expense Tracker ML backend is running",
            "endpoints": {
                "health": "GET /health",
                "train": "POST /train",
                "predict": "POST /predict",
                "categories": "GET /categories",
            },
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/train")
def train():
    payload = request.get_json(silent=True) or {}
    dataset_path = Path(payload.get("dataset_path", str(DEFAULT_DATASET_PATH)))
    bundle_path = Path(payload.get("bundle_path", str(DEFAULT_BUNDLE_PATH)))
    min_samples = int(payload.get("min_samples", 3))
    max_features = int(payload.get("max_features", 3000))

    try:
        metrics = train_logistic_regression(
            dataset_path=dataset_path,
            bundle_path=bundle_path,
            min_samples=min_samples,
            max_features=max_features,
        )
        return jsonify(
            {
                "message": "Model trained successfully",
                "dataset_path": str(dataset_path),
                "bundle_path": str(bundle_path),
                "metrics": metrics,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    item = payload.get("item", "")
    total_spent = payload.get("total_spent")
    payment_method = payload.get("payment_method", "")
    location = payload.get("location", "")
    bundle_path = Path(payload.get("bundle_path", str(DEFAULT_BUNDLE_PATH)))

    if not item:
        return jsonify({"error": "Field 'item' is required"}), 400
    if total_spent is None:
        return jsonify({"error": "Field 'total_spent' is required"}), 400

    try:
        result = predict_category(
            item=item,
            total_spent=float(total_spent),
            payment_method=payment_method,
            location=location,
            bundle_path=bundle_path,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/categories")
def categories():
    bundle_path = Path(request.args.get("bundle_path", str(DEFAULT_BUNDLE_PATH)))
    try:
        return jsonify({"categories": get_categories(bundle_path=bundle_path)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
