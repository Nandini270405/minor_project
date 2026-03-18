# Expense Tracker ML

A Streamlit-based expense tracker that stores transactions in SQLite, visualizes spending, and suggests categories using a logistic regression text model. A small Flask API is included for programmatic training and predictions.

## Features
- Add single or bulk transactions with manual categories or ML category suggestions.
- Automatic budget warnings (over 85% of income, spending spikes, large single transactions).
- Persistent storage in `expense_tracker.db` with filtering, pagination, and CSV export.
- Visuals: category bar chart, pie chart, and daily trend with rolling average.
- Built-in model training from the UI, CLI script, or API; ships with a pretrained bundle `lr_expense_bundle.pkl`.
- CLI comparison trains Logistic Regression, Linear SVM, Complement NB, Multinomial NB, SGDClassifier (hinge), and Random Forest on the same split and reports metrics.
- Backend API for health, training, prediction, and listing learned categories.

## Project Structure
- `expense_ml.py` — Streamlit app (UI, charts, alerts, DB CRUD, on-demand training).
- `backend.py` — Flask API for training/prediction endpoints.
- `model_utils.py` — ML pipeline (TF-IDF + amount + one-hot metadata, logistic regression).
- `expense_training.py` — CLI helper to retrain on `spending_patterns_detailed.csv`.
- Data/model artifacts: `spending_patterns_detailed.csv`, `lr_expense_bundle.pkl`, `expense_model.pkl`, `amount_scaler.pkl`, `tfidf_vectorizer.pkl`, `expense_tracker.db`.

## Quickstart
1) Create a virtual environment (recommended).
```
python -m venv .venv
.venv\Scripts\activate
```
2) Install dependencies.
```
python -m pip install --upgrade pip
python -m pip install streamlit flask pandas altair scikit-learn scipy joblib
```
3) Launch the Streamlit UI.
```
streamlit run expense_ml.py
```
4) (Optional) Run the Flask API in another shell.
```
python backend.py
```

## Dataset & Model
- Training data default: `spending_patterns_detailed.csv` with columns `Category`, `Item`, `Total Spent`, `Payment Method`, `Location`, and `Transaction Date`.
- Trained model bundle: `lr_expense_bundle.pkl` (created automatically if missing). UI “Retrain Model” or CLI/API will overwrite it.
- Categories with fewer than 3 samples are merged into `Other` during training.

## Using the UI (Streamlit)
- Add Transaction: choose `Expense` or `Income`; single or multiple entry lines (`item, amount` or `category, item, amount` for expenses).
- Category Mode: `Manual` uses the dropdown; `ML Suggestion` predicts with the model.
- Budget & Notifications: set “Monthly Income” in the sidebar to enable alerts.
- Database Viewer: filter by type, category, date range; paginate and download CSV.
- Charts: category totals, share pie, daily trend with 3-day rolling average.

## CLI Training
Retrain and compare models on the bundled dataset:
```
python expense_training.py
```
This writes `lr_expense_bundle.pkl`, trains additional models (Linear SVM, Complement NB, Multinomial NB, SGDClassifier hinge SVM, Random Forest) on the same split, and prints accuracy/precision/recall/F1 for each. Change dataset via flags inside the script or edit the paths.

## API Endpoints (Flask, default http://localhost:5000)
- `GET /health` → `{"status": "ok"}`
- `POST /train` with JSON `{dataset_path, bundle_path, min_samples, max_features}` → trains and returns metrics.
- `POST /predict` with JSON `{item, total_spent, payment_method, location, bundle_path?}` → returns `predicted_category` and `confidence`.
- `GET /categories?bundle_path=...` → lists learned categories.

Example predict:
```
curl -X POST http://localhost:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"item\": \"Taxi ride\", \"total_spent\": 320, \"payment_method\": \"Credit Card\", \"location\": \"City\"}"
```

## Data & Persistence
- Transactions persist in SQLite at `expense_tracker.db` (auto-created). UI writes every added row.
- Historical dataset preview is shown in-app for reference; edits in the UI do not alter the CSV.

## Notes
- Requires Python 3.10+.
- If you swap in your own CSV, ensure required columns exist; add `Payment Method`/`Location` columns (can be empty) to enable metadata features.
- When running both UI and API, use separate shells so `streamlit run` and `python backend.py` stay live.
