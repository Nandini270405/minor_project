from __future__ import annotations

import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from model_utils import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_DATASET_PATH,
    predict_category,
    train_logistic_regression,
)


st.set_page_config(page_title="Expense Tracker ML", layout="wide")

DEFAULT_EXPENSE_CATEGORIES = [
    "Food",
    "Groceries",
    "Rent",
    "Utilities",
    "Transport",
    "Shopping",
    "Health",
    "Education",
    "Entertainment",
    "Travel",
    "Other",
]
DB_PATH = Path("expense_tracker.db")


def init_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                item TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                payment_method TEXT,
                location TEXT,
                date TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def load_saved_transactions(db_path: Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT type AS Type,
                   item AS Item,
                   amount AS Amount,
                   category AS Category,
                   payment_method AS "Payment Method",
                   location AS Location,
                   date AS Date
            FROM transactions
            ORDER BY date DESC, id DESC
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(
            columns=["Type", "Item", "Amount", "Category", "Payment Method", "Location", "Date"]
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    return df


def load_db_view_transactions(db_path: Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT id AS ID,
                   type AS Type,
                   item AS Item,
                   amount AS Amount,
                   category AS Category,
                   payment_method AS "Payment Method",
                   location AS Location,
                   date AS Date
            FROM transactions
            ORDER BY id DESC
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    return df


def save_transaction(row: dict, db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO transactions (type, item, amount, category, payment_method, location, date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["Type"],
                row["Item"],
                float(row["Amount"]),
                row["Category"],
                row["Payment Method"],
                row["Location"],
                pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


@st.cache_data
def load_history(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    return df


def build_category_item_map(df: pd.DataFrame) -> dict[str, list[str]]:
    if "Category" not in df.columns or "Item" not in df.columns:
        return {}

    cleaned = df[["Category", "Item"]].dropna().copy()
    cleaned["Category"] = cleaned["Category"].astype(str).str.strip()
    cleaned["Item"] = cleaned["Item"].astype(str).str.strip()
    cleaned = cleaned[(cleaned["Category"] != "") & (cleaned["Item"] != "")]

    mapping: dict[str, list[str]] = {}
    for category, group in cleaned.groupby("Category"):
        unique_items = sorted(group["Item"].unique().tolist())
        mapping[category] = unique_items
    return mapping


def ensure_model(dataset_path: Path, bundle_path: Path) -> str:
    if bundle_path.exists():
        return "Model loaded"
    train_logistic_regression(dataset_path=dataset_path, bundle_path=bundle_path)
    return "Model trained automatically"


def init_state() -> None:
    if "transactions" not in st.session_state:
        st.session_state.transactions = load_saved_transactions(DB_PATH)
    if "monthly_income" not in st.session_state:
        st.session_state.monthly_income = 50000.0
    if "expense_categories" not in st.session_state:
        st.session_state.expense_categories = DEFAULT_EXPENSE_CATEGORIES.copy()


def show_notifications(expenses_df: pd.DataFrame, income: float) -> None:
    total_expense = float(expenses_df["Amount"].sum()) if not expenses_df.empty else 0.0

    if income <= 0:
        st.warning("Set monthly income to enable budget alerts.")
        return

    if total_expense > income:
        st.error("Budget Alert: Total expenses are greater than your monthly income.")
    elif total_expense >= income * 0.85:
        st.warning("Warning: You have used more than 85% of your monthly income.")
    else:
        st.success("Budget is currently within a healthy range.")

    if not expenses_df.empty:
        recent = expenses_df.sort_values("Date").tail(7)
        recent_avg = float(recent["Amount"].mean()) if not recent.empty else 0.0
        overall_avg = float(expenses_df["Amount"].mean())
        if overall_avg > 0 and recent_avg >= overall_avg * 1.5:
            st.warning("High Spending Alert: Recent spending is much higher than your average.")

        high_single_spend_threshold = income * 0.2
        if high_single_spend_threshold > 0 and (expenses_df["Amount"] > high_single_spend_threshold).any():
            st.warning("Large Transaction Alert: One or more expenses are above 20% of your income.")


def main() -> None:
    dataset_path = Path("spending_patterns_detailed.csv")
    bundle_path = Path("lr_expense_bundle.pkl")

    init_state()

    st.title("Expense Tracker Website (ML + Budget Alerts)")
    st.caption("Tracks expenses, predicts categories with Logistic Regression, and warns on overspending.")

    with st.sidebar:
        st.header("Settings")
        st.session_state.monthly_income = st.number_input(
            "Monthly Income", min_value=0.0, value=float(st.session_state.monthly_income), step=1000.0
        )

        if st.button("Retrain Model"):
            metrics = train_logistic_regression(dataset_path=dataset_path, bundle_path=bundle_path)
            st.success("Model retrained successfully")
            st.json(metrics)

        st.info(ensure_model(dataset_path=dataset_path, bundle_path=bundle_path))

        st.subheader("Expense Categories")
        categories_text = st.text_area(
            "One category per line",
            value="\n".join(st.session_state.expense_categories),
            height=180,
        )
        if st.button("Update Categories"):
            parsed = [c.strip() for c in categories_text.splitlines() if c.strip()]
            if not parsed:
                st.warning("At least one category is required.")
            else:
                deduped = list(dict.fromkeys(parsed))
                st.session_state.expense_categories = deduped
                st.success("Categories updated.")

        st.subheader("Data")
        if st.button("Reload Saved Transactions"):
            st.session_state.transactions = load_saved_transactions(DB_PATH)
            st.success("Reloaded data from database.")

    history_df = load_history(dataset_path=DEFAULT_DATASET_PATH if not dataset_path.exists() else dataset_path)
    category_item_map = build_category_item_map(history_df)

    if category_item_map and st.session_state.expense_categories == DEFAULT_EXPENSE_CATEGORIES:
        st.session_state.expense_categories = sorted(category_item_map.keys())

    st.subheader("Add Transaction")
    col1, col2, col3 = st.columns(3)
    with col1:
        trx_type = st.selectbox("Type", ["Expense", "Income"])
    with col2:
        entry_mode = st.radio("Entry Mode", ["Single", "Multiple"], horizontal=True)
    with col3:
        trx_date = st.date_input("Date")

    col4, col5 = st.columns(2)
    with col4:
        payment_method = st.selectbox(
            "Payment Method", ["", "Cash", "Debit Card", "Credit Card", "Digital Wallet"]
        )
    with col5:
        location = st.selectbox("Location", ["", "Online", "In-store", "Mobile App"])

    item = ""
    amount = 0.0
    bulk_text = ""
    category_mode = "Manual"
    selected_manual_category = "Income"

    if trx_type == "Expense":
        c1, c2 = st.columns(2)
        with c1:
            category_mode = st.radio("Category Mode", ["Manual", "ML Suggestion"], horizontal=True)
        with c2:
            selected_manual_category = st.selectbox(
                "Expense Category",
                st.session_state.expense_categories,
            )

    if entry_mode == "Single":
        amount = st.number_input("Amount", min_value=0.0, step=1.0)
        if trx_type == "Expense":
            if category_mode == "Manual":
                suggested_items = category_item_map.get(selected_manual_category, [])
                if suggested_items:
                    item_source = st.radio(
                        "Item Input",
                        ["Select Existing Item", "Enter New Item"],
                        horizontal=True,
                    )
                    if item_source == "Select Existing Item":
                        item = st.selectbox("Item", suggested_items)
                    else:
                        item = st.text_input("New Item", placeholder="Enter item name")
                else:
                    item = st.text_input("Item", placeholder="Enter item name")
            else:
                item = st.text_input("Item", placeholder="e.g. Milk, Taxi/Uber, Shoes")
        else:
            item = st.text_input("Item", placeholder="e.g. Salary")
    else:
        st.caption("Multiple entry format: `item, amount` or `category, item, amount` (for expenses).")
        bulk_text = st.text_area(
            "Paste multiple rows (one per line)",
            placeholder="Milk, 120\nGroceries, Bread, 45\nTaxi/Uber, 200",
            height=160,
        )

    if st.button("Add Transaction"):
        rows_to_add: list[dict] = []

        if entry_mode == "Single":
            if not item.strip():
                st.warning("Please enter item.")
                return
            if amount <= 0:
                st.warning("Amount must be greater than 0.")
                return

            predicted_category = "Income"
            if trx_type == "Expense":
                if category_mode == "ML Suggestion":
                    pred = predict_category(
                        item=item,
                        total_spent=amount,
                        payment_method=payment_method,
                        location=location,
                        bundle_path=bundle_path,
                    )
                    predicted_category = pred["predicted_category"]
                    st.success(
                        f"Predicted category: {predicted_category} (confidence {pred['confidence']:.2f}%)"
                    )
                else:
                    predicted_category = selected_manual_category

            rows_to_add.append(
                {
                    "Type": trx_type,
                    "Item": item.strip(),
                    "Amount": float(amount),
                    "Category": predicted_category,
                    "Payment Method": payment_method,
                    "Location": location,
                    "Date": pd.to_datetime(trx_date),
                }
            )
        else:
            lines = [ln.strip() for ln in bulk_text.splitlines() if ln.strip()]
            if not lines:
                st.warning("Please add at least one row in multiple mode.")
                return

            parse_errors: list[str] = []
            for idx, line in enumerate(lines, start=1):
                parts = [p.strip() for p in line.split(",")]
                local_category = "Income"
                local_item = ""
                local_amount = 0.0

                if trx_type == "Income":
                    if len(parts) != 2:
                        parse_errors.append(f"Line {idx}: use `item, amount`.")
                        continue
                    local_item = parts[0]
                    try:
                        local_amount = float(parts[1])
                    except ValueError:
                        parse_errors.append(f"Line {idx}: invalid amount.")
                        continue
                else:
                    if len(parts) == 2:
                        local_item = parts[0]
                        try:
                            local_amount = float(parts[1])
                        except ValueError:
                            parse_errors.append(f"Line {idx}: invalid amount.")
                            continue
                        local_category = selected_manual_category
                    elif len(parts) == 3:
                        local_category = parts[0] if category_mode == "Manual" else selected_manual_category
                        local_item = parts[1]
                        try:
                            local_amount = float(parts[2])
                        except ValueError:
                            parse_errors.append(f"Line {idx}: invalid amount.")
                            continue
                    else:
                        parse_errors.append(f"Line {idx}: use `item, amount` or `category, item, amount`.")
                        continue

                    if category_mode == "ML Suggestion":
                        pred = predict_category(
                            item=local_item,
                            total_spent=local_amount,
                            payment_method=payment_method,
                            location=location,
                            bundle_path=bundle_path,
                        )
                        local_category = pred["predicted_category"]

                if not local_item:
                    parse_errors.append(f"Line {idx}: item is empty.")
                    continue
                if local_amount <= 0:
                    parse_errors.append(f"Line {idx}: amount must be > 0.")
                    continue

                rows_to_add.append(
                    {
                        "Type": trx_type,
                        "Item": local_item,
                        "Amount": float(local_amount),
                        "Category": local_category,
                        "Payment Method": payment_method,
                        "Location": location,
                        "Date": pd.to_datetime(trx_date),
                    }
                )

            if parse_errors:
                st.warning("Some lines were skipped:\n- " + "\n- ".join(parse_errors[:8]))

        if not rows_to_add:
            st.warning("No valid transaction to add.")
            return

        new_rows_df = pd.DataFrame(rows_to_add)
        st.session_state.transactions = pd.concat(
            [st.session_state.transactions, new_rows_df], ignore_index=True
        )
        for row in rows_to_add:
            save_transaction(row, DB_PATH)

        st.success(f"{len(rows_to_add)} transaction(s) added.")

    trx_df = st.session_state.transactions.copy()
    if not trx_df.empty:
        trx_df["Date"] = pd.to_datetime(trx_df["Date"], errors="coerce")

    expense_df = trx_df[trx_df["Type"] == "Expense"] if not trx_df.empty else pd.DataFrame()
    income_df = trx_df[trx_df["Type"] == "Income"] if not trx_df.empty else pd.DataFrame()

    st.subheader("Current Month Summary")
    manual_income = float(income_df["Amount"].sum()) if not income_df.empty else 0.0
    base_income = float(st.session_state.monthly_income)
    effective_income = base_income + manual_income
    total_expense = float(expense_df["Amount"].sum()) if not expense_df.empty else 0.0
    remaining = effective_income - total_expense

    m1, m2, m3 = st.columns(3)
    m1.metric("Effective Income", f"Rs {effective_income:,.2f}")
    m2.metric("Total Expense", f"Rs {total_expense:,.2f}")
    m3.metric("Remaining", f"Rs {remaining:,.2f}")

    st.subheader("Notifications")
    show_notifications(expense_df, effective_income)

    st.subheader("Your Transactions")
    st.dataframe(trx_df.sort_values("Date", ascending=False) if not trx_df.empty else trx_df, use_container_width=True)

    st.subheader("Database Viewer")
    db_df = load_db_view_transactions(DB_PATH)
    if db_df.empty:
        st.info("No saved transactions in database yet.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            type_filter = st.selectbox("Filter Type", ["All"] + sorted(db_df["Type"].dropna().unique().tolist()))
        with f2:
            category_filter = st.selectbox(
                "Filter Category", ["All"] + sorted(db_df["Category"].dropna().unique().tolist())
            )
        with f3:
            date_range = st.date_input("Filter Date Range", value=())

        filtered_db = db_df.copy()
        if type_filter != "All":
            filtered_db = filtered_db[filtered_db["Type"] == type_filter]
        if category_filter != "All":
            filtered_db = filtered_db[filtered_db["Category"] == category_filter]
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_db = filtered_db[
                (filtered_db["Date"] >= pd.to_datetime(start_date))
                & (filtered_db["Date"] <= pd.to_datetime(end_date))
            ]

        p1, p2 = st.columns(2)
        with p1:
            page_size = int(st.selectbox("Rows per page", [10, 20, 50, 100], index=1))
        total_rows = len(filtered_db)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        with p2:
            page = int(st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1))

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered_db.iloc[start_idx:end_idx].copy()

        st.caption(f"Showing {len(page_df)} of {total_rows} filtered rows (Page {page}/{total_pages})")
        st.dataframe(page_df, use_container_width=True)
        st.download_button(
            "Download Filtered CSV",
            data=filtered_db.to_csv(index=False).encode("utf-8"),
            file_name="filtered_transactions.csv",
            mime="text/csv",
        )

    if not expense_df.empty:
        st.subheader("Expense Charts")

        category_data = (
            expense_df.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
        )
        category_data["AmountLabel"] = category_data["Amount"].map(lambda x: f"Rs {x:,.2f}")

        bar_chart = (
            alt.Chart(category_data)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("Amount:Q", title="Amount (Rs)"),
                y=alt.Y("Category:N", sort="-x", title="Category"),
                color=alt.Color("Category:N", legend=None, scale=alt.Scale(scheme="tealblues")),
                tooltip=[
                    alt.Tooltip("Category:N", title="Category"),
                    alt.Tooltip("Amount:Q", title="Amount", format=",.2f"),
                    alt.Tooltip("AmountLabel:N", title="Display"),
                ],
            )
            .properties(height=380, title="Category-wise Expenses")
        )
        st.altair_chart(bar_chart, use_container_width=True)

        daily_data = (
            expense_df.assign(TxnDate=expense_df["Date"].dt.date)
            .groupby("TxnDate", as_index=False)["Amount"]
            .sum()
        )
        daily_data["TxnDate"] = pd.to_datetime(daily_data["TxnDate"])
        daily_data = daily_data.sort_values("TxnDate")
        daily_data["RollingAvg3"] = daily_data["Amount"].rolling(3, min_periods=1).mean()

        trend_base = alt.Chart(daily_data).encode(
            x=alt.X("TxnDate:T", title="Date"),
            tooltip=[
                alt.Tooltip("TxnDate:T", title="Date"),
                alt.Tooltip("Amount:Q", title="Daily Expense", format=",.2f"),
                alt.Tooltip("RollingAvg3:Q", title="3-Day Avg", format=",.2f"),
            ],
        )
        trend_line = trend_base.mark_line(color="#0b6e4f", strokeWidth=3).encode(
            y=alt.Y("Amount:Q", title="Amount (Rs)")
        )
        trend_points = trend_base.mark_circle(size=70, color="#0b6e4f")
        rolling_line = trend_base.mark_line(color="#f4a261", strokeDash=[6, 4], strokeWidth=2).encode(
            y=alt.Y("RollingAvg3:Q")
        )

        st.altair_chart(
            (trend_line + trend_points + rolling_line).properties(height=360, title="Daily Expense Trend"),
            use_container_width=True,
        )

        pie_chart = (
            alt.Chart(category_data)
            .mark_arc(innerRadius=65, outerRadius=130)
            .encode(
                theta=alt.Theta("Amount:Q"),
                color=alt.Color("Category:N", scale=alt.Scale(scheme="tableau20")),
                tooltip=[
                    alt.Tooltip("Category:N", title="Category"),
                    alt.Tooltip("Amount:Q", title="Amount", format=",.2f"),
                ],
            )
            .properties(height=360, title="Expense Share by Category")
        )
        st.altair_chart(pie_chart, use_container_width=True)

    st.subheader("Historical Dataset Preview")
    st.dataframe(history_df.head(30), use_container_width=True)

    with st.expander("Category -> Item Reference (from dataset)"):
        if not category_item_map:
            st.info("No category-item mapping found in dataset.")
        else:
            for category in sorted(category_item_map.keys()):
                st.markdown(f"**{category}**")
                st.write(", ".join(category_item_map[category]))


if __name__ == "__main__":
    main()
