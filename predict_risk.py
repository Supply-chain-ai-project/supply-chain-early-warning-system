"""
WEEK 1 -> WEEK 2 bridge (FIXED encoding).

Scores shipments for late-delivery risk. Fixed the single-row encoding bug:
we now build the full encoded frame ONCE and index into it, so categorical
features (shipping mode, region) are preserved correctly.

Run with: python predict_risk.py
"""

import pandas as pd
import joblib

_model = joblib.load("risk_model.pkl")
_model_features = joblib.load("model_features.pkl")

_leakage_cols = ["Days for shipping (real)", "Delivery Status", "Order Status"]
_id_text_cols = [
    "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
    "Customer Id", "Customer Zipcode", "Customer Street",
    "Order Id", "Order Item Id", "Order Customer Id", "Product Card Id",
    "Product Image", "Product Description",
    "Order Zipcode", "Latitude", "Longitude",
    "Category Id", "Department Id", "Order Item Cardprod Id",
    "Product Category Id", "Order Profit Per Order",
    "shipping date (DateOrders)", "order date (DateOrders)",
    "Order Item Total", "Sales", "Order Item Product Price",
]
_target = "Late_delivery_risk"

_operational_prefixes = (
    "Days for shipment (scheduled)", "Shipping Mode", "Order Region",
    "Market", "Type", "Category Name",
)


def prepare_frame(df):
    """Encode a WHOLE dataframe the same way training did. Returns aligned X."""
    drop_cols = [c for c in (_leakage_cols + _id_text_cols + [_target]) if c in df.columns]
    X = df.drop(columns=drop_cols)
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.reindex(columns=_model_features, fill_value=0)
    return X


def score_frame(df):
    """Score every row in a dataframe. Returns list of result dicts."""
    X = prepare_frame(df)
    probas = _model.predict_proba(X)[:, 1]
    importances = pd.Series(_model.feature_importances_, index=_model_features)

    results = []
    for i in range(len(X)):
        proba = float(probas[i])
        if proba >= 0.60:
            level = "HIGH"
        elif proba >= 0.45:
            level = "MEDIUM"
        else:
            level = "LOW"

        active = X.columns[(X.iloc[i] != 0).values]
        operational_active = [f for f in active if any(f.startswith(p) for p in _operational_prefixes)]
        reasons = (
            importances[operational_active].sort_values(ascending=False).head(5).index.tolist()
            if operational_active else
            importances[active].sort_values(ascending=False).head(5).index.tolist()
        )
        results.append({
            "risk_probability": round(proba, 3),
            "risk_level": level,
            "top_reasons": reasons,
        })
    return results


def get_shipment_context(row):
    def safe(col):
        return row[col] if col in row and pd.notna(row[col]) else "N/A"
    return {
        "product": safe("Category Name"),
        "customer_city": safe("Order City") if ("Order City" in row and pd.notna(row.get("Order City", None))) else safe("Customer City"),
        "order_region": safe("Order Region"),
        "shipping_mode": safe("Shipping Mode"),
        "scheduled_days": safe("Days for shipment (scheduled)"),
        "market": safe("Market"),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/dataco_supply_chain.csv", encoding="latin-1")

    # Deliberately sample ACROSS shipping modes so we see the full risk range
    samples = []
    for mode in ["First Class", "Second Class", "Same Day", "Standard Class"]:
        samples.append(df[df["Shipping Mode"] == mode].sample(2, random_state=1))
    sample = pd.concat(samples).reset_index(drop=True)

    results = score_frame(sample)

    print("Scoring 2 shipments from EACH shipping mode:\n")
    print("(For reference, actual late rates: First 95%, Second 77%, Same Day 46%, Standard 38%)\n")
    high = med = low = 0
    for i in range(len(sample)):
        ctx = get_shipment_context(sample.iloc[i])
        r = results[i]
        level = r["risk_level"]
        high += level == "HIGH"; med += level == "MEDIUM"; low += level == "LOW"
        print("=" * 55)
        print(f"{ctx['product']} -> {ctx['customer_city']} ({ctx['order_region']})")
        print(f"Mode: {ctx['shipping_mode']} | Scheduled: {ctx['scheduled_days']} days")
        print(f"  RISK: {r['risk_probability']*100:.1f}%  [{level}]")
        print(f"  Top reasons: {r['top_reasons']}")
        print()
    print("=" * 55)
    print(f"Distribution: {high} HIGH, {med} MEDIUM, {low} LOW")