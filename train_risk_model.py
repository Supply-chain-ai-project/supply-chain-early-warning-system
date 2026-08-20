"""
WEEK 1 -- The prediction foundation.

Trains an XGBoost model to predict late-delivery RISK for a shipment
using ONLY information known BEFORE the shipment goes out.

The critical thing this script does correctly: it DROPS leakage columns
(actual delivery days, delivery status, etc.) that would only be known
AFTER delivery. A real early-warning system doesn't have those yet --
using them gives a fake ~97% accuracy that means nothing operationally.

Run with: python train_risk_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import joblib

print("Loading data...")
df = pd.read_csv("data/dataco_supply_chain.csv", encoding="latin-1")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")

# ---------------------------------------------------------
# TARGET: Late_delivery_risk (1 = late, 0 = on time)
# ---------------------------------------------------------
target = "Late_delivery_risk"

# ---------------------------------------------------------
# LEAKAGE COLUMNS -- these are known ONLY after delivery.
# Using them = cheating. We drop every one.
# ---------------------------------------------------------
leakage_cols = [
    "Days for shipping (real)",   # actual delivery time -- unknown before shipping
    "Delivery Status",            # e.g. "Late delivery" -- directly reveals the answer
    "Order Status",               # post-hoc status
]

# ---------------------------------------------------------
# ID / PII / free-text columns -- not useful for prediction,
# or would cause overfitting to specific customers/orders.
# ---------------------------------------------------------
id_text_cols = [
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

# Only drop columns that actually exist in the file
drop_cols = [c for c in (leakage_cols + id_text_cols) if c in df.columns]
print(f"Dropping {len(drop_cols)} leakage/ID columns to prevent cheating.")
print(f"  Leakage dropped: {[c for c in leakage_cols if c in df.columns]}\n")

df_model = df.drop(columns=drop_cols)

# Drop rows with missing target
df_model = df_model.dropna(subset=[target])

# Separate features and target
y = df_model[target]
X = df_model.drop(columns=[target])

# ---------------------------------------------------------
# Encode categorical columns (turn text into numbers)
# ---------------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Encoding {len(categorical_cols)} categorical columns...")
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Fill any remaining missing numeric values
X = X.fillna(0)

print(f"Final feature count: {X.shape[1]}\n")

# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------
print("Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 55)
print("RESULTS (predicting BEFORE delivery, no leakage)")
print("=" * 55)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 score:  {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["On time", "Late"]))

# ---------------------------------------------------------
# Feature importance -- what drives risk? (useful for explanation agent later)
# ---------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 features driving late-delivery risk:")
print(importances.head(15).to_string())

# ---------------------------------------------------------
# Save model + feature list for later use by the agents
# ---------------------------------------------------------
joblib.dump(model, "risk_model.pkl")
joblib.dump(list(X.columns), "model_features.pkl")
print("\nSaved model to risk_model.pkl and feature list to model_features.pkl")
