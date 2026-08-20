"""
BATCH EVALUATION -- produces the numbers your paper needs.

Measures three things:
  1. Prediction quality at scale (precision/recall/AUC on a big test set)
  2. Early-warning value: of shipments flagged HIGH risk, how many were ACTUALLY late?
     (This is the real operational value -- a warning that's usually right is useful.)
  3. Risk-tier calibration: do HIGH/MEDIUM/LOW tiers actually separate real late rates?

Run with: python evaluate.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from predict_risk import prepare_frame

print("Loading data + model...")
df = pd.read_csv("data/dataco_supply_chain.csv", encoding="latin-1")
model = joblib.load("risk_model.pkl")

target = "Late_delivery_risk"
df = df.dropna(subset=[target])

# Recreate the SAME test split used in training (same random_state=42)
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
print(f"Evaluating on {len(test_df)} held-out shipments.\n")

# Score the whole test set
X_test = prepare_frame(test_df)
probas = model.predict_proba(X_test)[:, 1]
y_true = test_df[target].values

# Assign risk tiers using the same thresholds as the dashboard
def tier(p):
    if p >= 0.60: return "HIGH"
    if p >= 0.45: return "MEDIUM"
    return "LOW"

test_df = test_df.copy()
test_df["risk_proba"] = probas
test_df["risk_tier"] = [tier(p) for p in probas]

# ---------------------------------------------------------
# 1. Overall prediction quality
# ---------------------------------------------------------
preds = (probas >= 0.5).astype(int)
print("=" * 55)
print("1. PREDICTION QUALITY (held-out test set)")
print("=" * 55)
print(f"ROC-AUC:   {roc_auc_score(y_true, probas):.3f}")
print(f"Precision: {precision_score(y_true, preds):.3f}")
print(f"Recall:    {recall_score(y_true, preds):.3f}")
print(f"F1 score:  {f1_score(y_true, preds):.3f}")
print()

# ---------------------------------------------------------
# 2. Early-warning value: precision of HIGH-risk flags
# ---------------------------------------------------------
print("=" * 55)
print("2. EARLY-WARNING VALUE")
print("=" * 55)
high_risk = test_df[test_df["risk_tier"] == "HIGH"]
if len(high_risk) > 0:
    actually_late = high_risk[target].mean()
    print(f"Shipments flagged HIGH risk: {len(high_risk)}")
    print(f"Of those, actually late:    {actually_late*100:.1f}%")
    print(f"  -> When the system raises a HIGH warning, it's right {actually_late*100:.0f}% of the time.")
print()

# ---------------------------------------------------------
# 3. Do the tiers actually separate real late rates?
# ---------------------------------------------------------
print("=" * 55)
print("3. RISK-TIER CALIBRATION")
print("=" * 55)
print("(A good system: HIGH tier = high real late rate, LOW tier = low real late rate)\n")
tier_summary = test_df.groupby("risk_tier").agg(
    shipments=("risk_tier", "count"),
    actual_late_rate=(target, "mean"),
).reindex(["HIGH", "MEDIUM", "LOW"])
tier_summary["actual_late_rate"] = (tier_summary["actual_late_rate"] * 100).round(1)
print(tier_summary.to_string())
print()

# Save the scored test set for the partner's charts
test_df[[target, "risk_proba", "risk_tier", "Shipping Mode", "Order Region",
         "Category Name"]].to_csv("evaluation_results.csv", index=False)
print("Saved scored results to evaluation_results.csv (for charts/paper).")
