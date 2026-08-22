"""
CHARTS for the paper's results section.

Generates publication-ready figures from evaluation_results.csv:
  1. Risk-tier calibration bar chart (HIGH/MEDIUM/LOW vs real late rate)
  2. Risk probability distribution (histogram)
  3. Late rate by shipping mode (the strongest real-world signal)
  4. Explanation quality scores (faithfulness + clarity)

Run with: python make_charts.py
Saves PNGs to a charts/ folder.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 11

df = pd.read_csv("evaluation_results.csv")
target = "Late_delivery_risk"

# ---------------------------------------------------------
# 1. Risk-tier calibration
# ---------------------------------------------------------
tier_rates = df.groupby("risk_tier")[target].mean().reindex(["HIGH", "MEDIUM", "LOW"]) * 100
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#e53935", "#fb8c00", "#43a047"]
bars = ax.bar(tier_rates.index, tier_rates.values, color=colors)
ax.set_ylabel("Actual late-delivery rate (%)")
ax.set_title("Risk-Tier Calibration:\nPredicted Tiers vs. Real Late Rates")
ax.set_ylim(0, 100)
for bar, val in zip(bars, tier_rates.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("charts/1_risk_tier_calibration.png", bbox_inches="tight")
plt.close()
print("Saved charts/1_risk_tier_calibration.png")

# ---------------------------------------------------------
# 2. Risk probability distribution
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["risk_proba"], bins=40, color="#1e88e5", edgecolor="white")
ax.axvline(0.60, color="#e53935", linestyle="--", label="HIGH threshold (0.60)")
ax.axvline(0.45, color="#fb8c00", linestyle="--", label="MEDIUM threshold (0.45)")
ax.set_xlabel("Predicted risk probability")
ax.set_ylabel("Number of shipments")
ax.set_title("Distribution of Predicted Risk Scores")
ax.legend()
plt.tight_layout()
plt.savefig("charts/2_risk_distribution.png", bbox_inches="tight")
plt.close()
print("Saved charts/2_risk_distribution.png")

# ---------------------------------------------------------
# 3. Late rate by shipping mode
# ---------------------------------------------------------
if "Shipping Mode" in df.columns:
    mode_rates = df.groupby("Shipping Mode")[target].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(mode_rates.index, mode_rates.values, color="#5e35b1")
    ax.set_ylabel("Actual late-delivery rate (%)")
    ax.set_title("Late-Delivery Rate by Shipping Mode")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)
    for bar, val in zip(bars, mode_rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.0f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("charts/3_late_by_shipping_mode.png", bbox_inches="tight")
    plt.close()
    print("Saved charts/3_late_by_shipping_mode.png")

# ---------------------------------------------------------
# 4. Explanation quality (hardcoded from your eval run)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
metrics = ["Faithfulness", "Clarity"]
scores = [4.50, 5.00]
bars = ax.bar(metrics, scores, color=["#00897b", "#3949ab"])
ax.set_ylim(0, 5)
ax.set_ylabel("Average score (1-5)")
ax.set_title("AI Explanation Quality\n(LLM-as-Judge, n=10)")
for bar, val in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("charts/4_explanation_quality.png", bbox_inches="tight")
plt.close()
print("Saved charts/4_explanation_quality.png")

print("\nAll charts saved to the charts/ folder.")
