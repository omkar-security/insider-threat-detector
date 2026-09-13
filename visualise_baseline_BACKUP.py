"""
Baseline Detection - Isolation Forest Results Visualisation
Produces a clean results dashboard for dissertation and viva demo.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

DB_PATH = r"C:\Users\acer\Documents\study material\dissertation\insider_threat.db"
OUT_DIR = r"C:\Users\acer\Documents\study material\dissertation"

# ── LOAD DATA ─────────────────────────────────────────────────────
conn   = sqlite3.connect(DB_PATH)
df     = pd.read_sql("SELECT * FROM unified_log", conn)
emps   = pd.read_sql("SELECT * FROM employees", conn)
conn.close()

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── FEATURE ENGINEERING ───────────────────────────────────────────
features = df.groupby("employee_id").agg(
    total_events            = ("log_id",          "count"),
    after_hours_ratio       = ("is_after_hours",   "mean"),
    external_ratio          = ("is_external",      "mean"),
    avg_data_volume_kb      = ("data_volume_kb",   "mean"),
    max_data_volume_kb      = ("data_volume_kb",   "max"),
    total_data_volume_kb    = ("data_volume_kb",   "sum"),
    high_sensitivity_ratio  = ("sensitivity",      lambda x: (x=="High").mean()),
    usb_events              = ("activity_type",    lambda x: (x=="usb").sum()),
    copy_ratio              = ("action",           lambda x: (x=="copy").mean()),
).reset_index()

features = features.merge(emps[["employee_id","is_insider","department","seniority"]], on="employee_id")
features["usb_per_day"] = features["usb_events"] / 180

ML_COLS = [
    "after_hours_ratio", "external_ratio", "avg_data_volume_kb",
    "max_data_volume_kb", "high_sensitivity_ratio",
    "usb_per_day", "copy_ratio", "total_data_volume_kb"
]

X      = features[ML_COLS].fillna(0)
y_true = features["is_insider"]

# ── RUN ISOLATION FOREST ──────────────────────────────────────────
iso    = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
iso.fit(X)
scores = iso.decision_function(X)          # higher = more normal
preds  = (iso.predict(X) == -1).astype(int)  # -1 = anomaly = insider

features["anomaly_score"] = scores
features["predicted"]     = preds

# Metrics
precision = precision_score(y_true, preds, zero_division=0)
recall    = recall_score(y_true, preds, zero_division=0)
f1        = f1_score(y_true, preds, zero_division=0)
auc       = roc_auc_score(y_true, -scores)
fpr, tpr, _ = roc_curve(y_true, -scores)
cm        = confusion_matrix(y_true, preds)

print("ISOLATION FOREST RESULTS")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1 Score  : {f1:.3f}")
print(f"AUC-ROC   : {auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_true, preds, target_names=["Normal","Insider"]))

# ── PLOT DASHBOARD ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor="#F8F9FA")
fig.suptitle(
    "Baseline Insider Threat Detection — Isolation Forest Results\n"
    "UK Financial Sector Synthetic Audit Log Dataset  |  500 Employees  |  6 Months",
    fontsize=15, fontweight="bold", color="#1B2631", y=0.98
)

gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

INSIDER_COL = "#E74C3C"
NORMAL_COL  = "#2980B9"
ACCENT      = "#1B4F72"

# ── PANEL 1: Metrics Summary ───────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor("#1B4F72")
metrics = {"Precision": precision, "Recall": recall, "F1 Score": f1, "AUC-ROC": auc}
for i, (label, val) in enumerate(metrics.items()):
    y_pos = 0.78 - i * 0.20
    ax0.text(0.5, y_pos + 0.06, label, transform=ax0.transAxes,
             ha="center", fontsize=11, color="#AED6F1", fontweight="bold")
    ax0.text(0.5, y_pos - 0.02, f"{val:.3f}", transform=ax0.transAxes,
             ha="center", fontsize=20, color="white", fontweight="bold")
ax0.set_xlim(0,1); ax0.set_ylim(0,1)
ax0.axis("off")
ax0.set_title("Performance Metrics", fontsize=11, fontweight="bold", color=ACCENT, pad=8)

# ── PANEL 2: Confusion Matrix ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal","Insider"])
disp.plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title("Confusion Matrix", fontsize=11, fontweight="bold", color=ACCENT, pad=8)
ax1.set_xlabel("Predicted Label", fontsize=9)
ax1.set_ylabel("True Label", fontsize=9)

# ── PANEL 3: ROC Curve ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(fpr, tpr, color=INSIDER_COL, lw=2.5, label=f"AUC = {auc:.3f}")
ax2.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")
ax2.fill_between(fpr, tpr, alpha=0.08, color=INSIDER_COL)
ax2.set_xlabel("False Positive Rate", fontsize=9)
ax2.set_ylabel("True Positive Rate", fontsize=9)
ax2.set_title("ROC Curve", fontsize=11, fontweight="bold", color=ACCENT, pad=8)
ax2.legend(fontsize=9)
ax2.set_facecolor("#FDFEFE")
ax2.grid(alpha=0.3)

# ── PANEL 4: Anomaly Score Distribution ───────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
normal_scores  = features[features["is_insider"]==0]["anomaly_score"]
insider_scores = features[features["is_insider"]==1]["anomaly_score"]
ax3.hist(normal_scores,  bins=35, color=NORMAL_COL,  alpha=0.7, label="Normal Employees", edgecolor="white")
ax3.hist(insider_scores, bins=15, color=INSIDER_COL, alpha=0.85, label="Insider Threats",  edgecolor="white")
threshold_val = np.percentile(scores, 5)
ax3.axvline(x=threshold_val, color="black", linestyle="--", lw=2, label=f"Decision Threshold ({threshold_val:.3f})")
ax3.set_xlabel("Anomaly Score  (lower = more anomalous)", fontsize=10)
ax3.set_ylabel("Number of Employees", fontsize=10)
ax3.set_title("Anomaly Score Distribution — Insiders vs Normal Employees", fontsize=11, fontweight="bold", color=ACCENT)
ax3.legend(fontsize=9)
ax3.set_facecolor("#FDFEFE")
ax3.grid(alpha=0.3)

# ── PANEL 5: Feature Importance (mean difference) ─────────────────
ax4 = fig.add_subplot(gs[1, 2])
feat_importance = {}
for col in ML_COLS:
    ins_mean = features[features["is_insider"]==1][col].mean()
    nor_mean = features[features["is_insider"]==0][col].mean()
    if nor_mean > 0:
        feat_importance[col] = (ins_mean - nor_mean) / nor_mean * 100
    else:
        feat_importance[col] = 0

fi_df = pd.Series(feat_importance).sort_values()
labels_clean = {
    "after_hours_ratio":      "After-Hours Ratio",
    "external_ratio":         "External Activity Ratio",
    "avg_data_volume_kb":     "Avg Data Volume",
    "max_data_volume_kb":     "Max Data Volume",
    "high_sensitivity_ratio": "High Sensitivity Ratio",
    "usb_per_day":            "USB Usage/Day",
    "copy_ratio":             "Copy Action Ratio",
    "total_data_volume_kb":   "Total Data Volume",
}
fi_df.index = [labels_clean.get(i,i) for i in fi_df.index]
colours = [INSIDER_COL if v > 0 else NORMAL_COL for v in fi_df.values]
fi_df.plot(kind="barh", ax=ax4, color=colours, edgecolor="white")
ax4.set_title("Feature Uplift\n(Insiders vs Normal, %)", fontsize=11, fontweight="bold", color=ACCENT)
ax4.set_xlabel("% Increase vs Normal", fontsize=9)
ax4.axvline(0, color="black", lw=0.8)
ax4.set_facecolor("#FDFEFE")
ax4.grid(alpha=0.3, axis="x")
ax4.tick_params(axis="y", labelsize=8)

# ── PANEL 6: PCA scatter ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
normal_mask  = features["is_insider"] == 0
insider_mask = features["is_insider"] == 1
ax5.scatter(X_pca[normal_mask, 0],  X_pca[normal_mask, 1],
            c=NORMAL_COL,  alpha=0.4, s=25, label="Normal Employees", edgecolors="none")
ax5.scatter(X_pca[insider_mask, 0], X_pca[insider_mask, 1],
            c=INSIDER_COL, alpha=0.9, s=80, label="Insider Threats",
            edgecolors="#922B21", linewidth=0.8, marker="D")
ax5.set_title(
    f"PCA Projection — Behavioural Feature Space\n"
    f"(Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})",
    fontsize=11, fontweight="bold", color=ACCENT
)
ax5.set_xlabel("Principal Component 1", fontsize=9)
ax5.set_ylabel("Principal Component 2", fontsize=9)
ax5.legend(fontsize=9)
ax5.set_facecolor("#FDFEFE")
ax5.grid(alpha=0.3)

# ── PANEL 7: Top Flagged Employees ────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
top_flagged = features.nsmallest(10, "anomaly_score")[
    ["employee_id", "anomaly_score", "is_insider", "department"]
].copy()
top_flagged["anomaly_score"] = top_flagged["anomaly_score"].round(4)
top_flagged["status"] = top_flagged["is_insider"].map({1:"INSIDER", 0:"NORMAL"})
bar_colours = [INSIDER_COL if x==1 else NORMAL_COL for x in top_flagged["is_insider"]]
bars = ax6.barh(top_flagged["employee_id"], -top_flagged["anomaly_score"],
                color=bar_colours, edgecolor="white")
ax6.set_title("Top 10 Most Anomalous\nEmployees", fontsize=11, fontweight="bold", color=ACCENT)
ax6.set_xlabel("Anomaly Score (higher bar = more anomalous)", fontsize=8)
ax6.set_facecolor("#FDFEFE")
ax6.grid(alpha=0.3, axis="x")
ax6.tick_params(axis="y", labelsize=8)
insider_patch = mpatches.Patch(color=INSIDER_COL, label="Confirmed Insider")
normal_patch  = mpatches.Patch(color=NORMAL_COL,  label="Normal Employee")
ax6.legend(handles=[insider_patch, normal_patch], fontsize=7, loc="lower right")

plt.savefig(f"{OUT_DIR}/baseline_results_dashboard.png",
            dpi=180, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"\nDashboard saved to: {OUT_DIR}baseline_results_dashboard.png")
