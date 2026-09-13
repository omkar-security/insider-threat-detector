"""
O4 — Privacy-Utility Tradeoff Experiment
=========================================
Runs the full anonymise -> detect pipeline at k = 2, 5, 10, 20,
records detection performance (F1, AUC-ROC) and information loss (NCP),
and saves the tradeoff curve figure + results CSV for the dissertation.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from anonymiser import (anonymise, prepare_log, pseudonymise,
                        features_from_anonymised, ANON_ML_COLS)

DB_PATH = r"C:\Users\acer\Documents\study material\dissertation\insider_threat.db"
OUT_DIR = r"C:\Users\acer\Documents\study material\dissertation"
K_VALUES = [2, 5, 10, 20]

conn = sqlite3.connect(DB_PATH)
log = pd.read_sql("SELECT * FROM unified_log", conn)
emps = pd.read_sql("SELECT * FROM employees", conn)
conn.close()

log = prepare_log(log, emps)
emps["pseudonym"] = emps["employee_id"].map(pseudonymise)

results = []
for k in K_VALUES:
    print(f"\n=== k = {k} ===")
    res = anonymise(log, k)
    feats = features_from_anonymised(res["data"]).merge(
        emps[["pseudonym", "is_insider"]], on="pseudonym")
    X, y = feats[ANON_ML_COLS].fillna(0), feats["is_insider"]
    iso = IsolationForest(contamination=0.05, random_state=42,
                          n_estimators=200).fit(X)
    preds = (iso.predict(X) == -1).astype(int)
    scores = iso.decision_function(X)
    row = {
        "k": k,
        "achieved_k": res["verification"]["achieved_k"],
        "levels": str(res["levels"]),
        "ncp": res["ncp"],
        "suppression_rate": res["suppression_rate"],
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "auc": roc_auc_score(y, -scores),
    }
    results.append(row)
    print({k2: (round(v, 3) if isinstance(v, float) else v)
           for k2, v in row.items()})

df = pd.DataFrame(results)
df.to_csv(f"{OUT_DIR}/tradeoff_results.csv", index=False)

# ── Tradeoff curve ────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 5.5), facecolor="#F8F9FA")
ax1.plot(df["k"], df["f1"], "o-", color="#2980B9", lw=2.5, ms=9, label="F1 (detection utility)")
ax1.plot(df["k"], df["auc"], "s--", color="#1B4F72", lw=2, ms=8, label="AUC-ROC")
ax1.set_xlabel("k (anonymity level)", fontsize=11)
ax1.set_ylabel("Detection performance", fontsize=11, color="#1B4F72")
ax1.set_ylim(0, 1.05)
ax1.set_xticks(K_VALUES)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df["k"], df["ncp"], "^-", color="#E74C3C", lw=2.5, ms=9, label="NCP (information loss)")
ax2.plot(df["k"], df["suppression_rate"], "v:", color="#922B21", lw=2, ms=8, label="Suppression rate")
ax2.set_ylabel("Privacy cost", fontsize=11, color="#E74C3C")
ax2.set_ylim(0, max(0.3, df["ncp"].max() * 1.4))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
plt.title("Privacy–Utility Tradeoff Across k-Anonymity Configurations",
          fontsize=13, fontweight="bold", color="#1B2631")
plt.savefig(f"{OUT_DIR}/tradeoff_curve.png", dpi=180,
            bbox_inches="tight", facecolor="#F8F9FA")
print(f"\nSaved: {OUT_DIR}/tradeoff_results.csv and tradeoff_curve.png")
