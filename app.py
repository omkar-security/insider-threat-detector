"""
Privacy-Preserving Audit Log Analysis System - Interactive Demo
================================================================
Streamlit interface for CSC8209 dissertation artefact.

Run with:
    streamlit run app.py

Capabilities:
    1. Dataset overview
    2. K-anonymity anonymiser (configurable k) with formal verification,
       NCP information loss, and export of the anonymised log
    3. Add new employees and new log records into the database
    4. Insider threat detection (Isolation Forest) on the ANONYMISED log,
       with raw-log baseline comparison and per-employee verdict lookup
"""

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, confusion_matrix)

from anonymiser import (anonymise, prepare_log, pseudonymise, verify_k,
                        features_from_anonymised, ANON_ML_COLS, QID_COLS,
                        DEPT_TO_DIVISION, SENIORITY_LEVELS, ACTION_TO_CATEGORY)

st.set_page_config(page_title="Privacy-Preserving Audit Log Analysis",
                   layout="wide", page_icon="🔒")

INSIDER_COL, NORMAL_COL, ACCENT = "#E74C3C", "#2980B9", "#1B4F72"

# Hide Streamlit's "Press Enter to submit form" hint under text inputs
st.markdown(
    "<style>[data-testid='InputInstructions']{display:none;}</style>",
    unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  SIDEBAR - configuration
# ------------------------------------------------------------------ #
st.sidebar.title("🔒 Configuration")
DB_PATH = st.sidebar.text_input(
    "SQLite database path",
    value=r"C:\Users\acer\Documents\study material\dissertation\insider_threat.db")
K_TARGET = st.sidebar.select_slider("Target k (k-anonymity)", options=[2, 5, 10, 20], value=5)
MAX_SUPP = st.sidebar.slider("Max suppression rate", 0.01, 0.10, 0.05, 0.01,
                             help="Fraction of records that may be suppressed "
                                  "to satisfy k before more generalisation is forced.")
st.sidebar.markdown("---")
st.sidebar.caption("CSC8209 · Privacy-Preserving Audit Log Analysis "
                   "for Insider Threat Detection · Newcastle University")


# ------------------------------------------------------------------ #
#  DATA ACCESS
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner="Loading database…")
def load_data(db_path, _version):
    conn = sqlite3.connect(db_path)
    log = pd.read_sql("SELECT * FROM unified_log", conn)
    emps = pd.read_sql("SELECT * FROM employees", conn)
    conn.close()
    return log, emps


@st.cache_data(show_spinner=f"Running anonymisation… (this takes ~30s on 750k rows)")
def run_anonymisation(db_path, k, max_supp, _version):
    log, emps = load_data(db_path, _version)
    res = anonymise(prepare_log(log, emps), k, max_suppression=max_supp)
    # class_sizes Series isn't cache-friendly at scale; keep a summary
    res["class_size_summary"] = res["verification"]["class_sizes"].describe()
    res["class_size_hist"] = np.histogram(
        res["verification"]["class_sizes"].clip(upper=500),
        bins=np.arange(0, 501, 10))   # fixed 0..500 bins, width 10 (narrow bars)
    res["verification"] = {k2: v for k2, v in res["verification"].items()
                           if k2 != "class_sizes"}
    return res


@st.cache_data(show_spinner="Training detection models…")
def run_detection(db_path, k, max_supp, _version):
    log, emps = load_data(db_path, _version)
    emps = emps.copy()
    emps["pseudonym"] = emps["employee_id"].map(pseudonymise)

    # ---- baseline on RAW log (performance ceiling, O2) ------------
    raw = log.copy()
    raw["is_high"] = (raw["sensitivity"] == "High").astype(int)
    raw["is_usb"] = (raw["activity_type"] == "usb").astype(int)
    raw["is_copy"] = (raw["action"] == "copy").astype(int)
    fr = raw.groupby("employee_id").agg(
        after_hours_ratio=("is_after_hours", "mean"),
        external_ratio=("is_external", "mean"),
        avg_data_volume_kb=("data_volume_kb", "mean"),
        max_data_volume_kb=("data_volume_kb", "max"),
        total_data_volume_kb=("data_volume_kb", "sum"),
        high_sensitivity_ratio=("is_high", "mean"),
        usb_events=("is_usb", "sum"),
        transfer_ratio=("is_copy", "mean"),
    ).reset_index()
    fr["usb_per_day"] = fr["usb_events"] / 180
    fr = fr.merge(emps[["employee_id", "is_insider"]], on="employee_id")

    def fit_iso(X, y):
        iso = IsolationForest(contamination=0.05, random_state=42,
                              n_estimators=200).fit(X)
        preds = (iso.predict(X) == -1).astype(int)
        scores = iso.decision_function(X)
        return {
            "preds": preds, "scores": scores,
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "auc": roc_auc_score(y, -scores),
            "roc": roc_curve(y, -scores),
            "cm": confusion_matrix(y, preds),
        }

    Xr = fr[ANON_ML_COLS].fillna(0)
    base = fit_iso(Xr, fr["is_insider"])

    # ---- detection on ANONYMISED log (O4) --------------------------
    res = anonymise(prepare_log(log, emps), k, max_suppression=max_supp)
    fa = features_from_anonymised(res["data"]).merge(
        emps[["pseudonym", "is_insider", "employee_id"]], on="pseudonym")
    Xa = fa[ANON_ML_COLS].fillna(0)
    anon_m = fit_iso(Xa, fa["is_insider"])
    fa["anomaly_score"] = anon_m["scores"]
    fa["predicted"] = anon_m["preds"]

    return base, anon_m, fa, res


def bump_version():
    st.session_state["db_version"] = st.session_state.get("db_version", 0) + 1


VERSION = st.session_state.get("db_version", 0)

st.title("Privacy-Preserving Audit Log Analysis System")
st.caption("Insider threat detection on k-anonymised financial-sector audit logs")

try:
    log_df, emps_df = load_data(DB_PATH, VERSION)
except Exception as e:
    st.error(f"Could not open database at `{DB_PATH}`: {e}")
    st.stop()

tab_over, tab_anon, tab_add, tab_detect = st.tabs(
    ["📊 Dataset Overview", "🛡️ Anonymiser", "➕ Add Employee / Logs", "🚨 Insider Detection"])

# ================================================================== #
#  TAB 1 - OVERVIEW
# ================================================================== #
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Log events", f"{len(log_df):,}")
    c2.metric("Employees", f"{len(emps_df):,}")
    c3.metric("Known insiders", int(emps_df["is_insider"].sum()))
    c4.metric("Distinct resources", f"{log_df['resource'].nunique():,}")

    st.subheader("Raw Log Sample (Identifiable Data)")
    st.dataframe(log_df.head(200), use_container_width=True, height=280)

    cA, cB = st.columns(2)

    def _bar(series, x_title, y_title):
        d = series.rename_axis(x_title).reset_index(name=y_title)
        return alt.Chart(d).mark_bar(color=NORMAL_COL).encode(
            x=alt.X(f"{x_title}:N", sort="-y", title=x_title),
            # scale domainMin=0 keeps the axis from ever dipping below zero
            y=alt.Y(f"{y_title}:Q", title=y_title,
                    scale=alt.Scale(domainMin=0, nice=True)),
            tooltip=[x_title, y_title],
        )

    with cA:
        st.subheader("Events by activity type")
        st.altair_chart(
            _bar(log_df["activity_type"].value_counts(), "activity_type", "events"),
            use_container_width=True)
    with cB:
        st.subheader("Employees by department")
        st.altair_chart(
            _bar(emps_df["department"].value_counts(), "department", "employees"),
            use_container_width=True)

# ================================================================== #
#  TAB 2 - ANONYMISER
# ================================================================== #
with tab_anon:
    st.subheader(f"K-Anonymity Anonymisation (target k = {K_TARGET})")
    st.markdown(
        "This step transforms the identifiable audit log into a privacy-preserving "
        "version that still supports analysis. Four **quasi-identifiers** - the "
        "attributes that could single someone out - are each replaced with a broader "
        "category: **identity → role / seniority tier**, "
        "**exact timestamp → time-of-day window**, "
        "**specific resource → sensitivity class**, and "
        "**exact action → behavioural category**. "
        "The system searches a lattice of generalisation levels and applies the "
        "combination that loses the least information while still meeting the target. "
        "Any residual **equivalence class** (a group sharing the same generalised "
        "values) that is still smaller than *k* is then **suppressed**, so every "
        "remaining record is indistinguishable from at least *k − 1* others.")

    if st.button("▶ Run anonymisation", type="primary"):
        st.session_state["anon_ran"] = True

    if st.session_state.get("anon_ran"):
        res = run_anonymisation(DB_PATH, K_TARGET, MAX_SUPP, VERSION)
        v = res["verification"]

        c1, c2, c3, c4, c5 = st.columns(5)
        ok = v["achieved_k"] >= K_TARGET
        c1.metric("Achieved k (verified)", v["achieved_k"],
                  delta="✓ compliant" if ok else "✗ FAILED")
        c2.metric("Equivalence classes", f"{v['n_classes']:,}")
        c3.metric("NCP (information loss)", f"{res['ncp']:.3f}")
        c4.metric("Records suppressed", f"{res['n_suppressed']:,}",
                  delta=f"{res['suppression_rate']:.2%}", delta_color="inverse")
        c5.metric("Median class size", f"{v['median_class_size']:.0f}")

        # plain-language notes on how each value responds to k / suppression
        c2.caption("(fewer classes as k rises - groups merge to stay ≥ k)")
        c3.caption("(higher as k rises or the suppression cap tightens - "
                   "more generalisation is forced)")
        c4.caption("(more when a higher suppression cap is allowed; "
                   "a tighter cap forces coarser generalisation instead)")

        st.markdown("**Generalisation configuration selected (lowest loss on lattice):**")
        lvl = res["levels"]
        lvl_desc = {
            "identity": ["Department + seniority", "Division + seniority", "Any role"],
            "time":     ["Exact hour", "6-hour window", "Day / night", "Any time"],
            "resource": ["Exact resource", "Sensitivity class", "Any resource"],
            "action":   ["Exact action", "Behavioural category", "Any action"],
        }
        st.table(pd.DataFrame({
            "QID attribute": list(lvl.keys()),
            "Level": [lvl[a] for a in lvl],
            "Meaning": [lvl_desc[a][lvl[a]] for a in lvl],
        }))

        st.subheader("Before vs After Anonymisation")
        st.caption("Columns are lined up so each raw attribute sits directly "
                   "above its generalised counterpart.")
        # matching column layout: raw attribute ↔ its generalised version
        pair_cols = ["Identity", "Role group", "Seniority",
                     "Time", "Resource", "Action"]

        raw_view = (log_df.merge(emps_df[["employee_id", "seniority"]],
                                 on="employee_id", how="left")
                    [["employee_id", "department", "seniority",
                      "timestamp", "resource", "action"]].head(8))
        raw_view.columns = pair_cols

        anon_view = res["data"][["pseudonym", "role_group", "seniority",
                                 "time_window", "resource_class",
                                 "action_class"]].head(8).copy()
        anon_view.columns = pair_cols

        cL, cR = st.columns(2)
        with cL:
            st.caption("RAW (identifiable)")
            st.dataframe(raw_view, use_container_width=True, hide_index=True)
        with cR:
            st.caption(f"ANONYMISED (k ≥ {v['achieved_k']} verified)")
            st.dataframe(anon_view, use_container_width=True, hide_index=True)

        st.subheader("Equivalence Class Size Distribution")
        counts, edges = res["class_size_hist"]
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.bar(edges[:-1], counts, width=np.diff(edges), color=NORMAL_COL,
               edgecolor="white", align="edge")
        ax.axvline(K_TARGET, color=INSIDER_COL, ls="--", lw=2,
                   label=f"k = {K_TARGET}")
        ax.set_xlim(0, 500)
        ax.set_xticks(range(0, 501, 100))
        ax.set_xlabel("Class size (clipped at 500)"); ax.set_ylabel("# classes")
        ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

        # ---- export ------------------------------------------------
        # name encodes both k and the suppression rate used, e.g. k20_supp9
        supp_pct = int(round(MAX_SUPP * 100))
        table_name = f"anonymised_log_k{K_TARGET}_supp{supp_pct}"
        cE1, cE2 = st.columns(2)
        with cE1:
            st.download_button(
                "⬇ Download anonymised log (CSV)",
                res["data"].to_csv(index=False).encode(),
                file_name=f"{table_name}.csv", mime="text/csv")
        with cE2:
            if st.button(f"💾 Save to database as `{table_name}`"):
                conn = sqlite3.connect(DB_PATH)
                res["data"].to_sql(table_name, conn,
                                   if_exists="replace", index=False)
                conn.close()
                st.success(f"Table `{table_name}` written "
                           f"({len(res['data']):,} rows). View it in DBeaver.")

# ================================================================== #
#  TAB 3 - ADD NEW EMPLOYEE / LOG RECORDS
# ================================================================== #
with tab_add:
    st.subheader("Add a New Employee")
    with st.form("new_emp"):
        c1, c2, c3 = st.columns(3)
        first_name = c1.text_input("First name")
        last_name = c2.text_input("Last name")
        role = c3.text_input("Role title", value="Analyst")
        c4, c5, _ = st.columns(3)
        dept = c4.selectbox("Department", sorted(DEPT_TO_DIVISION))
        senior = c5.selectbox("Seniority", SENIORITY_LEVELS)
        submitted = st.form_submit_button("Add employee", type="primary")
    if submitted:
        # require every field before touching the database
        missing = [lbl for lbl, val in
                   [("First name", first_name), ("Last name", last_name),
                    ("Role title", role)] if not val.strip()]
        if missing:
            st.error("Please fill in: " + ", ".join(missing) +
                     ". Employee was not added.")
        else:
            name = f"{first_name.strip()} {last_name.strip()}"
            conn = sqlite3.connect(DB_PATH)
            next_id = pd.read_sql(
                "SELECT MAX(CAST(SUBSTR(employee_id,4) AS INTEGER)) m FROM employees",
                conn)["m"][0] + 1
            new_id = f"EMP{next_id:04d}"
            pd.DataFrame([{
                "employee_id": new_id, "name": name, "department": dept,
                "role": role.strip(), "seniority": senior, "is_insider": 0,
                "pc_id": f"PC-{np.random.randint(1000, 9999)}",
                "email": f"{new_id.lower()}@financeuk-internal.co.uk",
            }]).to_sql("employees", conn, if_exists="append", index=False)
            conn.close()
            bump_version(); st.cache_data.clear()
            st.success(f"Employee **{new_id}** ({name}) added. "
                       f"Pseudonym will be `{pseudonymise(new_id)}`.")

    st.markdown("---")
    st.subheader("Add Log Records for an Employee")
    with st.form("new_log"):
        c1, c2, c3 = st.columns(3)
        emp_sel = c1.selectbox("Employee", emps_df["employee_id"].tolist())
        act_type = c2.selectbox("Activity type",
                                ["logon", "file_access", "email", "web", "usb"])
        action = c3.selectbox("Action", list(ACTION_TO_CATEGORY))
        c4, c5, c6 = st.columns(3)
        resource = c4.text_input("Resource", value="client_records.xlsx")
        sens = c5.selectbox("Sensitivity", ["Low", "Medium", "High"])
        vol = c6.number_input("Data volume (KB)", 0.0, 1e6, 250.0)
        c7, c8, c9 = st.columns(3)
        ts = c7.text_input("Start timestamp (YYYY-MM-DD HH:MM:SS)",
                           value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        external = c8.checkbox("External destination?")
        n_copies = c9.number_input("Repeat this event N times", 1, 500, 1)
        c10, c11, c12 = st.columns(3)
        interval_s = c10.number_input(
            "Interval between repeated events (seconds)", 0, 86400, 60,
            help="Each repeated event is placed this many seconds after the "
                 "previous one, so N events get distinct, realistic timestamps "
                 "instead of all sharing the same instant. Ignored when N = 1.")
        sub2 = st.form_submit_button("Add log record(s)", type="primary")
    if sub2:
        t = pd.to_datetime(ts)
        step = pd.Timedelta(seconds=int(interval_s))
        dept_row = emps_df.loc[emps_df["employee_id"] == emp_sel]
        conn = sqlite3.connect(DB_PATH)
        max_log = pd.read_sql("SELECT MAX(log_id) m FROM unified_log", conn)["m"][0]
        rows = pd.DataFrame([{
            "timestamp": str(t + step * i), "employee_id": emp_sel,
            "department": dept_row["department"].iloc[0],
            "activity_type": act_type, "resource": resource,
            "sensitivity": sens, "action": action,
            "is_after_hours": int((t + step * i).hour < 8 or (t + step * i).hour >= 18),
            "is_weekend": int((t + step * i).weekday() >= 5),
            "data_volume_kb": vol, "is_external": int(external),
            "is_insider": int(dept_row["is_insider"].iloc[0]),
            "is_suspicious": 0, "log_id": max_log + 1 + i,
        } for i in range(int(n_copies))])
        rows.to_sql("unified_log", conn, if_exists="append", index=False)
        conn.close()
        bump_version(); st.cache_data.clear()
        last_ts = str(t + step * (int(n_copies) - 1))
        st.success(f"{n_copies} log record(s) added for {emp_sel} "
                   f"(from {t} to {last_ts}). "
                   "Re-run the Anonymiser / Detection tabs to include them.")

# ================================================================== #
#  TAB 4 - DETECTION
# ================================================================== #
with tab_detect:
    st.subheader(f"Insider Threat Detection on Anonymised Log (k = {K_TARGET})")
    if st.button("▶ Run detection pipeline", type="primary"):
        st.session_state["det_ran"] = True

    if st.session_state.get("det_ran"):
        base, anon_m, fa, res = run_detection(DB_PATH, K_TARGET, MAX_SUPP, VERSION)

        st.markdown("**Privacy-utility tradeoff at this configuration:**")
        comp = pd.DataFrame({
            "Raw log (baseline ceiling)": [base["precision"], base["recall"],
                                           base["f1"], base["auc"]],
            f"Anonymised (k={K_TARGET})": [anon_m["precision"], anon_m["recall"],
                                           anon_m["f1"], anon_m["auc"]],
        }, index=["Precision", "Recall", "F1", "AUC-ROC"]).round(3)
        comp["Utility cost"] = (comp.iloc[:, 1] - comp.iloc[:, 0]).round(3)
        st.dataframe(comp, use_container_width=True)

        # plain-language reading of the anonymised-model metrics (live numbers)
        p, r, f1, auc = (anon_m["precision"], anon_m["recall"],
                         anon_m["f1"], anon_m["auc"])
        st.markdown(
            f"**What these numbers mean (anonymised model, k = {K_TARGET}):**\n\n"
            f"- **Precision ({p:.2f})** - when the system accuses someone, how often "
            f"is it right? Of everyone it flagged, {p:.0%} were true insiders.\n"
            f"- **Recall ({r:.2f})** - of all the real insiders, how many did it "
            f"actually catch? It caught {r:.0%} of them.\n"
            f"- **F1 ({f1:.2f})** - one balanced score blending precision and recall, "
            f"so you can compare setups with a single number.\n"
            f"- **AUC-ROC ({auc:.3f})** - how well the anomaly score ranks insiders "
            f"above normal people. 1.0 = perfect ordering, 0.5 = random guessing; "
            f"{auc:.3f} is {'near-perfect' if auc >= 0.95 else 'strong' if auc >= 0.8 else 'moderate'}.")

        cL, cR = st.columns(2)
        with cL:
            fig, ax = plt.subplots(figsize=(5, 4))
            for m, lab, col in [(base, "Raw baseline", NORMAL_COL),
                                (anon_m, f"Anonymised k={K_TARGET}", INSIDER_COL)]:
                fpr, tpr, _ = m["roc"]
                ax.plot(fpr, tpr, lw=2, color=col,
                        label=f"{lab} (AUC={m['auc']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("ROC Curve: Raw vs Anonymised", color=ACCENT)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            st.pyplot(fig)
        with cR:
            fig, ax = plt.subplots(figsize=(5, 4))
            ns = fa[fa["is_insider"] == 0]["anomaly_score"]
            ins = fa[fa["is_insider"] == 1]["anomaly_score"]
            ax.hist(ns, bins=35, color=NORMAL_COL, alpha=0.7, label="Normal")
            ax.hist(ins, bins=15, color=INSIDER_COL, alpha=0.85, label="Insider")
            ax.set_title("Anomaly scores (anonymised features)", color=ACCENT)
            ax.set_xlabel("Score (lower = more anomalous)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.subheader("Flagged Pseudonyms (Predicted Insiders)")
        flagged = fa[fa["predicted"] == 1].sort_values("anomaly_score")
        st.dataframe(
            flagged[["pseudonym", "employee_id", "anomaly_score",
                     "after_hours_ratio", "external_ratio",
                     "high_sensitivity_ratio", "transfer_ratio",
                     "is_insider"]].round(4),
            use_container_width=True, height=260, hide_index=True)
        st.caption(
            "The `pseudonym` column identifies each flagged entity; the "
            "`employee_id` column re-identifies it back to the real employee. "
            "In a production system this unmasking would sit behind a controlled, "
            "logged procedure and an analyst would normally see pseudonyms only - "
            "it is shown here to evaluate the artefact. `anomaly_score` is the "
            "Isolation Forest score: **lower = more anomalous**, and everything "
            "below the model's threshold is flagged. `is_insider` is the ground "
            "truth (1 = a genuine insider correctly caught, 0 = a false alarm).")

        # -- insiders the model FAILED to flag (false negatives) --------
        st.subheader("Missed Insiders (False Negatives)")
        missed = fa[(fa["is_insider"] == 1) & (fa["predicted"] == 0)] \
            .sort_values("anomaly_score")
        if len(missed):
            st.dataframe(
                missed[["employee_id", "pseudonym", "anomaly_score",
                        "after_hours_ratio", "external_ratio",
                        "high_sensitivity_ratio", "transfer_ratio"]].round(4),
                use_container_width=True, height=200, hide_index=True)
            st.caption(
                f"These **{len(missed)}** genuine insider(s) slipped through at "
                f"k = {K_TARGET}: their anomaly score stayed above the decision "
                "threshold, so they were not flagged. They are typically the "
                "subtlest exfiltrators whose behaviour sits closest to the "
                "busy-analyst and IT-administrator archetypes.")
        else:
            st.success(f"No insiders were missed at k = {K_TARGET} - every "
                       "genuine insider was flagged.")

        # -- normal staff wrongly flagged (false positives) -------------
        st.subheader("False Alarms (False Positives)")
        false_pos = fa[(fa["is_insider"] == 0) & (fa["predicted"] == 1)] \
            .sort_values("anomaly_score")
        if len(false_pos):
            st.dataframe(
                false_pos[["employee_id", "pseudonym", "anomaly_score",
                           "after_hours_ratio", "external_ratio",
                           "high_sensitivity_ratio", "transfer_ratio"]].round(4),
                use_container_width=True, height=200, hide_index=True)
            st.caption(
                f"These **{len(false_pos)}** normal employee(s) were flagged at "
                f"k = {K_TARGET} but are not genuine insiders. They are usually "
                "the legitimate-outlier archetypes - workaholics, IT "
                "administrators with heavy USB use, or heavy data analysts - "
                "whose behaviour naturally overlaps with insider indicators and "
                "creates realistic false-positive pressure.")
        else:
            st.success(f"No false alarms at k = {K_TARGET} - every flagged "
                       "entity was a genuine insider.")

        st.markdown("---")
        st.subheader("🔎 Check a Specific Employee")
        emp_q = st.selectbox("Employee ID", fa["employee_id"].sort_values())
        row = fa[fa["employee_id"] == emp_q].iloc[0]
        verdict = "🚨 FLAGGED AS INSIDER THREAT" if row["predicted"] == 1 \
            else "✅ Normal behaviour"
        st.markdown(f"**{emp_q}** → pseudonym `{row['pseudonym']}` → "
                    f"anomaly score **{row['anomaly_score']:.4f}** → {verdict}")
        st.caption(
            "The **anomaly score** is the Isolation Forest's measure of how unusual "
            "this entity's behaviour is compared with everyone else: **lower (more "
            "negative) = more anomalous**, higher = more typical. An entity is "
            "flagged only when its score falls below the model's decision threshold.")
        st.caption(f"Ground truth: {'insider' if row['is_insider'] else 'normal'} "
                   f"(evaluation only)")
