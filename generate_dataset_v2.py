"""
Synthetic Insider Threat Dataset Generator — v2 "Subtle Insiders"
==================================================================
Regenerates insider_threat.db so the privacy-utility tradeoff curve
tells an interesting story:

  * Heterogeneous NORMAL population (workaholics, IT admins, heavy
    analysts, external-facing staff) -> false-positive pressure, so
    the raw baseline lands ~0.75-0.9 F1 instead of a perfect 1.0.
  * SUBTLE insiders: small exfiltration chunks, mostly inside office
    hours, drawn from rare resource names and uncommon QID combos ->
    their malicious rows sit in small equivalence classes, so higher
    k suppresses proportionally more of their signal.

Schema is identical to v1 (tables `employees`, `unified_log`) so all
existing scripts (visualise_baseline.py, anonymiser.py, app.py,
tradeoff_experiment.py) run unchanged.
"""

import sqlite3
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

OUT_DB = "insider_threat.db"          # written to current directory
N_EMPLOYEES = 500
N_INSIDERS = 25
START = pd.Timestamp("2024-01-01")
N_DAYS = 180

# ------------------------------------------------------------------ #
#  TUNING KNOBS (found empirically so the curve bends)
# ------------------------------------------------------------------ #
MALICIOUS_EVENT_FRAC = 0.22   # fraction of an insider's events that are malicious
INSIDER_AFTER_HOURS_BUMP = 0.07
INSIDER_EXTERNAL_BUMP = 0.10
MALICIOUS_VOL_MULT = (3.0, 6.0)   # small-chunk exfil, x personal baseline
MALICIOUS_EVENING_P = 0.48        # 60% of malicious acts hide in office hours

# ------------------------------------------------------------------ #
DEPARTMENTS = {
    "Investment Banking": 0.16, "Retail Banking": 0.16, "Finance": 0.12,
    "Risk Management": 0.12, "Compliance": 0.10, "IT": 0.14,
    "HR": 0.08, "Operations": 0.12,
}
ROLES = {
    "Investment Banking": ["IB Analyst", "Associate", "VP", "Trader"],
    "Retail Banking": ["Branch Advisor", "Account Manager", "Teller Supervisor"],
    "Finance": ["Financial Accountant", "FP&A Analyst", "Treasury Analyst"],
    "Risk Management": ["Risk Analyst", "Credit Risk Officer", "Model Validator"],
    "Compliance": ["Compliance Officer", "AML Analyst", "Regulatory Advisor"],
    "IT": ["Systems Engineer", "DevOps Engineer", "Security Analyst", "DBA"],
    "HR": ["HR Advisor", "Recruiter", "HR Director"],
    "Operations": ["Operations Analyst", "Settlements Officer", "Process Manager"],
}
SENIORITY_P = {"Junior": 0.4, "Mid-level": 0.42, "Senior": 0.18}
DEPT_HIGH_SENS = {   # baseline probability that an accessed resource is High
    "Investment Banking": 0.16, "Retail Banking": 0.08, "Finance": 0.14,
    "Risk Management": 0.18, "Compliance": 0.22, "IT": 0.12,
    "HR": 0.15, "Operations": 0.07,
}

FIRST = ["James", "Olivia", "Mohamed", "Amelia", "Arjun", "Sophie", "Wei",
         "Isla", "Tomasz", "Grace", "Daniel", "Priya", "Ethan", "Zara",
         "Lucas", "Freya", "Kwame", "Hannah", "Omar", "Ella", "Noah",
         "Aisha", "George", "Maya", "Ryan", "Chloe", "Ivan", "Nadia"]
LAST = ["Smith", "Patel", "Jones", "Khan", "Williams", "Chen", "Taylor",
        "Novak", "Brown", "Okafor", "Davies", "Kowalski", "Evans", "Ali",
        "Wilson", "Nguyen", "Thomas", "Ivanov", "Johnson", "Osei",
        "Roberts", "Hussain", "Walker", "Zhang", "Wright", "Bennett"]

# Shared resource pools (so equivalence classes are healthy at low k)
WEBSITES_INT = ["financeuk-internal.co.uk", "sharepoint.com",
                "teams.microsoft.com", "office365.com"]
WEBSITES_EXT = ["bbc.co.uk", "gov.uk", "theguardian.com", "google.co.uk",
                "reuters.com", "ft.com", "linkedin.com", "bbc.co.uk/weather",
                "microsoft.com", "bloomberg.com", "hmrc.gov.uk"]
WEBSITES_RISKY = ["gmail.com", "dropbox.com", "wetransfer.com",
                  "pastebin.com", "mega.nz", "protonmail.com"]
FILES_COMMON = [f"{n}.{e}" for n in
                ["policy_update", "quarterly_report", "meeting_minutes",
                 "budget_2024", "team_rota", "training_pack", "expenses",
                 "board_pack", "risk_register", "audit_plan", "kyc_checklist",
                 "market_summary", "hr_handbook", "it_runbook",
                 "client_onboarding", "product_terms"]
                for e in ["pdf", "docx", "xlsx"]]
FILES_SENSITIVE = [f"{n}.xlsx" for n in
                   ["client_records", "salary_bandings", "trading_positions",
                    "merger_target_analysis", "counterparty_exposures",
                    "sar_filings", "customer_pii_extract", "deal_pipeline"]]
# rare, insider-favoured resources -> land in small equivalence classes
FILES_RARE = [f"export_batch_{i:02d}.csv" for i in range(1, 26)] + \
             [f"db_dump_{i:02d}.sql" for i in range(1, 15)]


def make_employees():
    depts = rng.choice(list(DEPARTMENTS), N_EMPLOYEES, p=list(DEPARTMENTS.values()))
    seniors = rng.choice(list(SENIORITY_P), N_EMPLOYEES, p=list(SENIORITY_P.values()))
    rows = []
    for i in range(N_EMPLOYEES):
        eid = f"EMP{i+1:04d}"
        rows.append({
            "employee_id": eid,
            "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "department": depts[i],
            "role": rng.choice(ROLES[depts[i]]),
            "seniority": seniors[i],
            "is_insider": 0,
            "pc_id": f"PC-{rng.integers(1000, 9999)}",
            "email": f"{eid.lower()}@financeuk-internal.co.uk",
        })
    emps = pd.DataFrame(rows)
    emps.loc[rng.choice(N_EMPLOYEES, N_INSIDERS, replace=False), "is_insider"] = 1
    return emps


def persona(dept, seniority, is_insider):
    """Per-person behavioural parameters. Archetypes create legitimate
    outliers among normals; insiders get only SUBTLE bumps."""
    p = {
        "events_day": max(4, rng.normal(9, 2)),
        "after_hours_p": np.clip(rng.beta(2, 20), 0.01, 0.5),     # ~9%
        "weekend_p": 0.05,
        "external_p": np.clip(rng.beta(3, 16), 0.02, 0.6),        # ~16%
        "usb_day": 0.0,
        "copy_p": np.clip(rng.beta(2, 16), 0.01, 0.5),            # ~11%
        "high_sens_p": np.clip(rng.normal(DEPT_HIGH_SENS[dept], 0.04), 0.01, 0.5),
        "vol_mu": rng.normal(5.0, 0.35),   # lognormal params, ~150-250 KB
        "vol_sig": 0.9,
        "risky_web_p": 0.01,
        "archetype": "regular",
    }
    r = rng.random()
    if r < 0.08:                                  # workaholic
        p.update(after_hours_p=np.clip(rng.normal(0.20, 0.04), 0.12, 0.35),
                 weekend_p=0.18, events_day=p["events_day"] * 1.3,
                 archetype="workaholic")
    elif r < 0.16 and dept == "IT":               # IT admin / on-call
        p.update(usb_day=rng.uniform(0.15, 0.5),
                 after_hours_p=np.clip(rng.normal(0.22, 0.05), 0.1, 0.5),
                 high_sens_p=min(0.5, p["high_sens_p"] + 0.08),
                 archetype="it_admin")
    elif r < 0.24:                                # heavy data analyst
        p.update(copy_p=np.clip(rng.normal(0.24, 0.04), 0.14, 0.4),
                 vol_mu=p["vol_mu"] + 0.55, archetype="analyst")
    elif r < 0.32:                                # external-facing
        p.update(external_p=np.clip(rng.normal(0.32, 0.05), 0.2, 0.5),
                 risky_web_p=0.03, archetype="external_facing")
    elif r < 0.36:                                # occasional USB user
        p.update(usb_day=rng.uniform(0.05, 0.2), archetype="usb_user")

    if is_insider:                                # SUBTLE drift only
        p["after_hours_p"] = min(0.55, p["after_hours_p"] + INSIDER_AFTER_HOURS_BUMP)
        p["external_p"] = min(0.7, p["external_p"] + INSIDER_EXTERNAL_BUMP)
        p["copy_p"] = min(0.6, p["copy_p"] + 0.05)
    return p


def sample_hours(n, after_hours_p, weekend_p):
    days = rng.integers(0, N_DAYS, n)
    dow = (START + pd.to_timedelta(days, "D")).dayofweek
    # push most events to weekdays
    wk_mask = dow >= 5
    move = wk_mask & (rng.random(n) > weekend_p * 4)
    days[move] = np.maximum(0, days[move] - rng.integers(1, 3, move.sum()))
    after = rng.random(n) < after_hours_p
    hours = np.where(after,
                     rng.choice([5, 6, 7, 19, 20, 21, 22, 23, 0], n),
                     np.clip(rng.normal(13, 3, n).astype(int), 8, 17))
    ts = (START + pd.to_timedelta(days, "D")
          + pd.to_timedelta(hours, "h")
          + pd.to_timedelta(rng.integers(0, 60, n), "m")
          + pd.to_timedelta(rng.integers(0, 60, n), "s"))
    return ts


def gen_normal_events(emp, p):
    n = int(p["events_day"] * N_DAYS * rng.uniform(0.9, 1.1))
    ts = sample_hours(n, p["after_hours_p"], p["weekend_p"])

    kind = rng.choice(["logon", "web", "file_access", "email", "usb"],
                      n, p=_activity_mix(p))
    resource = np.empty(n, dtype=object)
    action = np.empty(n, dtype=object)
    sens = np.empty(n, dtype=object)
    external = np.zeros(n, dtype=int)

    for i, k in enumerate(kind):
        if k == "logon":
            resource[i], action[i], sens[i] = "system", "logon", "Low"
        elif k == "web":
            if rng.random() < p["risky_web_p"]:
                resource[i] = rng.choice(WEBSITES_RISKY); external[i] = 1
            elif rng.random() < p["external_p"]:
                resource[i] = rng.choice(WEBSITES_EXT); external[i] = 1
            else:
                resource[i] = rng.choice(WEBSITES_INT)
            action[i], sens[i] = "visit", "Low"
        elif k == "file_access":
            hi = rng.random() < p["high_sens_p"]
            resource[i] = rng.choice(FILES_SENSITIVE if hi else FILES_COMMON)
            sens[i] = "High" if hi else rng.choice(["Low", "Medium"], p=[0.6, 0.4])
            action[i] = ("copy" if rng.random() < p["copy_p"]
                         else rng.choice(["read", "open", "print"], p=[0.5, 0.35, 0.15]))
        elif k == "email":
            ext = rng.random() < p["external_p"] * 0.6
            resource[i] = rng.choice(WEBSITES_RISKY[:2]) if (ext and rng.random() < 0.05) \
                else ("client-external.com" if ext else "financeuk-internal.co.uk")
            external[i] = int(ext)
            action[i] = "send"
            sens[i] = "High" if rng.random() < p["high_sens_p"] * 0.7 else "Medium"
        else:  # usb
            resource[i] = "usb_device"
            action[i] = rng.choice(["connect", "copy", "disconnect"], p=[0.4, 0.25, 0.35])
            sens[i] = "Medium"

    vol = np.round(rng.lognormal(p["vol_mu"], p["vol_sig"], n), 3)
    vol[kind == "logon"] = np.round(rng.uniform(50, 150, (kind == "logon").sum()), 3)
    return _frame(emp, ts, kind, resource, sens, action, vol, external,
                  suspicious=np.zeros(n, dtype=int))


def _activity_mix(p):
    usb_share = min(0.06, p["usb_day"] / max(4, p["events_day"]))
    rest = 1 - usb_share
    return [0.14 * rest, 0.38 * rest, 0.30 * rest, 0.18 * rest, usb_share]


def gen_malicious_events(emp, p, n_normal):
    """Small-chunk exfiltration hidden mostly inside office hours."""
    n = max(8, int(n_normal * MALICIOUS_EVENT_FRAC))
    ts = sample_hours(n, MALICIOUS_EVENING_P, 0.10)
    kind = rng.choice(["file_access", "email", "usb", "web"], n,
                      p=[0.45, 0.25, 0.18, 0.12])
    resource = np.empty(n, dtype=object)
    action = np.empty(n, dtype=object)
    sens = np.empty(n, dtype=object)
    external = np.zeros(n, dtype=int)
    for i, k in enumerate(kind):
        if k == "file_access":
            resource[i] = rng.choice(FILES_RARE if rng.random() < 0.55
                                     else FILES_SENSITIVE)
            action[i] = "copy" if rng.random() < 0.7 else "read"
            sens[i] = "High" if rng.random() < 0.75 else "Medium"
        elif k == "email":
            resource[i] = rng.choice(WEBSITES_RISKY[:3])
            action[i], sens[i], external[i] = "send", "High", 1
        elif k == "usb":
            resource[i] = "usb_device"
            action[i] = "copy"
            sens[i] = "High" if rng.random() < 0.5 else "Medium"
        else:
            resource[i] = rng.choice(WEBSITES_RISKY)
            action[i], sens[i], external[i] = "visit", "Medium", 1
    vol = np.round(rng.lognormal(p["vol_mu"], 0.5, n)
                   * rng.uniform(*MALICIOUS_VOL_MULT, n), 3)
    return _frame(emp, ts, kind, resource, sens, action, vol, external,
                  suspicious=np.ones(n, dtype=int))


def _frame(emp, ts, kind, resource, sens, action, vol, external, suspicious):
    hrs = ts.hour
    return pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "employee_id": emp["employee_id"],
        "department": emp["department"],
        "activity_type": kind,
        "resource": resource,
        "sensitivity": sens,
        "action": action,
        "is_after_hours": ((hrs < 8) | (hrs >= 18)).astype(int),
        "is_weekend": (ts.dayofweek >= 5).astype(int),
        "data_volume_kb": vol,
        "is_external": external,
        "is_insider": emp["is_insider"],
        "is_suspicious": suspicious,
    })


def main():
    emps = make_employees()
    logs = []
    for _, emp in emps.iterrows():
        p = persona(emp["department"], emp["seniority"], emp["is_insider"])
        normal = gen_normal_events(emp, p)
        parts = [normal]
        if emp["is_insider"]:
            parts.append(gen_malicious_events(emp, p, len(normal)))
        logs.append(pd.concat(parts, ignore_index=True))
    log = pd.concat(logs, ignore_index=True)
    log = log.sort_values("timestamp").reset_index(drop=True)
    log["log_id"] = np.arange(1, len(log) + 1)

    conn = sqlite3.connect(OUT_DB)
    emps.to_sql("employees", conn, if_exists="replace", index=False)
    log.to_sql("unified_log", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Written {OUT_DB}: {len(log):,} log rows, "
          f"{len(emps)} employees ({emps['is_insider'].sum()} insiders), "
          f"{int(log['is_suspicious'].sum()):,} malicious events "
          f"({log['is_suspicious'].mean():.2%} of log)")


if __name__ == "__main__":
    main()
