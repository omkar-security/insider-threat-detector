"""
K-Anonymity Anonymisation Module for Insider Threat Audit Logs
================================================================
Phase 3 of: Privacy-Preserving Audit Log Analysis for Insider Threat
Detection in Financial Institutions (CSC8209).

Generalises the four quasi-identifier (QID) attributes defined in the
interim report:

    1. User identity   -> department / seniority tier (direct ID removed,
                          replaced with a salted-hash pseudonym so that
                          per-entity behavioural aggregation remains
                          possible within the analysis window)
    2. Timestamp       -> time-of-day window (hour -> 6h window -> day/night -> any)
    3. Resource        -> sensitivity classification (resource -> sensitivity -> any)
    4. Action type     -> behavioural category (action -> category -> any)

Provides:
    * Generalisation hierarchies (global recoding, per-attribute levels)
    * anonymise()      - generalise + suppress until target k is satisfied
    * verify_k()       - programmatic proof of k-anonymity (min class size)
    * ncp()            - Normalised Certainty Penalty information-loss metric
                         (Wagner & Eckhoff, 2018)
"""

import hashlib
import pandas as pd
import numpy as np

# ------------------------------------------------------------------ #
#  GENERALISATION HIERARCHIES
# ------------------------------------------------------------------ #

# --- Identity: department -> division -> any ---------------------- #
DEPT_TO_DIVISION = {
    "Investment Banking": "Front Office",
    "Retail Banking":     "Front Office",
    "Finance":            "Front Office",
    "Risk Management":    "Control Functions",
    "Compliance":         "Control Functions",
    "IT":                 "Support Functions",
    "HR":                 "Support Functions",
    "Operations":         "Support Functions",
}
N_DEPTS = len(DEPT_TO_DIVISION)
DIVISION_SIZES = pd.Series(DEPT_TO_DIVISION).value_counts().to_dict()

SENIORITY_LEVELS = ["Junior", "Mid-level", "Senior"]

# --- Action: action -> behavioural category -> any ---------------- #
ACTION_TO_CATEGORY = {
    "logon":      "Session Activity",
    "connect":    "Session Activity",
    "disconnect": "Session Activity",
    "read":       "Information Access",
    "open":       "Information Access",
    "visit":      "Information Access",
    "send":       "Data Transfer",
    "copy":       "Data Transfer",
    "print":      "Data Transfer",
}
N_ACTIONS = len(ACTION_TO_CATEGORY)
CATEGORY_SIZES = pd.Series(ACTION_TO_CATEGORY).value_counts().to_dict()

# --- Timestamp: hour -> 6h window -> day/night -> any -------------- #
def _hour_to_window(h):
    if 0 <= h < 6:   return "Night (00-06)"
    if 6 <= h < 12:  return "Morning (06-12)"
    if 12 <= h < 18: return "Afternoon (12-18)"
    return "Evening (18-24)"

def _hour_to_daynight(h):
    return "Core Hours (06-18)" if 6 <= h < 18 else "Off Hours (18-06)"

# Leaf counts covered by each timestamp generalisation level (of 24 hours)
TIME_LEVEL_COVER = {0: 1, 1: 6, 2: 12, 3: 24}

QID_LEVELS = {          # max generalisation level per attribute
    "identity":  2,     # 0=dept+seniority, 1=division+seniority, 2=any role
    "time":      3,     # 0=hour, 1=6h window, 2=day/night, 3=any
    "resource":  2,     # 0=exact, 1=sensitivity class, 2=any
    "action":    2,     # 0=exact, 1=behavioural category, 2=any
}


def prepare_log(log: pd.DataFrame, employees: pd.DataFrame) -> pd.DataFrame:
    """Attach the seniority tier (part of the identity QID) to the log."""
    if "seniority" in log.columns:
        return log
    return log.merge(employees[["employee_id", "seniority"]],
                     on="employee_id", how="left")


def pseudonymise(employee_id: str, salt: str = "csc8209") -> str:
    """One-way salted hash pseudonym. Direct identity is removed but
    per-entity linkage is preserved for behavioural aggregation."""
    return "U-" + hashlib.sha256(f"{salt}:{employee_id}".encode()).hexdigest()[:10]


# ------------------------------------------------------------------ #
#  APPLY A GENERALISATION CONFIGURATION (GLOBAL RECODING)
# ------------------------------------------------------------------ #

def generalise(df: pd.DataFrame, levels: dict, n_resources: int = None) -> pd.DataFrame:
    """Return anonymised copy of the log at the given generalisation levels.

    Expects columns: employee_id, department, timestamp, resource,
    sensitivity, action (others pass through untouched).
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    hours = out["timestamp"].dt.hour

    # 1. Identity ---------------------------------------------------
    out["pseudonym"] = out["employee_id"].map(pseudonymise)
    li = levels["identity"]
    if li == 0:
        out["role_group"] = out["department"]
    elif li == 1:
        out["role_group"] = out["department"].map(DEPT_TO_DIVISION)
    else:
        out["role_group"] = "Any Role"
    out = out.drop(columns=["employee_id", "department"])

    # 2. Timestamp --------------------------------------------------
    lt = levels["time"]
    if lt == 0:
        out["time_window"] = hours.astype(str).str.zfill(2) + ":00"
    elif lt == 1:
        out["time_window"] = hours.map(_hour_to_window)
    elif lt == 2:
        out["time_window"] = hours.map(_hour_to_daynight)
    else:
        out["time_window"] = "Any Time"
    # exact timestamp is a direct re-identification vector -> drop it,
    # keep only the (coarse) date for temporal ordering of analysis
    out["date"] = out["timestamp"].dt.date.astype(str)
    out = out.drop(columns=["timestamp"])

    # 3. Resource ---------------------------------------------------
    lr = levels["resource"]
    if lr == 0:
        out["resource_class"] = out["resource"]
    elif lr == 1:
        out["resource_class"] = out["sensitivity"] + " Sensitivity Resource"
    else:
        out["resource_class"] = "Any Resource"
    out = out.drop(columns=["resource"])

    # 4. Action -----------------------------------------------------
    la = levels["action"]
    if la == 0:
        out["action_class"] = out["action"]
    elif la == 1:
        out["action_class"] = out["action"].map(ACTION_TO_CATEGORY)
    else:
        out["action_class"] = "Any Action"
    out = out.drop(columns=["action"])

    return out


QID_COLS = ["role_group", "seniority", "time_window", "resource_class", "action_class"]


# ------------------------------------------------------------------ #
#  K-ANONYMITY VERIFICATION
# ------------------------------------------------------------------ #

def verify_k(df: pd.DataFrame, qid_cols=None) -> dict:
    """Programmatic k-anonymity check: size of the smallest equivalence
    class over the QID columns. Returns achieved k and class statistics."""
    qid_cols = qid_cols or [c for c in QID_COLS if c in df.columns]
    sizes = df.groupby(qid_cols, dropna=False).size()
    return {
        "achieved_k":       int(sizes.min()) if len(sizes) else 0,
        "n_classes":        int(len(sizes)),
        "mean_class_size":  float(sizes.mean()) if len(sizes) else 0.0,
        "median_class_size": float(sizes.median()) if len(sizes) else 0.0,
        "class_sizes":      sizes,
    }


# ------------------------------------------------------------------ #
#  INFORMATION LOSS — NORMALISED CERTAINTY PENALTY
# ------------------------------------------------------------------ #

def ncp(levels: dict, n_resources: int) -> float:
    """NCP for a global-recoding configuration: for a categorical
    attribute generalised to a node covering m of M leaves,
    NCP = (m-1)/(M-1); averaged over the four QID attributes.
    (Xu et al. 2006 utility-based anonymisation; framework per
    Wagner & Eckhoff 2018.)"""
    def cat_ncp(m, M):
        return 0.0 if M <= 1 else (m - 1) / (M - 1)

    # identity (department dimension; seniority never generalised here)
    li = levels["identity"]
    if li == 0:
        id_ncp = 0.0
    elif li == 1:
        avg_div = np.mean(list(DIVISION_SIZES.values()))
        id_ncp = cat_ncp(avg_div, N_DEPTS)
    else:
        id_ncp = 1.0

    t_ncp = cat_ncp(TIME_LEVEL_COVER[levels["time"]], 24)

    lr = levels["resource"]
    if lr == 0:
        r_ncp = 0.0
    elif lr == 1:
        r_ncp = cat_ncp(n_resources / 3, n_resources)   # ~1/3 of leaves per class
    else:
        r_ncp = 1.0

    la = levels["action"]
    if la == 0:
        a_ncp = 0.0
    elif la == 1:
        avg_cat = np.mean(list(CATEGORY_SIZES.values()))
        a_ncp = cat_ncp(avg_cat, N_ACTIONS)
    else:
        a_ncp = 1.0

    return float(np.mean([id_ncp, t_ncp, r_ncp, a_ncp]))


# ------------------------------------------------------------------ #
#  MAIN ANONYMISATION ROUTINE
# ------------------------------------------------------------------ #

def _level_lattice():
    """All generalisation configurations, ordered by total level
    (least generalised first) so the minimum-loss solution is found."""
    combos = []
    for i in range(QID_LEVELS["identity"] + 1):
        for t in range(QID_LEVELS["time"] + 1):
            for r in range(QID_LEVELS["resource"] + 1):
                for a in range(QID_LEVELS["action"] + 1):
                    combos.append({"identity": i, "time": t, "resource": r, "action": a})
    combos.sort(key=lambda d: (sum(d.values()), d["resource"], d["time"]))
    return combos


def anonymise(df: pd.DataFrame, k: int, max_suppression: float = 0.05,
              start_levels: dict = None, verbose: bool = False) -> dict:
    """Anonymise `df` to satisfy k-anonymity with the lowest information
    loss found on the generalisation lattice.

    Strategy: global recoding over the level lattice; residual
    equivalence classes smaller than k are suppressed provided the
    suppression rate stays within `max_suppression`.

    Returns dict with: data, levels, verification, ncp,
    suppression_rate, n_suppressed.
    """
    n_resources = df["resource"].nunique()
    n_total = len(df)

    for levels in _level_lattice():
        if start_levels and any(levels[a] < start_levels[a] for a in levels):
            continue
        gen = generalise(df, levels, n_resources)
        sizes = gen.groupby(QID_COLS, dropna=False)["pseudonym"].transform("size")
        keep = sizes >= k
        supp_rate = 1 - keep.mean()
        if verbose:
            print(f"levels={levels}  suppression={supp_rate:.2%}")
        if supp_rate <= max_suppression:
            anon = gen[keep].reset_index(drop=True)
            check = verify_k(anon)
            assert check["achieved_k"] >= k, "k-anonymity verification failed"
            return {
                "data":             anon,
                "levels":           levels,
                "verification":     check,
                "ncp":              ncp(levels, n_resources),
                "suppression_rate": float(supp_rate),
                "n_suppressed":     int(n_total - len(anon)),
                "k_target":         k,
            }

    raise ValueError(f"No configuration satisfies k={k} within "
                     f"{max_suppression:.0%} suppression.")


# ------------------------------------------------------------------ #
#  FEATURE ENGINEERING ON ANONYMISED LOGS (for detection re-run)
# ------------------------------------------------------------------ #

def features_from_anonymised(anon: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the behavioural feature vector using ONLY anonymised
    columns, aggregated per pseudonym."""
    a = anon.copy()
    a["is_high_sens"] = (a["sensitivity"] == "High").astype(int)
    a["is_transfer"]  = (a["action_class"].isin(
        ["Data Transfer", "send", "copy", "print"])).astype(int)
    a["is_copy_like"] = (a["action_class"].isin(["copy", "Data Transfer"])).astype(int)
    a["is_usb"]       = (a["activity_type"] == "usb").astype(int)

    feats = a.groupby("pseudonym").agg(
        total_events           = ("log_id",        "count"),
        after_hours_ratio      = ("is_after_hours", "mean"),
        external_ratio         = ("is_external",    "mean"),
        avg_data_volume_kb     = ("data_volume_kb", "mean"),
        max_data_volume_kb     = ("data_volume_kb", "max"),
        total_data_volume_kb   = ("data_volume_kb", "sum"),
        high_sensitivity_ratio = ("is_high_sens",   "mean"),
        usb_events             = ("is_usb",         "sum"),
        transfer_ratio         = ("is_transfer",    "mean"),
    ).reset_index()
    feats["usb_per_day"] = feats["usb_events"] / 180
    return feats


ANON_ML_COLS = [
    "after_hours_ratio", "external_ratio", "avg_data_volume_kb",
    "max_data_volume_kb", "high_sensitivity_ratio",
    "usb_per_day", "transfer_ratio", "total_data_volume_kb",
]
