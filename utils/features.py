"""
features.py — Feature engineering on real MPLADS government data
Columns available: MP NAME, WORK, CATEGORY, STATE, CONSTITUENCY,
                   IDA, CITY, WARD, BLOCK, VILLAGE, RECOMMENDED DATE,
                   ALLOCATION AMOUNT, IDA APPROVAL, STATUS, HOUSE
"""
import pandas as pd
import numpy as np
 
 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # ── 1. Round-number allocation flag ─────────────────────────
    # 487000 repeating, 100000, 500000 — classic fabrication signal
    def is_round(x):
        for divisor in [100000, 50000, 10000, 5000]:
            if x % divisor == 0:
                return 1
        return 0
    df["is_round_amount"] = df["ALLOCATION AMOUNT"].apply(is_round)
 
    # ── 2. Allocation amount z-score per state ───────────────────
    state_stats = df.groupby("STATE")["ALLOCATION AMOUNT"].agg(["mean", "std"]).reset_index()
    state_stats.columns = ["STATE", "state_mean", "state_std"]
    df = df.merge(state_stats, on="STATE", how="left")
    df["amount_zscore"] = (
        (df["ALLOCATION AMOUNT"] - df["state_mean"]) /
        df["state_std"].replace(0, 1)
    ).fillna(0)
    df.drop(columns=["state_mean", "state_std"], inplace=True)
 
    # ── 3. MP work concentration ─────────────────────────────────
    # MPs recommending abnormally many works
    mp_count = df.groupby("MP NAME").size().reset_index(name="mp_work_count")
    df = df.merge(mp_count, on="MP NAME", how="left")
    mp_mean = df["mp_work_count"].mean()
    mp_std  = df["mp_work_count"].std()
    df["mp_concentration_zscore"] = (
        (df["mp_work_count"] - mp_mean) / (mp_std if mp_std else 1)
    )
 
    # ── 4. Repeated exact amount flag ───────────────────────────
    # Same MP recommending same amount repeatedly (like 487000 x7)
    mp_amount_counts = df.groupby(["MP NAME", "ALLOCATION AMOUNT"]).size().reset_index(name="repeat_count")
    df = df.merge(mp_amount_counts, on=["MP NAME", "ALLOCATION AMOUNT"], how="left")
    df["repeated_amount_flag"] = (df["repeat_count"] >= 3).astype(int)
 
    # ── 5. IDA rejection flag ────────────────────────────────────
    # Case/whitespace-insensitive: raw government data is inconsistently cased
    # ("Rejected by IDA" vs "REJECTED BY IDA" vs " Rejected by IDA ") and an exact
    # string match would silently flag nothing if the casing doesn't line up.
    ida_norm = df["IDA APPROVAL"].astype(str).str.strip().str.lower()
    status_norm = df["STATUS"].astype(str).str.strip().str.lower()

    df["ida_rejected"] = ida_norm.str.contains("rejected", na=False).astype(int)

    # ── 6. Unsanctioned flag ────────────────────────────────────
    df["is_unsanctioned"] = status_norm.str.contains("unsanctioned", na=False).astype(int)

    # ── 7. Action pending flag ───────────────────────────────────
    df["action_pending"] = ida_norm.str.contains("pending", na=False).astype(int)
 
    # ── 8. Year extracted from RECOMMENDED DATE ──────────────────
    df["rec_year"] = pd.to_datetime(df["RECOMMENDED DATE"], errors="coerce").dt.year
    df["rec_year"] = df["rec_year"].fillna(df["rec_year"].median())
 
    # ── 9. Bulk recommendation flag ─────────────────────────────
    # MP recommending many works in same year — suspicious clustering
    mp_year_count = df.groupby(["MP NAME", "rec_year"]).size().reset_index(name="mp_year_count")
    df = df.merge(mp_year_count, on=["MP NAME", "rec_year"], how="left")
    df["bulk_recommendation_flag"] = (df["mp_year_count"] >= 20).astype(int)
 
    # ── 10. Missing location flag ────────────────────────────────
    # Works with no city/ward/village — harder to verify physically
    df["missing_location"] = (
        df["CITY"].isna().astype(int) +
        df["WARD"].isna().astype(int) +
        df["VILLAGE"].isna().astype(int)
    ).clip(0, 1)
 
    return df
 
 
FEATURE_COLS = [
    "is_round_amount",
    "amount_zscore",
    "mp_work_count",
    "mp_concentration_zscore",
    "repeated_amount_flag",
    "ida_rejected",
    "is_unsanctioned",
    "action_pending",
    "bulk_recommendation_flag",
    "missing_location",
]
 