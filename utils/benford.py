"""
benford.py — Benford's Law on ALLOCATION AMOUNT
Real forensic accounting technique. Zero labels needed.
"""
import numpy as np
import pandas as pd

BENFORD_EXPECTED = {d: np.log10(1 + 1/d) for d in range(1, 10)}


def get_leading_digit(x):
    if x <= 0:
        return None
    for ch in str(int(x)):
        if ch != "0":
            return int(ch)
    return None


def benford_analysis(df: pd.DataFrame, amount_col: str = "ALLOCATION AMOUNT") -> dict:
    amounts = df[amount_col].dropna()
    amounts = amounts[amounts > 0]
    digits = amounts.apply(get_leading_digit).dropna().astype(int)
    observed = digits.value_counts().reindex(range(1, 10), fill_value=0)
    total = observed.sum()
    obs_freq = observed / total
    exp_freq = pd.Series(BENFORD_EXPECTED)
    exp_counts = exp_freq * total
    chi2 = float(((observed - exp_counts) ** 2 / exp_counts).sum())
    mad = float((obs_freq - exp_freq).abs().mean())

    if mad < 0.006:
        conformity = "Close conformity — Low risk"
    elif mad < 0.012:
        conformity = "Acceptable — Monitor"
    elif mad < 0.015:
        conformity = "Marginal — Investigate"
    else:
        conformity = "Non-conforming — High fraud risk"

    return {
        "observed_freq": obs_freq,
        "expected_freq": exp_freq,
        "chi2": chi2,
        "mad": mad,
        "conformity": conformity,
        "total": int(total),
    }


def benford_by_mp(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mp, gdf in df.groupby("MP NAME"):
        if len(gdf) < 8:
            continue
        r = benford_analysis(gdf)
        rows.append({
            "MP NAME": mp,
            "STATE": gdf["STATE"].iloc[0],
            "works_count": len(gdf),
            "chi2": round(r["chi2"], 2),
            "mad": round(r["mad"], 4),
            "conformity": r["conformity"],
            "benford_risk": "High" if r["mad"] >= 0.015 else
                            "Medium" if r["mad"] >= 0.012 else "Low"
        })
    return pd.DataFrame(rows).sort_values("mad", ascending=False)