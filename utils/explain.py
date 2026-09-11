"""
explain.py — Rule-based explanation templates using SHAP
No live LLM — no API latency, no failure risk on demo day.
"""
import numpy as np
import pandas as pd
 
# This order MUST exactly match the X_agg column order built in train.py's
# train_all(). The XGBoost model + SHAP explainer were fit on THIS 8-feature
# vector — not on FEATURE_COLS from features.py, which is a different,
# 10-feature list used only for the anomaly detectors' input, not for XGBoost.
SHAP_FEATURE_COLS = [
    "iso_score",
    "lof_score",
    "is_round_amount",
    "repeated_amount_flag",
    "ida_rejected",
    "bulk_recommendation_flag",
    "missing_location",
    "mp_concentration_zscore",
]
 
FEATURE_DESCRIPTIONS = {
    "iso_score":                 "flagged as an anomaly by Isolation Forest",
    "lof_score":                 "flagged as a local density outlier",
    "is_round_amount":           "round-number allocation amount (fabrication indicator)",
    "repeated_amount_flag":      "same exact amount recommended by this MP 3+ times",
    "ida_rejected":              "work rejected by IDA (Implementation District Authority)",
    "bulk_recommendation_flag":  "MP bulk-recommended 20+ works in a single year",
    "missing_location":          "work location details missing — cannot be physically verified",
    "mp_concentration_zscore":   "MP work volume far above national average",
}
 
TEMPLATES = {
    1: "Flagged primarily due to {f1}.",
    2: "Flagged primarily due to {f1}, compounded by {f2}.",
    3: "Flagged primarily due to {f1}, compounded by {f2} and {f3}.",
}
 
 
def explain_record(row: pd.Series, explainer, feature_cols: list = None) -> str:
    """
    feature_cols is accepted for backward compatibility with existing callers
    (e.g. app.py passes FEATURE_COLS) but is ignored — the SHAP explainer was
    fit on a fixed feature set (SHAP_FEATURE_COLS) that must not vary per call.
    """
    try:
        X_row = row[SHAP_FEATURE_COLS].fillna(0).values.reshape(1, -1).astype(float)
        shap_vals = explainer.shap_values(X_row)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = shap_vals[0]
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
        top_features = [SHAP_FEATURE_COLS[i] for i in top_idx if shap_vals[i] > 0]
        if not top_features:
            return "Multiple weak signals detected — borderline case, manual review recommended."
        desc = [FEATURE_DESCRIPTIONS.get(f, f.replace("_", " ")) for f in top_features]
        n = min(len(desc), 3)
        return TEMPLATES[n].format(
            f1=desc[0] if len(desc) > 0 else "",
            f2=desc[1] if len(desc) > 1 else "",
            f3=desc[2] if len(desc) > 2 else "",
        )
    except Exception:
        return "Explanation unavailable for this record."