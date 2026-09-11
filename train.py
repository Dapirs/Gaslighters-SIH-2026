import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
 
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from utils.features import engineer_features, FEATURE_COLS
from config import DATA_URL, SCORED_DATA_PATH, RISK_BAND_MEDIUM, RISK_BAND_HIGH, RANDOM_SEED
 
 
def normalize(scores):
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn + 1e-9)
 
 
def inject_weak_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Generate weak supervision labels from known fraud patterns.
    These are NOT ground truth — they are stand-ins for auditor-confirmed
    labels in the active learning loop we'd build in production.
    """
    labels = np.zeros(len(df))
    # Pattern 1: Round amount + unsanctioned + action pending
    labels[(df["is_round_amount"] == 1) &
           (df["is_unsanctioned"] == 1) &
           (df["action_pending"] == 1)] = 1
    # Pattern 2: Repeated exact amount by same MP (3+ times)
    labels[df["repeated_amount_flag"] == 1] = 1
    # Pattern 3: IDA rejected
    labels[df["ida_rejected"] == 1] = 1
    # Pattern 4: Bulk recommendation + missing location
    labels[(df["bulk_recommendation_flag"] == 1) &
           (df["missing_location"] == 1)] = 1
    return labels
 
 
def train_all(df: pd.DataFrame):
    print(f"Loaded {len(df)} real MPLADS records")
    print("Engineering features...")
    df = engineer_features(df)
 
    X = df[FEATURE_COLS].fillna(0)
 
    os.makedirs("models", exist_ok=True)
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    # ── 1. Isolation Forest ─────────────────────────────────────
    print("Training Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=0.10,
                          random_state=RANDOM_SEED, n_jobs=-1)
    iso.fit(X_scaled)
    iso_score = normalize(1 - iso.decision_function(X_scaled))
    df["iso_score"] = iso_score
    print(f"  IF flagged: {(iso.predict(X_scaled) == -1).sum()} anomalies")
 
    # ── 2. Local Outlier Factor ──────────────────────────────────
    print("Training Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.10, n_jobs=-1)
    lof.fit_predict(X_scaled)
    lof_score = normalize(-lof.negative_outlier_factor_)
    df["lof_score"] = lof_score
 
    # ── 3. Weak supervision labels ───────────────────────────────
    print("Generating weak supervision labels...")
    y_weak = inject_weak_labels(df)
    print(f"  Weak labels: {int(y_weak.sum())} suspicious records "
          f"({y_weak.mean()*100:.1f}%) — stand-in for auditor feedback")
 
    # ── 4. XGBoost calibration layer ────────────────────────────
    print("Training XGBoost calibration layer...")
    X_agg = np.column_stack([
        iso_score, lof_score,
        df["is_round_amount"].values,
        df["repeated_amount_flag"].values,
        df["ida_rejected"].values,
        df["bulk_recommendation_flag"].values,
        df["missing_location"].values,
        df["mp_concentration_zscore"].values,
    ])
    scale_pos = (y_weak == 0).sum() / max((y_weak == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        scale_pos_weight=scale_pos, eval_metric="auc",
        random_state=RANDOM_SEED, n_jobs=-1
    )
    xgb_model.fit(X_agg, y_weak, verbose=False)
    xgb_prob = xgb_model.predict_proba(X_agg)[:, 1]
    df["xgb_score"] = xgb_prob
 
    # ── 5. Final combined score ──────────────────────────────────
    df["combined_score"] = (
        0.30 * iso_score +
        0.25 * lof_score +
        0.45 * xgb_prob
    )
    df["risk_label"] = pd.cut(
        df["combined_score"],
        bins=[0, RISK_BAND_MEDIUM, RISK_BAND_HIGH, 1.0],
        labels=["Low", "Medium", "High"]
    )
 
    # ── 6. SHAP ─────────────────────────────────────────────────
    print("Building SHAP explainer...")
    explainer = shap.TreeExplainer(xgb_model)
 
    # Save
    with open("models/isolation_forest.pkl", "wb") as f: pickle.dump(iso, f)
    with open("models/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open("models/xgboost_model.pkl", "wb") as f: pickle.dump(xgb_model, f)
    with open("models/shap_explainer.pkl", "wb") as f: pickle.dump(explainer, f)
 
    df.to_csv(SCORED_DATA_PATH, index=False)
    print(f"\nDone. High risk: {(df['risk_label']=='High').sum()} records")
    print(f"Medium risk: {(df['risk_label']=='Medium').sum()} records")
    return df, xgb_model, explainer
 
 
if __name__ == "__main__":
    raw = pd.read_csv(DATA_URL, sep=";", low_memory=False, encoding="utf-8")
    train_all(raw)