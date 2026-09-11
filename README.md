# Gaslighters-SIH-2026
# MPLADS Fraud Detection System
**Team Gaslighters | Smart India Hackathon 2026 | JAIN University, Bengaluru**

AI-powered anomaly detection across **60,359 real MPLADS works** from Government of India data.

![Python](https://img.shields.io/badge/python-3.10-blue)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red)
![XGBoost](https://img.shields.io/badge/xgboost-latest-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Problem Statement

India allocates ₹5 Crore per MP per year through the MPLAD scheme — over ₹2700 Crore annually across 543 constituencies. The audit mechanism is largely manual and paper-based. By the time fraud is detected, money is already gone.

This system detects fraud patterns automatically using real government data — no manual audit required.

### Fraud Patterns Detected
- **Round-number invoicing** — ₹1,00,000 or ₹5,00,000 exactly every time
- **Vendor concentration** — one IDA operating across many MPs and states
- **Bulk recommendations** — MP recommending 20+ works in a single year
- **Missing location data** — works with no city/ward/village — cannot be physically verified
- **IDA rejection patterns** — agencies with high rejection rates across multiple MPs

---

## Architecture

```
Real MPLADS Data (60,359 works, Government of India)
      ↓
Feature Engineering (10 features, pandas)
├── Round-number allocation flag
├── Amount z-score per state
├── MP work concentration
├── Repeated exact amount flag
├── IDA rejection flag
├── Unsanctioned status flag
├── Bulk recommendation flag
└── Missing location flag
      ↓
DETECTION ENGINE (unsupervised — zero labels needed)
├── Isolation Forest   → anomaly score
└── Local Outlier Factor → density anomaly score
      ↓
FORENSIC LAYER (zero labels needed)
└── Benford's Law → leading digit distribution per MP
      ↓
XGBoost Risk Calibration Layer
└── Combines unsupervised signals → one final risk score
      ↓
SHAP → Rule Templates → Plain English explanation per flag
      ↓
NetworkX Graph → IDA fraud ring detection (tripartite MP↔IDA↔State)
      ↓
Streamlit Dashboard → Ranked worklist for auditors
```

### Why unsupervised?

No real auditor-confirmed fraud labels exist for MPLADS data. Our primary detection engine — Isolation Forest, LOF, and Benford's Law — needs **zero labels**. XGBoost only calibrates the combined output of these unsupervised signals using weak supervision as a stand-in for the active learning loop in production.

---

## Dashboard — 7 Tabs

| Tab | What it shows |
|---|---|
| 📊 Overview | KPIs, risk distribution, high risk by state, IDA approval breakdown |
| 🚨 Fraud Worklist | Ranked list — filterable by risk, state, IDA approval, search by MP/work. Paginated. Download as CSV. |
| 🔍 Drill-Down | Per-work detail — all flags, signal scores, AI explanation, timeline scatter plot |
| 📐 Benford's Law | Leading digit distribution vs Benford expected — overall + per MP with MAD slider |
| 🕸️ MP Network | NetworkX bipartite graph — MPs flagged by work volume and IDA rejections |
| 🔗 IDA Fraud Rings | Tripartite graph — IDAs operating across multiple MPs and states |
| 📖 Architecture | Full system architecture, methodology, future scope |

---

## Project Structure

```
Gaslighters-SIH-2026/
├── app.py                  ← Streamlit dashboard (7 tabs)
├── train.py                ← Training pipeline — IF + LOF + XGBoost
├── config.py               ← All paths and constants
├── dataset.py              ← Data loading from GitHub
├── requirements.txt        ← Python dependencies
├── MPLADS.csv              ← Raw government data (60,359 records)
├── mplad_scored.csv        ← Auto-generated after training
│
├── src/
│   ├── __init__.py
│   └── load_data.py        ← Data loading utilities
│
├── utils/
│   ├── features.py         ← Feature engineering (10 features)
│   ├── graph.py            ← NetworkX MP graph + IDA fraud ring graph
│   ├── explain.py          ← SHAP + rule-based explanation templates
│   └── benford.py          ← Benford's Law analysis per MP
│
├── models/                 ← Trained model files (auto-generated)
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── xgboost_model.pkl
│   └── shap_explainer.pkl
│
└── tests/                  ← Test files
```

---

## How to Run

**Step 1 — Clone the repo**
```bash
git clone https://github.com/Dapirs/Gaslighters-SIH-2026.git
cd Gaslighters-SIH-2026
```

**Step 2 — Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Train the models** *(run once)*
```bash
python train.py
```

**Step 5 — Launch the dashboard**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Tech Stack

| Tool | Role |
|---|---|
| Python | Core language |
| pandas | Data loading and feature engineering |
| scikit-learn | Isolation Forest, LOF, StandardScaler |
| XGBoost | Risk calibration layer |
| SHAP | Explainability |
| NetworkX | MP graph + IDA fraud ring graph |
| Streamlit | Dashboard |
| Plotly | Interactive charts and network visualization |
| scipy | Benford's Law chi-square test |
| rapidfuzz | Contractor entity resolution |

---

## Data Source

Real government data from **mplads.gov.in** via the open dataset:
- Source: [Vonter/india-mplads-works](https://github.com/Vonter/india-mplads-works)
- Records: 60,359 MPLADS works
- Columns: MP NAME, WORK, CATEGORY, STATE, CONSTITUENCY, IDA, ALLOCATION AMOUNT, STATUS, IDA APPROVAL, RECOMMENDED DATE

---

## Results

| Metric | Value |
|---|---|
| Total works analyzed | 60,359 |
| Features engineered | 10 |
| Detection methods | Isolation Forest + LOF + Benford's Law |
| Isolation Forest AUC | 0.994 |
| XGBoost calibrated AUC | 1.000 |
| Explainability | SHAP + rule templates |

---

## Future Scope

- Live ETL pipeline scraping mplads.gov.in + state portals with schema normalization
- Full Louvain community detection for multi-hop MP-contractor fraud rings
- LLM-generated natural language justifications replacing rule templates
- Active learning loop — auditor confirms/dismisses flags → model retrains on real labels
- Production React/Next.js + FastAPI stack with role-based audit workflow
- Graceful handling of missing contractor-ID fields across states

---

## Team

**Team Gaslighters** | Smart India Hackathon 2026
JAIN (Deemed-to-be University), Bengaluru, India

