"""
app.py — MPLAD Fraud Detection Dashboard
Real government data: 60,359 MPLADS works records
"""
import os, sys, pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.features import engineer_features, FEATURE_COLS
from utils.graph import (
    build_mp_graph, get_flagged_mps, graph_summary,
    build_ida_graph, get_flagged_idas, ida_graph_summary,
)
from utils.explain import explain_record
from utils.benford import benford_analysis, benford_by_mp
from config import (
    DATA_URL, SCORED_DATA_PATH, RANDOM_SEED,
    MIN_IDA_MPS, MIN_IDA_STATES, MIN_IDA_REJECTION_RATE,
)
 
st.set_page_config(page_title="MPLAD Fraud Detection", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")
 
# Session state defaults
st.session_state.setdefault("jump_to_mp", None)
 
@st.cache_data
def load_data():
    if os.path.exists(SCORED_DATA_PATH):
        df = pd.read_csv(SCORED_DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL, sep=";", low_memory=False, encoding="utf-8")
        df = engineer_features(df)
    return df
 
@st.cache_resource
def load_models():
    with open("models/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return xgb_model, explainer
 
@st.cache_resource
def load_graph(_df):
    return build_mp_graph(_df)
 
@st.cache_resource
def load_ida_graph(_df):
    return build_ida_graph(_df)
 
@st.cache_data
def filter_worklist(_df, rf, sf, af, search):
    """Cached filtering for the Fraud Worklist page."""
    filt = _df
    if rf != "All": filt = filt[filt["risk_label"] == rf]
    if sf != "All": filt = filt[filt["STATE"] == sf]
    if af != "All": filt = filt[filt["IDA APPROVAL"] == af]
    if search:
        mask = (filt["MP NAME"].str.contains(search, case=False, na=False) |
                filt["WORK"].str.contains(search, case=False, na=False))
        filt = filt[mask]
    return filt.sort_values("combined_score", ascending=False)
 
# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=80)
st.sidebar.title("MPLAD Fraud Detection")
st.sidebar.markdown("**Smart India Hackathon 2026**")
st.sidebar.markdown("**Team Gaslighters | JAIN University**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🚨 Fraud Worklist",
    "🔍 Drill-Down",
    "📐 Benford's Law",
    "🕸️ MP Network",
    "🔗 IDA Fraud Rings",
    "📖 Architecture",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Data:** 60,359 real MPLADS works")
st.sidebar.markdown("**Detection:** IF + LOF + XGBoost")
st.sidebar.markdown("**Forensic:** Benford's Law")
 
try:
    df = load_data()
    xgb_model, explainer = load_models()
    G = load_graph(df)
    G_ida = load_ida_graph(df)
except FileNotFoundError:
    st.error("Run `python models/train.py` first to train the models.")
    st.stop()
 
# Data freshness / source indicator
data_source = f"📁 Pre-scored local dataset ({SCORED_DATA_PATH})" if os.path.exists(SCORED_DATA_PATH) else "🌐 Live fetch from GitHub (Vonter/india-mplads-works)"
st.sidebar.markdown("---")
st.sidebar.caption(f"**Source:** {data_source}")
st.sidebar.caption(f"**Session loaded:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
 
# ══════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🔍 MPLAD Fund Utilization — Fraud Detection")
    st.markdown("AI-powered anomaly detection across **60,359 real MPLADS works** from Government of India data.")
 
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Works", f"{len(df):,}")
    c2.metric("High Risk", f"{(df['risk_label']=='High').sum():,}",
              delta=f"{(df['risk_label']=='High').mean()*100:.1f}%", delta_color="inverse")
    c3.metric("Medium Risk", f"{(df['risk_label']=='Medium').sum():,}")
    c4.metric("Total Funds", f"₹{df['ALLOCATION AMOUNT'].sum()/1e7:.0f} Cr")
    c5.metric("At-Risk Funds",
              f"₹{df[df['risk_label']=='High']['ALLOCATION AMOUNT'].sum()/1e7:.0f} Cr",
              delta_color="inverse")
 
    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.subheader("Risk Distribution")
        rc = df["risk_label"].value_counts().reset_index()
        rc.columns = ["Risk", "Count"]
        fig = px.bar(rc, x="Risk", y="Count", color="Risk",
                     color_discrete_map={"High":"#E74C3C","Medium":"#F39C12","Low":"#2ECC71"},
                     text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
 
    with cb:
        st.subheader("Status Breakdown")
        sv = df["STATUS"].fillna("Unknown").value_counts().reset_index()
        sv.columns = ["Status", "Count"]
        fig2 = px.pie(sv, names="Status", values="Count",
                      color_discrete_sequence=["#E74C3C","#F39C12","#2ECC71","#3498DB"])
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
 
    st.markdown("---")
    ca2, cb2 = st.columns(2)
    with ca2:
        st.subheader("High Risk Works by State")
        sr = df[df["risk_label"]=="High"].groupby("STATE").size().reset_index(name="count")
        fig3 = px.bar(sr.sort_values("count").tail(15),
                      x="count", y="STATE", orientation="h",
                      color="count", color_continuous_scale="Reds")
        fig3.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
 
    with cb2:
        st.subheader("IDA Approval Status")
        ia = df["IDA APPROVAL"].value_counts().reset_index()
        ia.columns = ["IDA Status", "Count"]
        fig4 = px.bar(ia, x="IDA Status", y="Count", color="IDA Status",
                      color_discrete_sequence=["#E74C3C","#F39C12","#2ECC71"])
        fig4.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
 
# ══════════════════════════════════════════════════
# FRAUD WORKLIST
# ══════════════════════════════════════════════════
elif page == "🚨 Fraud Worklist":
    st.title("🚨 Fraud Worklist — Ranked by Risk Score")
 
    c1, c2, c3, c4 = st.columns(4)
    rf = c1.selectbox("Risk Level", ["All","High","Medium","Low"])
    sf = c2.selectbox("State", ["All"] + sorted(df["STATE"].dropna().unique().tolist()))
    af = c3.selectbox("IDA Approval", ["All"] + sorted(df["IDA APPROVAL"].dropna().unique().tolist()))
    search = c4.text_input("Search MP / Work", placeholder="e.g. Sharma or road construction")
 
    filt = filter_worklist(df, rf, sf, af, search)
 
    # Jump-to-MP handoff from the Benford's Law page
    if st.session_state.get("jump_to_mp"):
        jm = st.session_state["jump_to_mp"]
        st.info(f"Filtered to MP flagged from Benford's Law page: **{jm}**")
        filt = filt[filt["MP NAME"] == jm]
        if st.button("✖️ Clear MP filter"):
            st.session_state["jump_to_mp"] = None
            st.rerun()
 
    disp_full = filt[["MP NAME","STATE","WORK","ALLOCATION AMOUNT",
                       "STATUS","IDA APPROVAL","combined_score","risk_label"]].copy()
 
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "⬇️ Download filtered worklist (CSV)",
            data=disp_full.to_csv(index=False).encode("utf-8"),
            file_name="mplad_flagged_worklist.csv",
            mime="text/csv",
            use_container_width=True,
        )
 
    # Pagination
    page_size = 50
    total_rows = len(disp_full)
    total_pages = max(1, (total_rows - 1) // page_size + 1)
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page_num - 1) * page_size
    end = start + page_size
 
    disp = disp_full.iloc[start:end].copy()
    if not disp.empty:
        disp["combined_score"] = disp["combined_score"].round(3)
        disp["ALLOCATION AMOUNT"] = disp["ALLOCATION AMOUNT"].apply(lambda x: f"₹{x/1e5:.2f}L")
        disp["WORK"] = disp["WORK"].str[:60] + "..."
 
    def highlight(row):
        if row["risk_label"] == "High":
            return ['background-color: #FDECEA; color: #000000'] * len(row)
        elif row["risk_label"] == "Medium":
            return ['background-color: #FEF9E7; color: #000000'] * len(row)
        else:
            return [''] * len(row)
 
    if disp.empty:
        st.warning("No records match the current filters.")
    else:
        st.dataframe(disp.style.apply(highlight, axis=1),
                     use_container_width=True, height=500)
    st.caption(f"Showing rows {start+1 if total_rows else 0}–{min(end, total_rows)} of {total_rows} records "
               f"(page {page_num} of {total_pages})")
 
# ══════════════════════════════════════════════════
# DRILL DOWN
# ══════════════════════════════════════════════════
elif page == "🔍 Drill-Down":
    st.title("🔍 Work Item Drill-Down")
 
    high = df[df["risk_label"]=="High"]
    if high.empty:
        st.info("No high-risk works found in the current dataset.")
        st.stop()
 
    mp_sel = st.selectbox("Select MP", sorted(high["MP NAME"].unique().tolist()))
    mp_works = high[high["MP NAME"] == mp_sel].sort_values("combined_score", ascending=False)
 
    if mp_works.empty:
        st.warning(f"No high-risk works found for {mp_sel}.")
        st.stop()
 
    work_sel = st.selectbox("Select Work", mp_works["WORK"].str[:80].tolist())
    row = mp_works[mp_works["WORK"].str[:80] == work_sel].iloc[0]
 
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Risk Score", f"{row['combined_score']:.3f}")
    c2.metric("Risk Label", row["risk_label"])
    c3.metric("Amount", f"₹{row['ALLOCATION AMOUNT']/1e5:.2f}L")
    c4.metric("Status", row["STATUS"] if pd.notna(row["STATUS"]) else "Unknown")
 
    state_amounts = df[df["STATE"] == row["STATE"]]["ALLOCATION AMOUNT"]
    state_avg = state_amounts.mean()
    # Reuse the same amount_zscore computed in features.py (per-state, zero-std safe)
    # rather than recomputing it here, so the UI always matches what the model saw.
    z = row.get("amount_zscore", None)
    if z is not None and pd.notna(z):
        c5.metric("vs State Avg", f"{z:+.1f}σ", delta_color="inverse" if abs(z) > 2 else "off")
    else:
        c5.metric("vs State Avg", "N/A")
 
    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        st.subheader("Work Details")
        details = {
            "MP Name": row["MP NAME"],
            "State": row["STATE"],
            "Constituency": row.get("CONSTITUENCY", "N/A"),
            "Category": row["CATEGORY"],
            "IDA": row["IDA"],
            "IDA Approval": row["IDA APPROVAL"],
            "Recommended Date": row["RECOMMENDED DATE"],
            "MP Total Work Count": row.get("mp_work_count", "N/A"),
            "Round Amount": "Yes" if row["is_round_amount"] else "No",
            "Repeated Amount": "Yes" if row["repeated_amount_flag"] else "No",
            "IDA Rejected (flag)": "Yes" if row.get("ida_rejected") else "No",
            "Action Pending (flag)": "Yes" if row.get("action_pending") else "No",
            "Missing Location": "Yes" if row["missing_location"] else "No",
            "Bulk Recommendation": "Yes" if row["bulk_recommendation_flag"] else "No",
        }
        for k, v in details.items():
            st.markdown(f"**{k}:** {v}")
 
    with cr:
        st.subheader("AI Explanation")
        explanation = explain_record(row, explainer, FEATURE_COLS)
        st.info(f"🤖 {explanation}")
 
        st.subheader("Signal Scores")
        signals = pd.DataFrame({
            "Signal": ["Isolation Forest","LOF","XGBoost Calibrated"],
            "Score": [row.get("iso_score",0), row.get("lof_score",0), row.get("xgb_score",0)]
        })
        fig = px.bar(signals, x="Score", y="Signal", orientation="h",
                     color="Score", color_continuous_scale="Reds", range_x=[0,1])
        fig.update_layout(height=220, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
 
    if "RECOMMENDED DATE" in mp_works.columns:
        st.markdown("---")
        st.subheader(f"Timeline of Flagged Works — {mp_sel}")
        timeline_df = mp_works.copy()
        timeline_df["RECOMMENDED DATE"] = pd.to_datetime(timeline_df["RECOMMENDED DATE"], errors="coerce")
        timeline_df = timeline_df.dropna(subset=["RECOMMENDED DATE"]).sort_values("RECOMMENDED DATE")
        if not timeline_df.empty:
            fig_tl = px.scatter(timeline_df, x="RECOMMENDED DATE", y="combined_score",
                                 size="ALLOCATION AMOUNT", color="combined_score",
                                 color_continuous_scale="Reds",
                                 hover_data=["WORK"],
                                 title="Flagged works over time (bubble size = allocation amount)")
            fig_tl.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.caption("No valid dates available for timeline.")
 
    st.markdown("---")
    st.subheader(f"All High Risk Works by {mp_sel}")
    st.dataframe(
        mp_works[["WORK","ALLOCATION AMOUNT","STATUS","IDA APPROVAL","combined_score"]]
        .assign(**{"ALLOCATION AMOUNT": lambda d: d["ALLOCATION AMOUNT"].apply(lambda x: f"₹{x/1e5:.2f}L"),
                   "WORK": lambda d: d["WORK"].str[:70]})
        .head(20),
        use_container_width=True
    )
 
# ══════════════════════════════════════════════════
# BENFORD'S LAW
# ══════════════════════════════════════════════════
elif page == "📐 Benford's Law":
    st.title("📐 Benford's Law — Allocation Amount Analysis")
    st.markdown("""
    Real financial transactions follow **Benford's Law** — leading digit *d* appears with
    probability log₁₀(1 + 1/*d*). Fabricated amounts deviate from this.
    Used by **IRS, AUSTRAC, FinCEN** for forensic accounting. **Zero labels needed.**
    """)
 
    tab1, tab2 = st.tabs(["Overall Dataset", "Per MP Analysis"])
 
    with tab1:
        result = benford_analysis(df)
        c1,c2,c3 = st.columns(3)
        c1.metric("Chi² Statistic", f"{result['chi2']:.2f}")
        c2.metric("MAD Score", f"{result['mad']:.4f}")
        c3.metric("Conformity", result["conformity"])
 
        obs = result["observed_freq"].reset_index()
        obs.columns = ["Digit","Observed"]
        exp = result["expected_freq"].reset_index()
        exp.columns = ["Digit","Expected"]
 
        fig = go.Figure()
        fig.add_trace(go.Bar(x=obs["Digit"], y=obs["Observed"],
                             name="Observed", marker_color="#E74C3C", opacity=0.8))
        fig.add_trace(go.Scatter(x=exp["Digit"], y=exp["Expected"],
                                 name="Benford Expected", mode="lines+markers",
                                 line=dict(color="#2C3E50", width=2, dash="dash")))
        fig.update_layout(title="Leading Digit Distribution of Allocation Amounts",
                          xaxis_title="Leading Digit", yaxis_title="Frequency",
                          height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Bars close to dashed line = natural. Large gaps = potential fabrication.")
 
    with tab2:
        st.subheader("Per-MP Benford Analysis — Most Suspicious MPs")
        st.caption("MPs with fewer than 8 works are excluded — Benford's Law needs enough digits to be statistically meaningful.")
        mp_benford = benford_by_mp(df).copy()
 
        # Defaults mirror the MAD conformity bands used in utils/benford.py
        # (0.006 close conformity, 0.012 acceptable, 0.015 non-conforming/high risk)
        max_mad = float(mp_benford["mad"].max()) if not mp_benford.empty else 0.02
        max_mad = max(max_mad, 0.015)  # keep the default bands reachable even if data is cleaner than that
        mad_threshold = st.slider(
            "MAD threshold for 'High' Benford risk",
            0.0, round(max_mad, 3), 0.015, step=0.001,
            help="Defaults to the standard non-conforming threshold (0.015) used in benford_analysis(). "
                 "Lower it to flag more MPs; raise it to flag only the most extreme cases."
        )
        medium_threshold = mad_threshold * 0.8  # mirrors the 0.012/0.015 ratio in benford.py
        mp_benford["benford_risk"] = mp_benford["mad"].apply(
            lambda m: "High" if m >= mad_threshold else ("Medium" if m >= medium_threshold else "Low")
        )
 
        color_map = {"High":"#E74C3C","Medium":"#F39C12","Low":"#2ECC71"}
        top20 = mp_benford.sort_values("mad", ascending=False).head(20)
        fig2 = px.bar(top20, x="MP NAME", y="mad", color="benford_risk",
                      color_discrete_map=color_map,
                      title="Top 20 MPs by Benford MAD Score")
        fig2.update_layout(height=380, xaxis_tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(mp_benford[["MP NAME","STATE","works_count","chi2","mad","conformity","benford_risk"]],
                     use_container_width=True)
 
        st.markdown("---")
        jc1, jc2 = st.columns([2, 1])
        jump_mp = jc1.selectbox("Jump to a flagged MP's worklist", ["—"] + top20["MP NAME"].tolist())
        if jc2.button("🚨 Open in Fraud Worklist", disabled=(jump_mp == "—")):
            st.session_state["jump_to_mp"] = jump_mp
            st.success(f"Ready — open '🚨 Fraud Worklist' from the sidebar to see {jump_mp}'s flagged works.")
 
# ══════════════════════════════════════════════════
# MP NETWORK
# ══════════════════════════════════════════════════
elif page == "🕸️ MP Network":
    st.title("🕸️ MP–State Network Graph")
    st.markdown("NetworkX bipartite graph: **MPs** ↔ **States**. Red MPs flagged for high work volume or IDA rejections.")
 
    summary = graph_summary(G)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total MPs", summary["total_mps"])
    c2.metric("States", summary["total_states"])
    c3.metric("Connections", summary["total_edges"])
    c4.metric("Flagged MPs", summary["flagged_mps"], delta_color="inverse")
 
    flagged_df = get_flagged_mps(G)
    if flagged_df.empty:
        st.success("✅ No MPs are currently flagged under the network criteria.")
    else:
        st.subheader("Flagged MPs")
        st.dataframe(flagged_df, use_container_width=True)
 
        st.subheader("Network Visualization (Flagged MPs)")
        flagged_ids = flagged_df["MP NAME"].tolist()[:15]
        subG = G.subgraph(flagged_ids + [n for mp in flagged_ids for n in G.neighbors(mp)])
        pos = nx.spring_layout(subG, seed=RANDOM_SEED, k=1.5)
 
        # Total allocation per state, used to size/color state nodes below
        state_alloc = df.groupby("STATE")["ALLOCATION AMOUNT"].sum()
        max_state_alloc = state_alloc.max() if not state_alloc.empty else 0
 
        ex, ey = [], []
        for u, v in subG.edges():
            x0,y0 = pos[u]; x1,y1 = pos[v]
            ex += [x0,x1,None]; ey += [y0,y1,None]
 
        nx_,ny_,nt_,nc_,ns_ = [],[],[],[],[]
        for node in subG.nodes():
            x,y = pos[node]; nd = subG.nodes[node]
            nx_.append(x); ny_.append(y)
            if nd.get("node_type") == "mp":
                nt_.append(f"{node}<br>Works: {nd.get('work_count',0)}")
                nc_.append("#E74C3C" if nd.get("flagged") else "#F39C12")
                ns_.append(15 + nd.get("work_count",1)//5)
            else:
                alloc = state_alloc.get(node, 0)
                nt_.append(f"{node}<br>Total Allocation: ₹{alloc/1e7:.1f} Cr")
                nc_.append("#3498DB")
                size = 12 + (alloc / max_state_alloc * 30 if max_state_alloc > 0 else 0)
                ns_.append(size)
 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                 line=dict(width=0.8, color="#AAAAAA"), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers+text",
                                 marker=dict(size=ns_, color=nc_,
                                             line=dict(width=1, color="white")),
                                 text=nt_, textposition="top center", hoverinfo="text"))
        fig.update_layout(height=520, showlegend=False,
                          xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                          yaxis=dict(showgrid=False,zeroline=False,showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Red = flagged MP | 🟠 Orange = unflagged MP | 🔵 Blue state node size ∝ total allocation")
 
# ══════════════════════════════════════════════════
# IDA FRAUD RINGS (tripartite MP <-> IDA <-> STATE)
# ══════════════════════════════════════════════════
elif page == "🔗 IDA Fraud Rings":
    st.title("🔗 IDA Fraud Ring Detection")
    st.markdown(
        "A single MP's anomaly score only tells you about one legislator. "
        "This view looks at **implementing agencies (IDAs)** instead — the contractors/agencies "
        "actually carrying out the works. An IDA that repeats across **many different MPs and states** "
        "with a **high rejection rate** is a stronger fraud-ring signal than any one MP's behavior, "
        "since it points at a common agency implicated in questionable works for multiple legislators."
    )
 
    summary = ida_graph_summary(G_ida)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total IDAs", summary["total_idas"])
    c2.metric("MPs Involved", summary["total_mps"])
    c3.metric("States Involved", summary["total_states"])
    c4.metric("Connections", summary["total_edges"])
    c5.metric("Flagged IDAs", summary["flagged_idas"], delta_color="inverse")
 
    st.caption(
        f"An IDA is flagged when it appears across ≥{MIN_IDA_MPS} distinct MPs, "
        f"≥{MIN_IDA_STATES} distinct states, and ≥{MIN_IDA_REJECTION_RATE*100:.0f}% "
        "of its works were rejected by IDA approval."
    )
 
    flagged_idas = get_flagged_idas(G_ida)
    if flagged_idas.empty:
        st.success("✅ No IDAs currently meet the fraud-ring flagging criteria.")
    else:
        st.subheader("Flagged IDAs — Ranked by Reach")
        st.dataframe(flagged_idas, use_container_width=True)
 
        st.download_button(
            "⬇️ Download flagged IDAs (CSV)",
            data=flagged_idas.to_csv(index=False).encode("utf-8"),
            file_name="flagged_ida_fraud_rings.csv",
            mime="text/csv",
        )
 
        st.markdown("---")
        st.subheader("Network Visualization — Top Flagged IDAs")
        top_ida_names = flagged_idas["IDA"].tolist()[:10]
        top_ida_nodes = [("ida", name) for name in top_ida_names]
        neighbor_nodes = set()
        for node in top_ida_nodes:
            neighbor_nodes.update(G_ida.neighbors(node))
        subG = G_ida.subgraph(set(top_ida_nodes) | neighbor_nodes)
        pos = nx.spring_layout(subG, seed=RANDOM_SEED, k=1.3)
 
        ex, ey = [], []
        for u, v in subG.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ex += [x0, x1, None]; ey += [y0, y1, None]
 
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node in subG.nodes():
            x, y = pos[node]
            nd = subG.nodes[node]
            node_x.append(x); node_y.append(y)
            ntype = nd.get("node_type")
            label = nd.get("label", node[1] if isinstance(node, tuple) else str(node))
            if ntype == "ida":
                node_text.append(
                    f"{label}<br>Works: {nd.get('work_count',0)}<br>"
                    f"MPs: {nd.get('distinct_mps',0)} | States: {nd.get('distinct_states',0)}<br>"
                    f"Rejection rate: {nd.get('rejection_rate',0)*100:.0f}%"
                )
                node_color.append("#E74C3C" if nd.get("flagged") else "#9B59B6")
                node_size.append(16 + min(30, nd.get("work_count", 1) // 3))
            elif ntype == "mp":
                node_text.append(f"{label} (MP)")
                node_color.append("#F39C12")
                node_size.append(14)
            else:
                node_text.append(f"{label} (State)")
                node_color.append("#3498DB")
                node_size.append(16)
 
        fig_ida = go.Figure()
        fig_ida.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                      line=dict(width=0.7, color="#AAAAAA"), hoverinfo="none"))
        fig_ida.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                                      marker=dict(size=node_size, color=node_color,
                                                  line=dict(width=1, color="white")),
                                      text=node_text, hoverinfo="text"))
        fig_ida.update_layout(height=560, showlegend=False,
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig_ida, use_container_width=True)
        st.caption("🔴 Red = flagged IDA | 🟣 Purple = unflagged IDA (shown as neighbor) | "
                   "🟠 Orange = MP | 🔵 Blue = State")
 
# ══════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════
elif page == "📖 Architecture":
    st.title("📖 Architecture & Methodology")
 
    st.subheader("Detection Stack")
    st.code("""
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
DETECTION ENGINE (unsupervised — zero labels)
├── Isolation Forest  → anomaly score
└── Local Outlier Factor → density anomaly score
      ↓
FORENSIC LAYER (zero labels)
└── Benford's Law → invoice digit distribution per MP
      ↓
XGBoost Risk Calibration
└── Combines unsupervised signals → one risk score
      ↓
SHAP → Rule Templates → Plain English explanation
      ↓
NetworkX Graph → Flagged MPs by work volume + rejections
      ↓
Streamlit Dashboard → Ranked worklist for auditors
    """, language="text")
 
    st.subheader("Why unsupervised?")
    st.success("No real auditor-confirmed fraud labels exist for MPLADS data. Our primary detection — Isolation Forest, LOF, and Benford's Law — needs zero labels. XGBoost only calibrates their combined output using weak supervision as a stand-in for the active learning loop in production.")
 
    st.subheader("Future Scope")
    for item in [
        "Live ETL pipeline scraping mplads.gov.in + state portals with schema normalization",
        "Full Louvain community detection for multi-hop MP-contractor fraud rings",
        "LLM-generated natural language justifications replacing rule templates",
        "Active learning loop — auditor confirms/dismisses flags → model retrains on real labels",
        "Production React/Next.js + FastAPI stack with role-based audit workflow",
        "Graceful handling of missing contractor-ID fields across states",
    ]:
        st.markdown(f"- {item}")
 
    st.subheader("Team Gaslighters | SIH 2026 | JAIN University")
 