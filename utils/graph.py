"""
graph.py — NetworkX graphs for MPLADS
1. MP <-> STATE bipartite graph — flags MPs with abnormally high work counts
   or rejection patterns (build_mp_graph / get_flagged_mps / graph_summary).
2. MP <-> IDA <-> STATE tripartite graph — flags implementing agencies (IDAs)
   that repeat across many different MPs and states with an elevated
   rejection rate: a much stronger fraud-ring signal than any single MP's
   behavior, since it points at a common contractor/agency rather than
   one legislator (build_ida_graph / get_flagged_idas / ida_graph_summary).
"""
import networkx as nx
import pandas as pd
from config import MIN_IDA_MPS, MIN_IDA_STATES, MIN_IDA_REJECTION_RATE
 
 
def build_mp_graph(df: pd.DataFrame):
    G = nx.Graph()
 
    for state in df["STATE"].unique():
        G.add_node(state, node_type="state")
 
    mp_stats = df.groupby("MP NAME").agg(
        work_count=("WORK", "count"),
        total_amount=("ALLOCATION AMOUNT", "sum"),
        rejected=("ida_rejected", "sum"),
        unsanctioned=("is_unsanctioned", "sum"),
        state=("STATE", "first")
    ).reset_index()
 
    for _, row in mp_stats.iterrows():
        flagged = (row["work_count"] >= 50) or (row["rejected"] >= 5)
        G.add_node(row["MP NAME"], node_type="mp",
                   work_count=int(row["work_count"]),
                   total_amount=int(row["total_amount"]),
                   rejected=int(row["rejected"]),
                   flagged=flagged)
        G.add_edge(row["MP NAME"], row["state"],
                   weight=int(row["work_count"]))
 
    return G
 
 
def get_flagged_mps(G) -> pd.DataFrame:
    rows = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "mp" and data.get("flagged"):
            rows.append({
                "MP NAME": node,
                "work_count": data["work_count"],
                "total_amount_lakhs": round(data["total_amount"] / 1e5, 2),
                "ida_rejections": data["rejected"],
            })
    return pd.DataFrame(rows).sort_values("work_count", ascending=False)
 
 
def graph_summary(G):
    mps = [n for n, d in G.nodes(data=True) if d.get("node_type") == "mp"]
    states = [n for n, d in G.nodes(data=True) if d.get("node_type") == "state"]
    flagged = [n for n, d in G.nodes(data=True)
               if d.get("node_type") == "mp" and d.get("flagged")]
    return {
        "total_mps": len(mps),
        "total_states": len(states),
        "total_edges": G.number_of_edges(),
        "flagged_mps": len(flagged),
    }
 
 
# ══════════════════════════════════════════════════════════════
# TRIPARTITE MP <-> IDA <-> STATE GRAPH — fraud ring detection
# ══════════════════════════════════════════════════════════════
 
# Node ids are (node_type, name) tuples rather than plain names, because an
# IDA name and an MP name could theoretically collide as raw strings across
# three node types in one graph — the type prefix keeps them distinct.
# Thresholds (MIN_IDA_MPS, MIN_IDA_STATES, MIN_IDA_REJECTION_RATE) live in
# config.py alongside the project's other tunable thresholds.
 
 
def build_ida_graph(df: pd.DataFrame):
    """
    Builds a tripartite graph MP <-> IDA <-> STATE.
 
    An IDA (implementing agency) that shows up across many different MPs
    and states, with an elevated rejection rate, is a stronger fraud-ring
    signal than a single MP's anomaly score — it suggests a common
    contractor/agency involved in questionable works for multiple
    legislators, not just one MP's behavior.
    """
    G = nx.Graph()
    df_ida = df.dropna(subset=["IDA"])
 
    for state in df_ida["STATE"].unique():
        G.add_node(("state", state), node_type="state", label=state)
 
    for mp in df_ida["MP NAME"].unique():
        G.add_node(("mp", mp), node_type="mp", label=mp)
 
    ida_stats = df_ida.groupby("IDA").agg(
        work_count=("WORK", "count"),
        distinct_mps=("MP NAME", "nunique"),
        distinct_states=("STATE", "nunique"),
        rejected=("ida_rejected", "sum"),
        total_amount=("ALLOCATION AMOUNT", "sum"),
    ).reset_index()
 
    for _, row in ida_stats.iterrows():
        rejection_rate = row["rejected"] / row["work_count"] if row["work_count"] else 0.0
        flagged = (
            row["distinct_mps"] >= MIN_IDA_MPS and
            row["distinct_states"] >= MIN_IDA_STATES and
            rejection_rate >= MIN_IDA_REJECTION_RATE
        )
        G.add_node(("ida", row["IDA"]), node_type="ida",
                   label=row["IDA"],
                   work_count=int(row["work_count"]),
                   distinct_mps=int(row["distinct_mps"]),
                   distinct_states=int(row["distinct_states"]),
                   rejected=int(row["rejected"]),
                   rejection_rate=float(rejection_rate),
                   total_amount=int(row["total_amount"]),
                   flagged=bool(flagged))
 
    mp_ida = df_ida.groupby(["MP NAME", "IDA"]).size().reset_index(name="w")
    for _, r in mp_ida.iterrows():
        G.add_edge(("mp", r["MP NAME"]), ("ida", r["IDA"]), weight=int(r["w"]))
 
    ida_state = df_ida.groupby(["IDA", "STATE"]).size().reset_index(name="w")
    for _, r in ida_state.iterrows():
        G.add_edge(("ida", r["IDA"]), ("state", r["STATE"]), weight=int(r["w"]))
 
    return G
 
 
def get_flagged_idas(G) -> pd.DataFrame:
    rows = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "ida" and data.get("flagged"):
            rows.append({
                "IDA": data["label"],
                "work_count": data["work_count"],
                "distinct_mps": data["distinct_mps"],
                "distinct_states": data["distinct_states"],
                "rejection_rate_pct": round(data["rejection_rate"] * 100, 1),
                "total_amount_lakhs": round(data["total_amount"] / 1e5, 2),
            })
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    return df_out.sort_values(["distinct_mps", "rejection_rate_pct"], ascending=False)
 
 
def ida_graph_summary(G):
    idas = [n for n, d in G.nodes(data=True) if d.get("node_type") == "ida"]
    mps = [n for n, d in G.nodes(data=True) if d.get("node_type") == "mp"]
    states = [n for n, d in G.nodes(data=True) if d.get("node_type") == "state"]
    flagged = [n for n, d in G.nodes(data=True)
               if d.get("node_type") == "ida" and d.get("flagged")]
    return {
        "total_idas": len(idas),
        "total_mps": len(mps),
        "total_states": len(states),
        "total_edges": G.number_of_edges(),
        "flagged_idas": len(flagged),
    }
 