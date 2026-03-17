"""
F1 Strategy & Race Replay
==========================
Unified app — two top-level tabs:
  Strategy Recommender  — ML-powered forward-looking strategy tool
  Race Replay           — lap-by-lap autoplay with ML vs reality comparison

SETUP:
    pip install streamlit plotly pandas fastf1 scikit-learn numpy

RUN:
    cd ~/Downloads && streamlit run f1_app.py

REQUIRES:
    f1_core.py in the same folder
"""

import warnings
warnings.filterwarnings("ignore")

import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import fastf1

from f1_core import (
    build_degradation_model, predict_lap_time, simulate_strategy,
    simulate_full_grid, monte_carlo_simulation, calc_degradation_rates,
    load_multi_race_data, get_schedule,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    COMPOUNDS, COMPOUND_COLORS, TEAM_COLORS, YEARS,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="F1 Strategy", page_icon="🏁", layout="wide")

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Rajdhani', monospace !important; }
.f1-header {
    background: linear-gradient(90deg,#0a0a0a,#1a0005);
    border-bottom: 3px solid #e8002d; padding:14px 28px;
    margin:-1rem -1rem 1.5rem -1rem; display:flex; align-items:center; gap:16px;
}
.f1-logo { background:#e8002d; color:#fff; font-weight:900; font-size:18px;
    padding:5px 14px; letter-spacing:3px;
    clip-path:polygon(0 0,100% 0,88% 100%,12% 100%); }
.f1-title { font-size:24px; font-weight:700; color:#fff; letter-spacing:4px; text-transform:uppercase; }
.f1-title span { color:#e8002d; }
.f1-badge { margin-left:auto; font-size:11px; letter-spacing:3px; color:#27F4D2; text-transform:uppercase; }
.metric-card { background:#141414; border:1px solid #1e1e1e;
    border-left:3px solid #e8002d; border-radius:2px; padding:10px 16px; }
.metric-label { font-size:10px; letter-spacing:2px; color:#555; text-transform:uppercase; margin-bottom:2px; }
.metric-value { font-size:20px; font-weight:700; color:#fff; }
.metric-value.red{color:#e8002d;}.metric-value.green{color:#27F4D2;}
.metric-value.orange{color:#FF8000;}.metric-value.yellow{color:#FFF200;}
.rec-card { background:#0f0f0f; border:1px solid #222; border-radius:4px; padding:16px 20px; margin-bottom:12px; }
.rec-title { font-size:13px; letter-spacing:2px; color:#555; text-transform:uppercase; margin-bottom:6px; }
.rec-value { font-size:28px; font-weight:700; }
.rec-sub { font-size:12px; color:#666; margin-top:4px; }
.warn-box { background:#1a0005; border:1px solid #e8002d44; border-left:3px solid #e8002d;
    padding:10px 14px; border-radius:2px; font-size:13px; color:#aaa; }
.event-card { background:#0f0f0f; border:1px solid #1e1e1e; border-radius:3px;
    padding:8px 12px; margin:3px 0; font-size:12px; }
.event-good  { border-left:3px solid #27F4D2; color:#27F4D2; }
.event-bad   { border-left:3px solid #e8002d; color:#e8002d; }
.event-warn  { border-left:3px solid #FFF200; color:#FFF200; }
.event-info  { border-left:3px solid #888;    color:#aaa; }
.driver-row  { display:flex; align-items:center; gap:8px; padding:3px 6px;
    border-radius:2px; margin:1px 0; font-size:12px; font-weight:600; }
section[data-testid="stSidebar"] { background:#0a0a0a !important; border-right:1px solid #1a1a1a !important; }
.stTabs [data-baseweb="tab-list"] { background:#0f0f0f; border-bottom:1px solid #222; gap:4px; }
.stTabs [data-baseweb="tab"] { background:#141414 !important; color:#666 !important;
    border:1px solid #222 !important; border-radius:2px 2px 0 0 !important;
    font-family:'Rajdhani',monospace !important; font-size:13px; }
.stTabs [aria-selected="true"] { background:#1a0005 !important; color:#e8002d !important;
    border-color:#e8002d !important; }
</style>
""", unsafe_allow_html=True)

# ── Shared constants ──────────────────────────────────────────
CARD, WHITE = "#141414", "#f0f0f0"
COMPOUND_SHORT = {"SOFT":"S","MEDIUM":"M","HARD":"H","INTERMEDIATE":"I","WET":"W","UNKNOWN":"?"}
SESSION_PALETTE = ["#E8002D","#3671C6","#FF8000","#27F4D2","#FF87BC","#52E252","#FFD700","#B6BABD"]
LAYOUT_BASE = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(family="Rajdhani, monospace", color=WHITE, size=13),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)
AXIS = dict(gridcolor="#1e1e1e", zerolinecolor="#1e1e1e")

def team_color(team):
    for k, v in TEAM_COLORS.items():
        if k.lower() in str(team).lower(): return v
    return "#888"

def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(color="#555", size=14))
    fig.update_layout(**LAYOUT_BASE)
    return fig

# ═══════════════════════════════════════════════════════════════
#  STRATEGY TAB — charts
# ═══════════════════════════════════════════════════════════════
def chart_feature_importance(model):
    if model is None: return empty_fig("Model not trained")
    try:
        pre       = model.named_steps["preprocessor"]
        num_names = NUMERIC_FEATURES
        cat_names = list(pre.named_transformers_["cat"]
                            .get_feature_names_out(CATEGORICAL_FEATURES))
        all_names = num_names + cat_names
        importances = model.named_steps["model"].feature_importances_
        agg = {}
        for name, imp in zip(all_names, importances):
            group = name.split("_")[0] if "_" in name and name.split("_")[0] in CATEGORICAL_FEATURES else name
            agg[group] = agg.get(group, 0) + imp
        names_agg = list(agg.keys())
        imps_agg  = list(agg.values())
        order     = np.argsort(imps_agg)
        colors    = [SESSION_PALETTE[i % len(SESSION_PALETTE)] for i in range(len(names_agg))]
        fig = go.Figure(go.Bar(
            x=[imps_agg[i] for i in order], y=[names_agg[i] for i in order],
            orientation="h", marker_color=[colors[i] for i in order],
            hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>"))
        fig.update_layout(**LAYOUT_BASE, title="Feature Importances",
                          xaxis=dict(title="Importance", **AXIS), yaxis=dict(**AXIS))
        return fig
    except Exception as e:
        return empty_fig(f"Could not build: {e}")

def chart_degradation(laps_df, model):
    fig = go.Figure()
    for comp in COMPOUNDS:
        df = laps_df[laps_df["Compound"] == comp].copy()
        if df.empty: continue
        color = COMPOUND_COLORS[comp]
        med = df.groupby("TyreLife")["LapTimeSec"].median().reset_index()
        med = med[med["TyreLife"] <= 40]
        fig.add_trace(go.Scatter(x=med["TyreLife"], y=med["LapTimeSec"],
            mode="lines+markers", name=f"{comp} actual",
            line=dict(color=color, width=2), marker=dict(size=4, opacity=0.6)))
        if model:
            life_range = np.arange(1, 41)
            preds = [predict_lap_time(model, l, comp, 0.5, 2) for l in life_range]
            fig.add_trace(go.Scatter(x=life_range, y=preds,
                mode="lines", name=f"{comp} model",
                line=dict(color=color, width=2, dash="dash")))
    fig.update_layout(**LAYOUT_BASE, title="Tyre Degradation — Actual vs Model",
                      xaxis=dict(title="Tyre Life (laps)", **AXIS),
                      yaxis=dict(title="Lap Time (s)", **AXIS))
    return fig

def chart_strategy_comparison(strategy_results, current_lap, total_laps):
    if not strategy_results: return empty_fig()
    names = list(strategy_results.keys())
    times = [v["total_time"] for v in strategy_results.values()]
    min_time = min(times)
    colors = ["#27F4D2" if t == min_time else "#333" for t in times]
    deltas = [t - min_time for t in times]
    fig = go.Figure(go.Bar(x=names, y=times, marker_color=colors,
        text=[f"+{d:.1f}s" if d > 0 else "✓ BEST" for d in deltas],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}s total<extra></extra>"))
    fig.update_layout(**LAYOUT_BASE, title="Strategy Comparison — Predicted Total Race Time",
                      xaxis=dict(**AXIS), yaxis=dict(title="Time (s)", **AXIS))
    return fig

def chart_race_situation(current_lap, total_laps, current_compound,
                          current_tyre_life, pit_recommendations, strategy_results):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[current_lap], y=["Race Progress"], orientation="h",
        marker_color="#e8002d", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Bar(x=[total_laps - current_lap], y=["Race Progress"],
        orientation="h", marker_color="#1a1a1a", showlegend=False,
        hoverinfo="skip", base=current_lap))
    if pit_recommendations:
        best = min(pit_recommendations, key=lambda x: x["total_time"])
        for pit_lap in best.get("pit_laps", []):
            fig.add_vline(x=pit_lap, line_dash="dash", line_color="#FF8000",
                          annotation_text=f"PIT L{pit_lap}", annotation_font_color="#FF8000")
    fig.add_vline(x=current_lap, line_color="#e8002d",
                  annotation_text=f"NOW L{current_lap}", annotation_font_color="#e8002d")
    fig.update_layout(**LAYOUT_BASE, title="Race Situation", barmode="stack",
                      xaxis=dict(title="Lap", range=[0, total_laps], **AXIS),
                      yaxis=dict(**AXIS), height=150)
    return fig

def chart_lap_times_history(laps_df, circuit):
    fig = go.Figure()
    for comp in COMPOUNDS:
        df = laps_df[laps_df["Compound"] == comp]
        if df.empty: continue
        color = COMPOUND_COLORS[comp]
        r,g,b = (int(color.lstrip("#")[j:j+2],16) for j in (0,2,4))
        fig.add_trace(go.Violin(y=df["LapTimeSec"], name=comp,
            box_visible=True, meanline_visible=True,
            line_color=color, fillcolor=f"rgba({r},{g},{b},0.15)"))
    fig.update_layout(**LAYOUT_BASE, title=f"Historical Lap Times at {circuit}",
                      yaxis=dict(title="Lap Time (s)", **AXIS), violinmode="group")
    return fig

def chart_monte_carlo(mc_results, best_strat):
    if not mc_results:
        return empty_fig("Run Monte Carlo first"), go.Figure()
    names  = list(mc_results.keys())
    colors = [SESSION_PALETTE[i % len(SESSION_PALETTE)] for i in range(len(names))]
    fig1   = go.Figure()
    for name, color, times in zip(names, colors, mc_results.values()):
        r,g,b = (int(color.lstrip("#")[j:j+2],16) for j in (0,2,4))
        fig1.add_trace(go.Violin(y=times, name=name, box_visible=True,
            meanline_visible=True, line_color=color,
            fillcolor=f"rgba({r},{g},{b},0.15)", opacity=0.85,
            hovertemplate=f"<b>{name}</b><br>%{{y:.1f}}s<extra></extra>"))
    fig1.update_layout(**LAYOUT_BASE, title="MC — Total Race Time Distribution",
                       yaxis=dict(title="Total Race Time (s)", **AXIS), violinmode="group")
    n_sims    = len(next(iter(mc_results.values())))
    all_times = np.stack(list(mc_results.values()), axis=1)
    winners   = np.argmin(all_times, axis=1)
    win_pcts  = [(winners == i).sum() / n_sims * 100 for i in range(len(names))]
    bar_colors = ["#27F4D2" if n == best_strat else "#333" for n in names]
    fig2 = go.Figure(go.Bar(x=names, y=win_pcts, marker_color=bar_colors,
        text=[f"{p:.1f}%" for p in win_pcts], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Wins %{y:.1f}%<extra></extra>"))
    fig2.update_layout(**LAYOUT_BASE, title=f"Win Probability ({n_sims:,} sims)",
                       xaxis=dict(**AXIS), yaxis=dict(title="Win %", range=[0,115], **AXIS))
    return fig1, fig2

def mc_confidence_table(mc_results):
    rows = []
    best_p50 = min(np.percentile(v, 50) for v in mc_results.values())
    for name, times in mc_results.items():
        p10, p50, p90 = np.percentile(times, [10,50,90])
        n = len(times)
        win_pct = sum(times[i] == min(v[i] for v in mc_results.values())
                      for i in range(n)) / n * 100
        rows.append({"Strategy":name, "P10":f"{p10:.1f}s", "P50":f"{p50:.1f}s",
                     "P90":f"{p90:.1f}s", "Spread":f"±{(p90-p10)/2:.1f}s",
                     "Win %":f"{win_pct:.1f}%",
                     "vs Median":f"+{p50-best_p50:.1f}s" if p50>best_p50 else "BEST"})
    return pd.DataFrame(rows).sort_values("P50")

# ═══════════════════════════════════════════════════════════════
#  REPLAY TAB — data loading + charts
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def load_full_race(year, round_number):
    try:
        sess = fastf1.get_session(year, round_number, "R")
        sess.load(laps=True, telemetry=False, weather=True, messages=False)
        laps = sess.laps.copy()
        laps["LapTimeSec"] = laps["LapTime"].apply(
            lambda t: t.total_seconds() if pd.notna(t) and hasattr(t,"total_seconds") else np.nan)
        pos_cols = ["Driver","LapNumber","Position","LapTimeSec","Compound",
                    "TyreLife","Stint","Team","PitInTime","PitOutTime"]
        laps = laps[[c for c in pos_cols if c in laps.columns]].copy()
        if "Position" not in laps.columns or laps["Position"].isna().all():
            laps["CumTime"]  = laps.groupby("Driver")["LapTimeSec"].cumsum()
            laps["Position"] = laps.groupby("LapNumber")["CumTime"].rank().astype(int)
        else:
            laps["Position"] = pd.to_numeric(laps["Position"], errors="coerce")
        laps["Pitted"] = laps["PitInTime"].notna() if "PitInTime" in laps.columns else False
        weather_summary = {}
        if not sess.weather_data.empty:
            weather_summary = {
                "TrackTemp": float(sess.weather_data["TrackTemp"].mean()),
                "AirTemp":   float(sess.weather_data["AirTemp"].mean()),
                "Rainfall":  bool(sess.weather_data.get("Rainfall", pd.Series([False])).any()),
            }
        results = sess.results[["Abbreviation","FullName","TeamName",
                                  "Position","GridPosition","Points","Status"]].copy() \
            if not sess.results.empty else pd.DataFrame()
        return {"laps": laps, "results": results,
                "total_laps": int(laps["LapNumber"].max()),
                "weather": weather_summary,
                "event_name": sess.event["EventName"], "year": year}
    except Exception as e:
        return None

def extract_stints(laps_df, driver):
    drv = laps_df[laps_df["Driver"] == driver].sort_values("LapNumber")
    stints, current_compound, stint_start = [], None, None
    for _, row in drv.iterrows():
        comp = str(row.get("Compound","UNKNOWN")).upper()
        lap  = int(row["LapNumber"])
        if comp != current_compound:
            if current_compound is not None:
                stints.append({"compound":current_compound,"start":stint_start,"end":lap-1})
            current_compound, stint_start = comp, lap
    if current_compound:
        stints.append({"compound":current_compound,"start":stint_start,
                        "end":int(drv["LapNumber"].max())})
    return stints

def ml_optimal_strategy(ml_model, current_lap, total_laps, current_compound,
                         tyre_life, stint, driver=None, team=None,
                         track_temp=35.0, air_temp=22.0):
    if ml_model is None: return "No model", [], []
    remaining = total_laps - current_lap
    if remaining <= 0: return "Stay Out", [], []
    mid, early, late = (current_lap + remaining//2,
                        current_lap + int(remaining*0.35),
                        current_lap + int(remaining*0.65))
    next_compounds = [c for c in COMPOUNDS if c != current_compound]
    nc1 = next_compounds[0] if next_compounds else "MEDIUM"
    nc2 = next_compounds[1] if len(next_compounds) > 1 else "HARD"
    strategies = [
        ("Stay Out",           [],           []),
        (f"1-Stop Early→{nc1}", [early],     [nc1]),
        (f"1-Stop Mid→{nc1}",   [mid],       [nc1]),
        (f"1-Stop Late→{nc1}",  [late],      [nc1]),
        (f"2-Stop",            [early,mid+5],[nc1,nc2]),
    ]
    results = simulate_strategy(ml_model, current_lap, total_laps,
                                 current_compound, tyre_life, stint, strategies,
                                 track_temp=track_temp, air_temp=air_temp,
                                 driver=driver, team=team)
    best = min(results, key=lambda k: results[k]["total_time"])
    return best, results[best]["pit_laps"], results[best]["compounds"]

def precompute_ml_recommendations(ml_model, laps_df, total_laps,
                                   track_temp=35.0, air_temp=22.0):
    recs = {}
    if ml_model is None: return recs
    for drv in laps_df["Driver"].unique():
        recs[drv] = {}
        drv_laps  = laps_df[laps_df["Driver"] == drv].sort_values("LapNumber")
        if drv_laps.empty: continue
        team = drv_laps["Team"].iloc[0] if "Team" in drv_laps.columns else None
        for _, row in drv_laps.iterrows():
            lap       = int(row["LapNumber"])
            comp      = str(row.get("Compound","MEDIUM")).upper()
            tyre_life = int(row.get("TyreLife",1)) if pd.notna(row.get("TyreLife")) else 1
            stint     = int(row.get("Stint",1))    if pd.notna(row.get("Stint"))    else 1
            if comp not in COMPOUNDS: comp = "MEDIUM"
            name, pit_laps, compounds = ml_optimal_strategy(
                ml_model, lap, total_laps, comp, tyre_life, stint,
                driver=drv, team=team, track_temp=track_temp, air_temp=air_temp)
            recs[drv][lap] = {"rec":name, "pit_laps":pit_laps, "compounds":compounds}
    return recs

def evaluate_strategy_accuracy(actual_stints, ml_recs_for_driver, total_laps):
    if not actual_stints: return 0
    actual_pit_laps  = [s["start"] for s in actual_stints[1:]]
    actual_compounds = [s["compound"] for s in actual_stints]
    first_rec        = ml_recs_for_driver.get(1, {}) if ml_recs_for_driver else {}
    rec_pit_laps     = first_rec.get("pit_laps", [])
    rec_compounds    = first_rec.get("compounds", [])
    score = 0
    if len(actual_pit_laps) == len(rec_pit_laps): score += 40
    for rec_lap in rec_pit_laps:
        for actual_lap in actual_pit_laps:
            if   abs(actual_lap - rec_lap) <= 3: score += 30; break
            elif abs(actual_lap - rec_lap) <= 6: score += 15; break
    for comp in rec_compounds:
        if comp in actual_compounds:
            score += 30 // max(1, len(rec_compounds))
    return min(100, score)

def build_race_events(laps_df, total_laps):
    events = {lap: [] for lap in range(1, total_laps+1)}
    if "Pitted" in laps_df.columns:
        for _, row in laps_df[laps_df["Pitted"]==True].iterrows():
            lap  = int(row["LapNumber"])
            comp = COMPOUND_SHORT.get(str(row.get("Compound","?")).upper(),"?")
            events[lap].append({"type":"pit","text":f"{row['Driver']} pits → {comp}","driver":row["Driver"]})
    if "Position" in laps_df.columns:
        tmp = laps_df.copy()
        tmp["PosDelta"] = tmp.groupby("Driver")["Position"].diff()
        for _, row in tmp[tmp["PosDelta"] <= -3].iterrows():
            events[int(row["LapNumber"])].append(
                {"type":"gain","text":f"{row['Driver']} +{int(abs(row['PosDelta']))} pos","driver":row["Driver"]})
        for _, row in tmp[tmp["PosDelta"] >= 3].iterrows():
            events[int(row["LapNumber"])].append(
                {"type":"loss","text":f"{row['Driver']} -{int(row['PosDelta'])} pos","driver":row["Driver"]})
    if "LapTimeSec" in laps_df.columns:
        sess_med = laps_df["LapTimeSec"].median()
        sc_laps  = laps_df.groupby("LapNumber")["LapTimeSec"].median()
        sc_laps  = sc_laps[sc_laps > sess_med * 1.15].index.tolist()
        for lap in sc_laps:
            if lap in events:
                events[lap].append({"type":"sc","text":"⚠️ Possible SC / VSC","driver":None})
    return events

# ── Replay charts ─────────────────────────────────────────────
def chart_positions(laps_df, current_lap, highlighted=None):
    fig  = go.Figure()
    if "Position" not in laps_df.columns: return fig
    data = laps_df[laps_df["LapNumber"] <= current_lap].copy()
    drivers = (data.groupby("Driver")["Position"].last()
                   .sort_values().index.tolist())
    for drv in drivers:
        grp   = data[data["Driver"]==drv].sort_values("LapNumber")
        color = team_color(grp["Team"].iloc[0] if "Team" in grp.columns else "")
        width = 3 if (highlighted and drv in highlighted) else 1.5
        opacity = 1.0 if (not highlighted or drv in highlighted) else 0.2
        fig.add_trace(go.Scatter(x=grp["LapNumber"], y=grp["Position"],
            mode="lines", name=drv, line=dict(color=color, width=width),
            opacity=opacity,
            hovertemplate=f"<b>{drv}</b> Lap %{{x}} → P%{{y}}<extra></extra>"))
    fig.add_vline(x=current_lap, line_color="#e8002d", line_dash="dash",
                  annotation_text=f"LAP {current_lap}", annotation_font_color="#e8002d")
    fig.update_layout(**LAYOUT_BASE, title="Race Positions",
        xaxis=dict(title="Lap", range=[1, laps_df["LapNumber"].max()], **AXIS),
        yaxis=dict(title="Position", autorange="reversed", dtick=1, **AXIS),
        hovermode="x unified", height=450)
    return fig

def chart_lap_times_replay(laps_df, current_lap, highlighted=None):
    fig  = go.Figure()
    data = laps_df[(laps_df["LapNumber"] <= current_lap) &
                   laps_df["LapTimeSec"].notna() &
                   (laps_df["LapTimeSec"] > 0) &
                   (laps_df["LapTimeSec"] < 200)].copy()
    for drv, grp in data.groupby("Driver"):
        color   = team_color(grp["Team"].iloc[0] if "Team" in grp.columns else "")
        opacity = 1.0 if (not highlighted or drv in highlighted) else 0.15
        fig.add_trace(go.Scatter(x=grp["LapNumber"], y=grp["LapTimeSec"],
            mode="lines", name=drv, line=dict(color=color, width=1.5),
            opacity=opacity,
            hovertemplate=f"<b>{drv}</b> L%{{x}}: %{{y:.3f}}s<extra></extra>"))
    fig.add_vline(x=current_lap, line_color="#e8002d", line_dash="dash")
    fig.update_layout(**LAYOUT_BASE, title="Lap Times",
        xaxis=dict(title="Lap", **AXIS), yaxis=dict(title="Time (s)", **AXIS),
        hovermode="x unified", height=450)
    return fig

def chart_tyre_strategies_replay(laps_df, total_laps, current_lap):
    fig  = go.Figure()
    seen = set()
    final = (laps_df[laps_df["LapNumber"] == laps_df["LapNumber"].max()]
             .sort_values("Position")[["Driver"]]
             if "Position" in laps_df.columns else pd.DataFrame())
    driver_order = final["Driver"].tolist() if not final.empty else laps_df["Driver"].unique().tolist()
    for drv in reversed(driver_order):
        for stint in extract_stints(laps_df[laps_df["LapNumber"] <= current_lap], drv):
            comp  = stint["compound"]
            color = COMPOUND_COLORS.get(comp,"#888")
            show  = comp not in seen; seen.add(comp)
            fig.add_trace(go.Bar(
                x=[stint["end"]-stint["start"]+1], y=[drv],
                base=[stint["start"]-1], orientation="h",
                marker_color=color, name=comp, legendgroup=comp, showlegend=show,
                hovertemplate=f"<b>{drv}</b> {comp} L{stint['start']}–{stint['end']}<extra></extra>"))
    fig.add_vline(x=current_lap, line_color="#e8002d", line_dash="dash")
    fig.update_layout(**LAYOUT_BASE, title="Tyre Strategies", barmode="overlay",
        xaxis=dict(title="Lap", range=[0, total_laps], **AXIS), yaxis=dict(**AXIS),
        height=max(300, len(driver_order) * 22))
    return fig

def chart_strategy_accuracy_replay(accuracy_data):
    if not accuracy_data: return empty_fig()
    df = pd.DataFrame(accuracy_data).sort_values("Score", ascending=True)
    colors = ["#27F4D2" if s>=70 else "#FF8000" if s>=40 else "#e8002d" for s in df["Score"]]
    fig = go.Figure(go.Bar(x=df["Score"], y=df["Driver"], orientation="h",
        marker_color=colors, text=[f"{s}%" for s in df["Score"]], textposition="outside",
        hovertemplate="<b>%{y}</b> %{x}% accuracy<extra></extra>"))
    fig.update_layout(**LAYOUT_BASE, title="ML Strategy Accuracy vs Actual Race",
        xaxis=dict(title="Accuracy %", range=[0,115], **AXIS), yaxis=dict(**AXIS))
    return fig

def leaderboard_html(laps_df, current_lap):
    lap_data = laps_df[laps_df["LapNumber"] == current_lap].copy()
    if lap_data.empty:
        lap_data = laps_df[laps_df["LapNumber"] == laps_df["LapNumber"].max()].copy()
    if "Position" not in lap_data.columns:
        return "<p style='color:#555'>No position data</p>"
    rows = []
    for _, row in lap_data.sort_values("Position").iterrows():
        pos    = int(row["Position"]) if pd.notna(row["Position"]) else "?"
        drv    = row["Driver"]
        comp   = COMPOUND_SHORT.get(str(row.get("Compound","?")).upper(),"?")
        t_life = int(row.get("TyreLife",0)) if pd.notna(row.get("TyreLife",0)) else "?"
        color  = team_color(row.get("Team",""))
        comp_c = COMPOUND_COLORS.get(str(row.get("Compound","?")).upper(),"#888")
        rows.append(f'<div class="driver-row" style="background:#0f0f0f">'
                    f'<span style="color:#555;width:22px;text-align:right">{pos}</span>'
                    f'<span style="color:{color};width:36px">{drv}</span>'
                    f'<span style="color:{comp_c};font-size:11px;width:18px">{comp}</span>'
                    f'<span style="color:#555;font-size:11px">{t_life}L</span></div>')
    return "".join(rows)

# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="f1-header">
    <div class="f1-logo">F1</div>
    <div class="f1-title"><span>F1</span> STRATEGY LAB</div>
    <div class="f1-badge">ML-POWERED • FASTF1</div>
</div>""", unsafe_allow_html=True)

# ── Top-level tab switcher ────────────────────────────────────
main_tab_strat, main_tab_replay = st.tabs(["🏎 Strategy Recommender", "🎬 Race Replay"])

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — context-aware per active tab
#  Streamlit sidebar renders regardless of which tab is shown,
#  so we show all controls and let the main tabs use their own.
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏁 F1 STRATEGY LAB")
    st.markdown("---")

    # ── STRATEGY section ─────────────────────────────────────
    st.markdown("### 🏎 Strategy — Circuit")
    train_year = st.selectbox("Training data from", YEARS, index=0, key="train_yr")
    with st.spinner("Loading schedule..."):
        schedule_strat = get_schedule(train_year)

    if schedule_strat:
        circuit_names = [e["EventName"].replace(" Grand Prix","").replace(" GP","")
                         for e in schedule_strat]
        circuit = st.selectbox("Circuit", circuit_names,
                                index=len(circuit_names)-1, key="circuit_sel")
        full_event_name = next(e["EventName"] for e in schedule_strat
                                if circuit in e["EventName"])
        use_years = st.multiselect("Train on seasons",
                                    [y for y in YEARS if y <= train_year],
                                    default=[train_year], key="use_years")
    else:
        circuit = None
        st.error("Could not load schedule")

    st.markdown("---")
    st.markdown("### 🌡 Conditions")
    track_temp = st.slider("Track temp (°C)", 20, 60, 35, key="track_temp")
    air_temp   = st.slider("Air temp (°C)",   10, 40, 22, key="air_temp")
    sel_driver = st.text_input("Driver code", value="VER", key="sel_driver")
    sel_team   = st.text_input("Team", value="Red Bull Racing", key="sel_team")

    st.markdown("---")
    st.markdown("### 🚦 Race Situation")
    total_laps_strat = st.number_input("Total laps", 40, 78, 58, step=1, key="tl_strat")
    current_lap_strat = st.slider("Current lap", 1, int(total_laps_strat)-5, 15, key="cl_strat")
    current_pos   = st.number_input("Position", 1, 20, 5, step=1, key="pos")
    current_compound = st.selectbox("Tyre", COMPOUNDS, index=0, key="compound")
    current_tyre_life = st.slider("Tyre age (laps)", 1, 40, 10, key="tyre_life")
    current_stint = st.number_input("Stint #", 1, 4, 1, step=1, key="stint")
    gap_ahead = st.number_input("Gap ahead (s)", 0.0, 60.0, 3.5, step=0.5, key="gap_a")
    gap_behind = st.number_input("Gap behind (s)", 0.0, 60.0, 2.0, step=0.5, key="gap_b")

    st.markdown("---")
    st.markdown("### 🚗 Traffic & SC")
    traffic_active = st.toggle("In traffic", value=False, key="traffic")
    traffic_loss   = st.slider("Traffic loss (s/lap)", 0.1, 3.0, 0.5, 0.1,
                                disabled=not traffic_active, key="tloss")
    traffic_laps_ahead = st.slider("Laps in traffic", 0, 20, 5,
                                    disabled=not traffic_active, key="tlaps")
    sc_active   = st.toggle("Safety Car deployed", value=False, key="sc")
    sc_duration = st.slider("SC duration (laps)", 1, 10, 3,
                             disabled=not sc_active, key="sc_dur")
    sc_opportunity = st.toggle("Pit under SC", value=True,
                                disabled=not sc_active, key="sc_opp")

    st.markdown("---")
    st.markdown("### 🎲 Monte Carlo")
    run_mc  = st.toggle("Run Monte Carlo", value=False, key="run_mc")
    n_sims  = st.select_slider("Simulations", [100,250,500,1000], value=500,
                                disabled=not run_mc, key="n_sims")
    sc_prob = st.slider("SC prob per lap (%)", 1, 15, 4,
                         disabled=not run_mc, key="sc_prob") / 100.0

    st.markdown("---")
    # ── REPLAY section ───────────────────────────────────────
    st.markdown("### 🎬 Replay — Race Select")
    rp_year = st.selectbox("Season", YEARS, index=1, key="rp_yr")
    with st.spinner("Loading schedule..."):
        schedule_replay = get_schedule(rp_year)

    rp_race_data = None
    if schedule_replay:
        rp_race_names = [e["EventName"].replace(" Grand Prix","") for e in schedule_replay]
        rp_race_name  = st.selectbox("Race", rp_race_names,
                                      index=len(rp_race_names)-1, key="rp_race")
        rp_full_name  = next(e["EventName"] for e in schedule_replay
                              if rp_race_name in e["EventName"])
        rp_round_num  = next(e["RoundNumber"] for e in schedule_replay
                              if rp_race_name in e["EventName"])
        with st.spinner(f"Loading {rp_race_name} {rp_year}..."):
            rp_race_data = load_full_race(rp_year, rp_round_num)
        if rp_race_data:
            st.success(f"✅ {rp_race_data['total_laps']} laps loaded")

            st.markdown("### 🤖 Replay ML Model")
            rp_train_years = st.multiselect("Train on seasons", YEARS,
                default=[y for y in YEARS if y < rp_year][:3] or [rp_year],
                key="rp_train_yrs")
            if st.button("🔬 Train Replay Model", use_container_width=True):
                with st.spinner("Loading training data..."):
                    rp_train_laps = load_multi_race_data(rp_full_name, rp_train_years)
                if not rp_train_laps.empty:
                    with st.spinner("Training model..."):
                        rp_model, rp_mae, rp_err = build_degradation_model(rp_train_laps)
                    if rp_model:
                        st.session_state["rp_model"] = rp_model
                        st.session_state["rp_mae"]   = rp_mae
                        t = rp_race_data["weather"].get("TrackTemp", 35.0)
                        a = rp_race_data["weather"].get("AirTemp",   22.0)
                        with st.spinner("Simulating full grid — all cars, all laps..."):
                                grid_sim = simulate_full_grid(
                                rp_model, rp_race_data["laps"],
                                rp_race_data["total_laps"], t, a)
                        st.session_state["rp_grid"] = grid_sim
                        with st.spinner("Computing per-driver strategy predictions..."):
                            rp_recs = precompute_ml_recommendations(
                                rp_model, rp_race_data["laps"],
                                rp_race_data["total_laps"], t, a)
                        st.session_state["rp_recs"] = rp_recs
                        st.success(f"✅ MAE ±{rp_mae:.2f}s — grid simulation ready")
                    else:
                        st.warning(f"Model error: {rp_err}")
                else:
                    st.warning("No training data found")

            if "rp_mae" in st.session_state:
                st.info(f"🤖 Model active — MAE ±{st.session_state['rp_mae']:.2f}s")

            st.markdown("### 🎮 Playback")
            play_speed = st.select_slider("Speed", [0.3,0.5,1.0,2.0,5.0], value=1.0,
                                           format_func=lambda x: f"{x}x", key="play_speed")
            st.markdown("### 🔍 Focus Drivers")
            all_drivers_rp = sorted(rp_race_data["laps"]["Driver"].unique().tolist())
            focus_drivers  = st.multiselect("Highlight", all_drivers_rp,
                                             default=[], key="focus_drivers")

    st.markdown("---")
    st.markdown("### 💾 Cache")
    if st.button("🗑 Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════
#  TAB 1 — STRATEGY RECOMMENDER
# ═══════════════════════════════════════════════════════════════
with main_tab_strat:
    if not circuit or not schedule_strat:
        st.error("Could not load schedule — check your connection.")
        st.stop()

    years_to_load = use_years if use_years else [train_year]
    with st.spinner(f"Loading {circuit} data..."):
        laps_df = load_multi_race_data(full_event_name, years_to_load)

    if laps_df.empty:
        st.error(f"No data for {circuit}. Try another circuit or season.")
    else:
        with st.spinner("Training model..."):
            model, mae, err = build_degradation_model(laps_df)
        if err:
            st.warning(f"Model: {err}")

        deg_rates  = calc_degradation_rates(laps_df)
        n_laps_data = len(laps_df)
        n_races     = laps_df["Year"].nunique() if "Year" in laps_df.columns else 1
        model_accuracy = f"±{mae:.2f}s" if mae else "N/A"
        laps_remaining = int(total_laps_strat) - int(current_lap_strat)

        # Metrics row
        cols = st.columns(6)
        for col, (label, val, cls) in zip(cols, [
            ("Circuit", circuit, "red"),
            ("Data Laps", f"{n_laps_data:,}", ""),
            ("Seasons", str(n_races), "green"),
            ("Model MAE", model_accuracy, "green" if mae and mae<1.5 else "orange"),
            ("Lap", f"{current_lap_strat}/{int(total_laps_strat)}", ""),
            ("Remaining", f"{laps_remaining} laps", "red"),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                            f'<div class="metric-value {cls}">{val}</div></div>',
                            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.plotly_chart(chart_race_situation(current_lap_strat, total_laps_strat,
            current_compound, current_tyre_life, [], {}),
            use_container_width=True, key="race_sit")

        if model is None:
            st.warning("Not enough data to generate ML recommendations.")
        else:
            remaining    = total_laps_strat - current_lap_strat
            mid          = current_lap_strat + remaining // 2
            late         = current_lap_strat + int(remaining * 0.65)
            early        = current_lap_strat + int(remaining * 0.35)
            comp_order   = sorted([c for c in COMPOUNDS if c != current_compound],
                                   key=lambda c: deg_rates.get(c, 0))
            next_comp1   = comp_order[0] if comp_order else "MEDIUM"
            next_comp2   = comp_order[1] if len(comp_order) > 1 else "HARD"
            strategies_to_sim = [
                ("Stay Out (No Stop)", [], []),
                (f"1-Stop Early → {next_comp1}", [early], [next_comp1]),
                (f"1-Stop Mid → {next_comp1}",   [mid],   [next_comp1]),
                (f"1-Stop Late → {next_comp1}",  [late],  [next_comp1]),
                (f"2-Stop → {next_comp1}/{next_comp2}", [early, mid+5], [next_comp1, next_comp2]),
            ]
            sc_lap_set = list(range(current_lap_strat+1, current_lap_strat+1+sc_duration)) if sc_active else []
            traffic_lap_set = (list(range(current_lap_strat,
                                          current_lap_strat + traffic_laps_ahead))
                               if traffic_active else [])
            if sc_active and sc_opportunity and sc_lap_set:
                strategies_to_sim.append((f"🟡 SC Pit → {next_comp1}", [sc_lap_set[0]+1], [next_comp1]))

            with st.spinner("Simulating strategies..."):
                strategy_results = simulate_strategy(
                    model, current_lap_strat, total_laps_strat,
                    current_compound, current_tyre_life, current_stint,
                    strategies_to_sim,
                    sc_laps=sc_lap_set, traffic_laps=traffic_lap_set,
                    traffic_loss=traffic_loss if traffic_active else 0.0,
                    track_temp=track_temp, air_temp=air_temp,
                    driver=sel_driver.strip().upper(), team=sel_team.strip())

            best_strat = min(strategy_results, key=lambda k: strategy_results[k]["total_time"])
            best_data  = strategy_results[best_strat]
            pit_loss   = 22.0
            undercut_possible = gap_behind < pit_loss * 0.7 and gap_ahead > 3.0
            overcut_possible  = gap_ahead < pit_loss and gap_behind > 5.0

            # Recommendations
            st.markdown("## 🎯 Strategy Recommendations")
            rc1, rc2, rc3, rc4 = st.columns(4)
            stops = len(best_data["pit_laps"])
            stop_label = "NO STOP" if stops == 0 else f"{stops}-STOP"
            with rc1:
                st.markdown(f'<div class="rec-card"><div class="rec-title">Recommended</div>'
                            f'<div class="rec-value" style="color:#27F4D2">{stop_label}</div>'
                            f'<div class="rec-sub">{best_strat}</div></div>', unsafe_allow_html=True)
            with rc2:
                pit_window = f"L{best_data['pit_laps'][0]}±3" if best_data["pit_laps"] else "N/A"
                st.markdown(f'<div class="rec-card"><div class="rec-title">Pit Window</div>'
                            f'<div class="rec-value" style="color:#FF8000">{pit_window}</div>'
                            f'<div class="rec-sub">Optimal lap to box</div></div>', unsafe_allow_html=True)
            with rc3:
                next_c = best_data["compounds"][0] if best_data["compounds"] else current_compound
                cc = COMPOUND_COLORS.get(next_c, "#888")
                st.markdown(f'<div class="rec-card"><div class="rec-title">Next Compound</div>'
                            f'<div class="rec-value" style="color:{cc}">{next_c}</div>'
                            f'<div class="rec-sub">Based on deg rates</div></div>', unsafe_allow_html=True)
            with rc4:
                if undercut_possible:
                    st.markdown('<div class="rec-card"><div class="rec-title">Undercut</div>'
                                '<div class="rec-value green">VIABLE</div></div>', unsafe_allow_html=True)
                elif overcut_possible:
                    st.markdown('<div class="rec-card"><div class="rec-title">Overcut</div>'
                                '<div class="rec-value yellow">VIABLE</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="rec-card"><div class="rec-title">Cover/Threat</div>'
                                '<div class="rec-value" style="color:#444">NONE</div></div>', unsafe_allow_html=True)

            # Monte Carlo
            mc_results = {}
            if run_mc:
                with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
                    mc_results = monte_carlo_simulation(
                        model, current_lap_strat, total_laps_strat, current_compound,
                        current_tyre_life, current_stint, strategies_to_sim, laps_df,
                        n_simulations=n_sims, sc_prob_per_lap=sc_prob,
                        traffic_active=traffic_active, traffic_loss=traffic_loss,
                        traffic_laps_ahead=traffic_laps_ahead if traffic_active else 0,
                        track_temp=track_temp, air_temp=air_temp,
                        driver=sel_driver.strip().upper(), team=sel_team.strip())
                if mc_results:
                    all_mc = np.stack(list(mc_results.values()), axis=1)
                    win_counts = [(np.argmin(all_mc, axis=1)==i).sum() for i in range(len(mc_results))]
                    mc_best = list(mc_results.keys())[np.argmax(win_counts)]
                    mc_pct  = max(win_counts) / n_sims * 100
                    if mc_best != best_strat:
                        st.warning(f"⚠️ MC disagrees: recommends **{mc_best}** "
                                   f"({mc_pct:.0f}% win rate) vs deterministic **{best_strat}**.")

            # Tabs
            stab_labels = ["📊 Strategy Comparison","📉 Tyre Degradation",
                           "📈 Lap Time History","🔢 Full Data"]
            if run_mc and mc_results: stab_labels.insert(1, "🎲 Monte Carlo")
            stabs = st.tabs(stab_labels)
            si = 0
            with stabs[si]: si+=1
            with stabs[si-1]:
                st.plotly_chart(chart_strategy_comparison(strategy_results, current_lap_strat, total_laps_strat),
                                use_container_width=True, key="s_strat_comp")
                rows = []
                for name, data in strategy_results.items():
                    delta = data["total_time"] - strategy_results[best_strat]["total_time"]
                    rows.append({"✓":"✓" if name==best_strat else "","Strategy":name,
                                 "Pit Laps":", ".join(str(l) for l in data["pit_laps"]) or "—",
                                 "Compounds":" → ".join(data["compounds"]) or "Stay",
                                 "Total":f"{data['total_time']:.1f}s",
                                 "vs Best":f"+{delta:.1f}s" if delta>0 else "BEST"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            if run_mc and mc_results:
                with stabs[si]: si+=1
                with stabs[si-1]:
                    fig_d, fig_w = chart_monte_carlo(mc_results, best_strat)
                    c1, c2 = st.columns([3,2])
                    with c1: st.plotly_chart(fig_d, use_container_width=True, key="s_mc_d")
                    with c2: st.plotly_chart(fig_w, use_container_width=True, key="s_mc_w")
                    st.dataframe(mc_confidence_table(mc_results), hide_index=True, use_container_width=True)

            with stabs[si]: si+=1
            with stabs[si-1]:
                dc1, dc2 = st.columns([3,2])
                with dc1: st.plotly_chart(chart_degradation(laps_df, model),
                                          use_container_width=True, key="s_deg")
                with dc2: st.plotly_chart(chart_feature_importance(model),
                                          use_container_width=True, key="s_feat")

            with stabs[si]: si+=1
            with stabs[si-1]:
                st.plotly_chart(chart_lap_times_history(laps_df, circuit),
                                use_container_width=True, key="s_hist")

            with stabs[si]:
                st.markdown(f"**{len(laps_df):,} laps from {len(years_to_load)} season(s)**")
                st.dataframe(laps_df.head(200), hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 2 — RACE REPLAY
# ═══════════════════════════════════════════════════════════════
with main_tab_replay:
    if rp_race_data is None:
        st.info("Select a race in the sidebar to begin replay.")
    else:
        rp_laps      = rp_race_data["laps"]
        rp_results   = rp_race_data["results"]
        rp_weather   = rp_race_data["weather"]
        rp_total     = rp_race_data["total_laps"]
        rp_model     = st.session_state.get("rp_model")
        rp_recs      = st.session_state.get("rp_recs", {})
        highlighted  = focus_drivers if focus_drivers else None
        play_delay   = 1.0 / play_speed

        # Metrics
        cols = st.columns(6)
        for col, (label, val, cls) in zip(cols, [
            ("Race",    rp_race_name,                             "red"),
            ("Season",  str(rp_year),                            ""),
            ("Laps",    str(rp_total),                           ""),
            ("Drivers", str(len(all_drivers_rp)),                "green"),
            ("Track °C",f"{rp_weather.get('TrackTemp',0):.0f}°" if rp_weather else "—", "orange"),
            ("Rainfall","YES" if rp_weather.get("Rainfall") else "NO",
                        "red" if rp_weather.get("Rainfall") else "green"),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                            f'<div class="metric-value {cls}">{val}</div></div>',
                            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Precompute events + accuracy
        race_events   = build_race_events(rp_laps, rp_total)
        accuracy_data = []
        for drv in all_drivers_rp:
            stints     = extract_stints(rp_laps, drv)
            drv_ml_rec = rp_recs.get(drv, {})
            score      = evaluate_strategy_accuracy(stints, drv_ml_rec, rp_total)
            rec_lap1   = drv_ml_rec.get(1, {})
            accuracy_data.append({
                "Driver":  drv, "Score": score, "Stints": len(stints),
                "ML Rec":  rec_lap1.get("rec","No model"),
                "ML Pits": ", ".join(str(l) for l in rec_lap1.get("pit_laps",[])) or "—",
            })

        # ── Playback controls ─────────────────────────────────
        c1, c2, c3, c4 = st.columns([1,1,1,4])
        with c1: play_btn  = st.button("▶ Play",  use_container_width=True, key="rp_play")
        with c2: pause_btn = st.button("⏸ Pause", use_container_width=True, key="rp_pause")
        with c3: reset_btn = st.button("⏮ Reset", use_container_width=True, key="rp_reset")
        with c4: manual_lap = st.slider("Jump to lap", 1, rp_total, 1, key="rp_manual")

        if "rp_current_lap" not in st.session_state: st.session_state.rp_current_lap = 1
        if "rp_playing"     not in st.session_state: st.session_state.rp_playing     = False

        if play_btn:  st.session_state.rp_playing = True
        if pause_btn: st.session_state.rp_playing = False
        if reset_btn:
            st.session_state.rp_current_lap = 1
            st.session_state.rp_playing     = False
            st.rerun()
        if not st.session_state.rp_playing:
            st.session_state.rp_current_lap = manual_lap

        current_rp_lap = st.session_state.rp_current_lap
        grid_sim       = st.session_state.get("rp_grid", {})

        st.markdown(f"### 🔴 LAP {current_rp_lap} / {rp_total}")

        # ── PRIMARY: Predicted vs Actual positions side by side ──
        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.markdown("#### 🤖 ML Predicted Positions")
            if grid_sim:
                # Build predicted position traces from grid_sim up to current lap
                fig_pred = go.Figure()
                # Collect all drivers from grid sim
                all_sim_drivers = set()
                for lap_data in grid_sim.values():
                    for car in lap_data:
                        all_sim_drivers.add(car["driver"])

                for drv in sorted(all_sim_drivers):
                    laps_x, pos_y, pits_x, pits_y = [], [], [], []
                    for lap in range(1, current_rp_lap + 1):
                        lap_data = grid_sim.get(lap, [])
                        car = next((c for c in lap_data if c["driver"] == drv), None)
                        if car:
                            laps_x.append(lap)
                            pos_y.append(car["predicted_pos"])
                            if car["pit_this_lap"]:
                                pits_x.append(lap)
                                pits_y.append(car["predicted_pos"])

                    color   = team_color(next((c["team"] for c in grid_sim.get(1,[])
                                               if c["driver"]==drv), ""))
                    width   = 3 if (highlighted and drv in highlighted) else 1.5
                    opacity = 1.0 if (not highlighted or drv in highlighted) else 0.2

                    fig_pred.add_trace(go.Scatter(
                        x=laps_x, y=pos_y, mode="lines", name=drv,
                        line=dict(color=color, width=width), opacity=opacity,
                        hovertemplate=f"<b>{drv}</b> L%{{x}} → P%{{y}}<extra></extra>"))
                    if pits_x:
                        fig_pred.add_trace(go.Scatter(
                            x=pits_x, y=pits_y, mode="markers", name=f"{drv} pit",
                            marker=dict(color=color, size=9, symbol="triangle-down",
                                        line=dict(color="#fff", width=1)),
                            showlegend=False,
                            hovertemplate=f"<b>{drv}</b> PIT L%{{x}}<extra></extra>"))

                fig_pred.add_vline(x=current_rp_lap, line_color="#e8002d",
                                   line_dash="dash",
                                   annotation_text=f"L{current_rp_lap}",
                                   annotation_font_color="#e8002d")
                fig_pred.update_layout(
                    **LAYOUT_BASE,
                    xaxis=dict(title="Lap", range=[1, rp_total], **AXIS),
                    yaxis=dict(title="Position", autorange="reversed", dtick=1, **AXIS),
                    hovermode="x unified", height=450, showlegend=True)
                st.plotly_chart(fig_pred, use_container_width=True,
                                key=f"rp_pred_{current_rp_lap}")

                # Undercut events this lap
                cur_lap_grid = grid_sim.get(current_rp_lap, [])
                undercuts = [c for c in cur_lap_grid if c.get("undercut_flag")]
                if undercuts:
                    for uc in undercuts:
                        st.markdown(
                            f'<div class="event-card event-good">'
                            f'⚡ Undercut: <b>{uc["driver"]}</b> pitted and rejoined ahead</div>',
                            unsafe_allow_html=True)
            else:
                st.info("Train the Replay ML model to see predicted positions")
                st.plotly_chart(
                    chart_positions(rp_laps, current_rp_lap, highlighted),
                    use_container_width=True, key=f"rp_pos_fb_{current_rp_lap}")

        with graph_col2:
            st.markdown("#### 🏁 Actual Positions")
            st.plotly_chart(
                chart_positions(rp_laps, current_rp_lap, highlighted),
                use_container_width=True, key=f"rp_pos_{current_rp_lap}")

        # ── Divergence callout: where do pred and actual disagree? ──
        if grid_sim:
            cur_pred = {c["driver"]: c["predicted_pos"] for c in grid_sim.get(current_rp_lap, [])}
            diverged = []
            for _, row in rp_laps[rp_laps["LapNumber"]==current_rp_lap].iterrows():
                drv      = row["Driver"]
                act_pos  = row.get("Position")
                pred_pos = cur_pred.get(drv)
                if pred_pos and pd.notna(act_pos):
                    delta = int(act_pos) - pred_pos
                    if abs(delta) >= 2:
                        diverged.append((drv, pred_pos, int(act_pos), delta))
            if diverged:
                diverged.sort(key=lambda x: abs(x[3]), reverse=True)
                parts = []
                for drv, pred, act, delta in diverged[:4]:
                    arrow = f"▲{abs(delta)}" if delta < 0 else f"▼{abs(delta)}"
                    color = "#27F4D2" if delta < 0 else "#e8002d"
                    parts.append(f'<span style="color:{color}">{drv} P{pred}→P{act} {arrow}</span>')
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>Model divergence at L{current_rp_lap}:</b> '
                    + " &nbsp;|&nbsp; ".join(parts) + "</div>",
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Lap times + tyre strategies ───────────────────────
        lt_col, tyr_col = st.columns(2)
        with lt_col:
            st.markdown("#### ⏱ Lap Times")
            st.plotly_chart(
                chart_lap_times_replay(rp_laps, current_rp_lap, highlighted),
                use_container_width=True, key=f"rp_lt_{current_rp_lap}")
        with tyr_col:
            st.markdown("#### 🏁 Tyre Strategies")
            st.plotly_chart(
                chart_tyre_strategies_replay(rp_laps, rp_total, current_rp_lap),
                use_container_width=True, key=f"rp_tyr_{current_rp_lap}")

        # ── Leaderboard + events + ML verdict ─────────────────
        lb_col, ev_col, ml_col = st.columns([1, 1, 1])

        with lb_col:
            st.markdown("#### 🏆 Actual Leaderboard")
            st.markdown(leaderboard_html(rp_laps, current_rp_lap), unsafe_allow_html=True)

            if grid_sim:
                st.markdown("#### 🤖 Predicted Leaderboard")
                cur_lap_grid = sorted(grid_sim.get(current_rp_lap, []),
                                      key=lambda c: c["predicted_pos"])
                rows_html = ""
                for car in cur_lap_grid[:10]:
                    drv   = car["driver"]
                    pos   = car["predicted_pos"]
                    comp  = car["compound"]
                    gap   = car["gap_to_leader"]
                    color = team_color(car.get("team",""))
                    comp_c = COMPOUND_COLORS.get(comp,"#888")
                    gap_str = "LEADER" if gap == 0 else f"+{gap:.1f}s"
                    rows_html += (
                        f'<div class="driver-row" style="background:#0f0f0f">'
                        f'<span style="color:#555;width:22px;text-align:right">{pos}</span>'
                        f'<span style="color:{color};width:36px">{drv}</span>'
                        f'<span style="color:{comp_c};font-size:11px;width:18px">'
                        f'{COMPOUND_SHORT.get(comp,"?")}</span>'
                        f'<span style="color:#555;font-size:11px">{gap_str}</span>'
                        f'</div>')
                st.markdown(rows_html, unsafe_allow_html=True)

        with ev_col:
            st.markdown("#### 📡 Events")
            events_html = ""
            for look_back in range(min(4, current_rp_lap), 0, -1):
                lap_n = current_rp_lap - look_back + 1
                for ev in race_events.get(lap_n, []):
                    cls = {"pit":"event-info","gain":"event-good",
                           "loss":"event-bad","sc":"event-warn"}.get(ev["type"],"event-info")
                    events_html += (f'<div class="event-card {cls}">'
                                    f'L{lap_n} {ev["text"]}</div>')
            if not events_html:
                events_html = '<div class="event-card event-info">No events this lap</div>'
            st.markdown(events_html, unsafe_allow_html=True)

        with ml_col:
            st.markdown("#### 🤖 Strategy Verdict")
            if not rp_recs:
                st.caption("Train the Replay ML model in the sidebar")
            elif not focus_drivers:
                st.caption("Select drivers in sidebar to see per-driver verdict")
            else:
                cur_lap_data = rp_laps[rp_laps["LapNumber"] == current_rp_lap]
                for drv in focus_drivers[:5]:
                    drv_data = cur_lap_data[cur_lap_data["Driver"] == drv]
                    if drv_data.empty: continue
                    row       = drv_data.iloc[0]
                    comp      = str(row.get("Compound","?")).upper()
                    tyre_life = int(row.get("TyreLife",0)) if pd.notna(row.get("TyreLife",0)) else 0
                    comp_c    = COMPOUND_COLORS.get(comp,"#888")
                    drv_rec   = rp_recs.get(drv,{}).get(current_rp_lap,{})
                    rec_name  = drv_rec.get("rec","—")
                    pit_laps  = drv_rec.get("pit_laps",[])
                    rec_comps = drv_rec.get("compounds",[])
                    actual_stints   = extract_stints(rp_laps, drv)
                    actual_pit_laps = [s["start"] for s in actual_stints[1:]]
                    if pit_laps:
                        agreed = any(abs(pl-al)<=3 for pl in pit_laps for al in actual_pit_laps)
                    else:
                        agreed = len(actual_pit_laps) == 0
                    verdict_c = "#27F4D2" if agreed else "#E8002D"
                    verdict   = "✅ AGREED" if agreed else "❌ DIVERGED"
                    pit_str   = f"pit L{'/'.join(str(l) for l in pit_laps)}" if pit_laps else "stay out"
                    # Also show predicted vs actual position
                    pred_pos = cur_pred.get(drv,"?") if grid_sim else "?"
                    act_pos  = row.get("Position","?")
                    pos_str  = (f"P{pred_pos}→P{int(act_pos)}" if pred_pos!="?" and
                                pd.notna(act_pos) else "")
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:6px;border-left-color:{verdict_c}">'
                        f'<div class="metric-label">{drv} — '
                        f'<span style="color:{comp_c}">{comp}</span>'
                        f' {tyre_life}L {pos_str}</div>'
                        f'<div class="metric-value" style="color:#fff;font-size:13px">{rec_name}</div>'
                        f'<div style="font-size:11px;color:#666">{pit_str} {"→".join(rec_comps)}</div>'
                        f'<div style="font-size:11px;color:{verdict_c};letter-spacing:1px">{verdict}</div>'
                        f'</div>', unsafe_allow_html=True)

        # ── Strategy accuracy expander ─────────────────────────
        with st.expander("📊 ML Strategy Accuracy vs Actual Race", expanded=False):
            st.plotly_chart(chart_strategy_accuracy_replay(accuracy_data),
                            use_container_width=True, key="rp_acc")
            if not rp_recs:
                st.caption("Train the Replay ML model in the sidebar to see real accuracy.")
            st.dataframe(pd.DataFrame(accuracy_data), hide_index=True, use_container_width=True)

        # ── Autoplay ──────────────────────────────────────────
        if st.session_state.rp_playing:
            if st.session_state.rp_current_lap < rp_total:
                time.sleep(play_delay)
                st.session_state.rp_current_lap += 1
                st.rerun()
            else:
                st.session_state.rp_playing = False
                st.success("🏁 Race finished!")
