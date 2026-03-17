"""
f1_research.py — Full Evaluation Pipeline
==========================================
Paper title:
  "Beyond Accuracy: Cross-Season Generalisation and Strategic Error Thresholds
   in F1 Tyre Degradation Prediction"

Three experiments:
  1. Within-season CV         — 5-fold, 6 models, MAE/RMSE/R² → Table 1
  2. Cross-season generalisation — rolling temporal splits → Table 2 + Figure 4
  3. Strategic sensitivity threshold — noise injection σ sweep → Table 3 + Figure 5

SETUP:
    pip install fastf1 scikit-learn xgboost pandas numpy scipy tqdm matplotlib seaborn

RUN:
    cd ~/Downloads/F1App && python f1_research.py

OUTPUT:
    results/
      metrics.csv            — Exp 1: MAE/RMSE/R² per model per circuit per fold
      summary.csv            — Exp 1: mean ± std per model (Table 1)
      cross_season.csv       — Exp 2: generalisation MAE per split per model (Table 2)
      threshold.csv          — Exp 3: strategy flip rate per σ level per circuit (Table 3)
      significance.csv       — paired t-test results
      sensitivity.csv        — binary + time-delta sensitivity per model per circuit
      laps_<circuit>.csv     — raw featurised lap data (reproducibility)
      figures/
        fig1_mae_comparison.png
        fig2_mae_violin.png
        fig3_mae_heatmap.png
        fig4_cross_season.png
        fig5_threshold_curve.png
        fig6_degradation_curves.png
        fig7_strategy_sensitivity.png
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import fastf1
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed — skipping. Run: pip install xgboost")

# ── Paths ─────────────────────────────────────────────────────
CACHE_DIR   = Path.home() / "f1_cache"
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
for d in [CACHE_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Configuration ─────────────────────────────────────────────
CIRCUITS = [
    "Bahrain Grand Prix",
    "Spanish Grand Prix",
    "British Grand Prix",
    "Italian Grand Prix",
    "Monaco Grand Prix",
    "Hungarian Grand Prix",
]
SEASONS           = [2022, 2023, 2024]
N_FOLDS           = 5
RANDOM_STATE      = 42
COMPOUNDS         = ["SOFT", "MEDIUM", "HARD"]
PIT_LOSS_S        = 22.0
RACE_SNAPSHOTS    = [0.25, 0.35, 0.45, 0.55, 0.65]
NOISE_SIGMAS      = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
N_NOISE_SIMS      = 30

NUMERIC_FEATURES = [
    "TyreLife", "CompoundEnc", "RaceLapPct", "Stint",
    "FuelLoad", "IsWarmup", "TrackTemp", "AirTemp",
    "CircuitThrottlePct", "CircuitBrakePct", "CircuitAvgSpeed",
]
CATEGORICAL_FEATURES = ["Driver", "Team"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PALETTE = {
    "Naive (compound mean)": "#888780",
    "Linear Regression":     "#4C72B0",
    "Ridge Regression":      "#64B5CD",
    "Random Forest":         "#55A868",
    "Gradient Boosting":     "#E8002D",
    "XGBoost":               "#FF8000",
}


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — MODELS
# ═══════════════════════════════════════════════════════════════

def get_models():
    models = {
        "Naive (compound mean)": None,
        "Linear Regression":     LinearRegression(),
        "Ridge Regression":      Ridge(alpha=1.0),
        "Random Forest":         RandomForestRegressor(
                                     n_estimators=200, max_depth=10,
                                     min_samples_leaf=5, n_jobs=1,
                                     random_state=RANDOM_STATE),
        "Gradient Boosting":     GradientBoostingRegressor(
                                     n_estimators=300, max_depth=5,
                                     learning_rate=0.04, min_samples_leaf=5,
                                     subsample=0.85, random_state=RANDOM_STATE),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            min_child_weight=5, subsample=0.85, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0)
    return models


def build_pipeline(estimator):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ], remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def clone_estimator(estimator):
    return estimator.__class__(**estimator.get_params())


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_race(year, circuit_name):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        keyword  = circuit_name.replace(" Grand Prix", "").strip()
        matches  = schedule[schedule["EventName"].str.contains(
                       keyword, case=False, na=False)]
        if matches.empty:
            return None
        round_num = int(matches.iloc[0]["RoundNumber"])

        sess = fastf1.get_session(year, round_num, "R")
        sess.load(laps=True, telemetry=True, weather=True, messages=False)
        laps = sess.laps.copy()

        laps = laps[laps["LapTime"].notna()].copy()
        laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
        laps = laps[
            (laps["LapTimeSec"] > 60) &
            (laps["LapTimeSec"] < 200) &
            (laps["Compound"].isin(COMPOUNDS)) &
            (laps["PitInTime"].isna()) &
            (laps["PitOutTime"].isna())
        ]
        if len(laps) < 50:
            return None

        total_laps         = int(laps["LapNumber"].max())
        laps["TotalLaps"]  = total_laps
        laps["RaceLapPct"] = laps["LapNumber"] / total_laps
        laps["FuelLoad"]   = (110.0 - laps["LapNumber"] * 1.5).clip(lower=0)
        laps["IsWarmup"]   = (laps["TyreLife"] <= 2).astype(int)
        laps["CompoundEnc"]= laps["Compound"].map({"SOFT":0,"MEDIUM":1,"HARD":2})

        if not sess.weather_data.empty:
            wx = (sess.weather_data[["Time","TrackTemp","AirTemp"]]
                  .copy().sort_values("Time"))
            # Drop existing weather cols to avoid duplicates after merge
            laps = laps.drop(columns=["TrackTemp","AirTemp"], errors="ignore")
            laps = laps.sort_values("LapStartTime")
            laps = pd.merge_asof(
                laps,
                wx.rename(columns={"Time":"LapStartTime"}),
                on="LapStartTime", direction="nearest")
        else:
            laps["TrackTemp"] = 35.0
            laps["AirTemp"]   = 22.0

        throttle_pcts, brake_pcts, avg_speeds = [], [], []
        sampled = (laps.groupby("Driver")
                       .apply(lambda g: g.nsmallest(1, "LapTimeSec"))
                       .reset_index(drop=True))
        for _, row in sampled.head(5).iterrows():
            try:
                lap_row = sess.laps[
                    (sess.laps["LapNumber"] == row["LapNumber"]) &
                    (sess.laps["Driver"]    == row["Driver"])
                ]
                if lap_row.empty: continue
                tel = lap_row.iloc[0].get_car_data()
                if tel.empty: continue
                throttle_pcts.append((tel["Throttle"] > 80).mean() * 100)
                brake_pcts.append(tel["Brake"].astype(float).mean() * 100)
                avg_speeds.append(tel["Speed"].mean())
            except Exception:
                continue

        laps["CircuitThrottlePct"] = float(np.mean(throttle_pcts)) if throttle_pcts else 65.0
        laps["CircuitBrakePct"]    = float(np.mean(brake_pcts))    if brake_pcts    else 12.0
        laps["CircuitAvgSpeed"]    = float(np.mean(avg_speeds))    if avg_speeds    else 210.0
        laps["Circuit"]            = circuit_name
        laps["Year"]               = year

        # Deduplicate columns — FastF1 sometimes produces duplicates after merges
        laps = laps.loc[:, ~laps.columns.duplicated()]

        keep = ALL_FEATURES + ["LapTimeSec","Compound","Circuit","Year",
                                "LapNumber","TyreLife"]
        laps = (laps[[c for c in keep if c in laps.columns]]
                .dropna(subset=ALL_FEATURES + ["LapTimeSec"]))
        return laps

    except Exception as e:
        print(f"    Failed {circuit_name} {year}: {e}")
        return None


def circuit_defaults_from(df):
    return {col: float(df[col].median())
            for col in ["CircuitThrottlePct","CircuitBrakePct","CircuitAvgSpeed",
                        "TrackTemp","AirTemp"]}


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — NAIVE BASELINE
# ═══════════════════════════════════════════════════════════════

def naive_predict(train_df, test_df):
    medians = train_df.groupby("Compound")["LapTimeSec"].median().to_dict()
    overall = train_df["LapTimeSec"].median()
    return test_df["Compound"].map(medians).fillna(overall).values


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — STRATEGY SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_strategy_time(model, circuit_def, total_laps, current_lap,
                            compound, tyre_life, stint,
                            pit_laps, next_compounds,
                            track_temp, air_temp,
                            noise_sigma=0.0, rng=None):
    remaining  = total_laps - current_lap
    pit_queue  = list(zip(pit_laps, next_compounds))
    t_life     = tyre_life
    comp       = compound
    st         = stint
    start_fuel = max(0.0, 110.0 - current_lap * 1.5)
    total_time = 0.0
    cd         = circuit_def

    for offset in range(remaining):
        lap = current_lap + offset
        if pit_queue and pit_queue[0][0] == lap:
            total_time += PIT_LOSS_S
            _, comp = pit_queue.pop(0)
            t_life  = 1
            st     += 1
        else:
            t_life += 1

        fuel_load = max(0.0, start_fuel - offset * 1.5)
        row = pd.DataFrame([{
            "TyreLife":           t_life,
            "CompoundEnc":        {"SOFT":0,"MEDIUM":1,"HARD":2}.get(comp, 1),
            "RaceLapPct":         lap / total_laps,
            "Stint":              st,
            "FuelLoad":           fuel_load,
            "IsWarmup":           1 if t_life <= 2 else 0,
            "TrackTemp":          track_temp,
            "AirTemp":            air_temp,
            "CircuitThrottlePct": cd.get("CircuitThrottlePct", 65.0),
            "CircuitBrakePct":    cd.get("CircuitBrakePct",    12.0),
            "CircuitAvgSpeed":    cd.get("CircuitAvgSpeed",   210.0),
            "Driver":             "UNKNOWN",
            "Team":               "UNKNOWN",
        }])
        try:
            lt = model.predict(row)[0]
        except Exception:
            lt = 90.0

        if noise_sigma > 0 and rng is not None:
            lt += rng.normal(0, noise_sigma)
        total_time += lt

    return total_time


def build_strategies(current_lap, total_laps, compound):
    remaining = total_laps - current_lap
    mid       = current_lap + remaining // 2
    early     = current_lap + int(remaining * 0.35)
    late      = current_lap + int(remaining * 0.65)
    nxt       = [c for c in COMPOUNDS if c != compound]
    nc1       = nxt[0] if nxt        else "HARD"
    nc2       = nxt[1] if len(nxt)>1 else "MEDIUM"
    return [
        ("Stay Out",     [],            []),
        ("Early 1-Stop", [early],       [nc1]),
        ("Mid 1-Stop",   [mid],         [nc1]),
        ("Late 1-Stop",  [late],        [nc1]),
        ("2-Stop",       [early,mid+5], [nc1,nc2]),
    ]


def best_strategy(model, circuit_def, total_laps, current_lap,
                  compound, tyre_life, track_temp, air_temp,
                  noise_sigma=0.0, rng=None):
    strategies = build_strategies(current_lap, total_laps, compound)
    results    = []
    for name, pit_laps, next_comps in strategies:
        t = simulate_strategy_time(
            model, circuit_def, total_laps, current_lap,
            compound, tyre_life, 1,
            pit_laps, next_comps, track_temp, air_temp,
            noise_sigma=noise_sigma, rng=rng)
        results.append((name, t))
    results.sort(key=lambda x: x[1])
    best_name   = results[0][0]
    best_time   = results[0][1]
    second_time = results[1][1] if len(results) > 1 else best_time
    return best_name, best_time, second_time


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — EXPERIMENT 1: WITHIN-SEASON CV
# ═══════════════════════════════════════════════════════════════

def run_exp1_cv(all_data):
    print("\n\nEXPERIMENT 1 — Within-season 5-fold CV")
    print("="*50)

    models_dict  = get_models()
    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics_rows = []
    sens_rows    = []

    for circuit, circuit_df in all_data.items():
        print(f"\n  {circuit}  ({len(circuit_df)} laps)")
        print(f"  {'Model':<25} {'MAE':>7} {'RMSE':>7} {'R2':>7}")
        print(f"  {'-'*48}")

        circuit_df = circuit_df.loc[:, ~circuit_df.columns.duplicated()].copy()
        X         = circuit_df[ALL_FEATURES + ["Compound"]].copy()
        y         = circuit_df["LapTimeSec"].values
        cd        = circuit_defaults_from(circuit_df)
        fold_recs = {name: [] for name in models_dict}

        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
            y_tr, y_te = y[tr_idx], y[te_idx]

            for model_name, estimator in models_dict.items():
                if estimator is None:
                    preds   = naive_predict(X_tr.assign(LapTimeSec=y_tr), X_te)
                    trained = None
                else:
                    pipe = build_pipeline(clone_estimator(estimator))
                    pipe.fit(X_tr[ALL_FEATURES], y_tr)
                    preds   = pipe.predict(X_te[ALL_FEATURES])
                    trained = pipe

                mae  = mean_absolute_error(y_te, preds)
                rmse = np.sqrt(mean_squared_error(y_te, preds))
                r2   = r2_score(y_te, preds)

                metrics_rows.append({
                    "Circuit": circuit, "Model": model_name, "Fold": fold_idx+1,
                    "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4),
                    "N_train": len(tr_idx), "N_test": len(te_idx),
                })

                if trained is not None:
                    try:
                        total_laps = int(circuit_df["LapNumber"].max())
                        snap_lap   = int(total_laps * 0.4)
                        rec, _, _  = best_strategy(
                            trained, cd, total_laps, snap_lap,
                            "MEDIUM", 12,
                            cd.get("TrackTemp", 35.0),
                            cd.get("AirTemp",   22.0))
                        fold_recs[model_name].append(rec)
                    except Exception:
                        fold_recs[model_name].append("Error")

            if fold_idx == 0:
                for row in metrics_rows[-len(models_dict):]:
                    print(f"  {row['Model']:<25} "
                          f"{row['MAE']:>7.4f} {row['RMSE']:>7.4f} {row['R2']:>7.4f}")

        all_recs = {n: r for n, r in fold_recs.items() if r}
        if all_recs:
            modal_recs = {n: max(set(r), key=r.count) for n, r in all_recs.items()}
            sens_rows.append({
                "Circuit":          circuit,
                "All_Models_Agree": len(set(modal_recs.values())) == 1,
                **{f"Rec_{n.replace(' ','_')}": v
                   for n, v in modal_recs.items()},
            })

    metrics_df = pd.DataFrame(metrics_rows)
    sens_df    = pd.DataFrame(sens_rows)

    summary = (metrics_df.groupby("Model")[["MAE","RMSE","R2"]]
               .agg(["mean","std"]).round(4))
    summary.columns = ["MAE_mean","MAE_std","RMSE_mean","RMSE_std","R2_mean","R2_std"]
    summary = summary.sort_values("MAE_mean")

    print("\n\n  Overall ranking (mean ± std):")
    print(f"  {'Model':<25} {'MAE mean±std':>18} {'R2 mean':>10}")
    print(f"  {'-'*56}")
    for name, row in summary.iterrows():
        print(f"  {name:<25} "
              f"{row['MAE_mean']:.4f}±{row['MAE_std']:.4f}  "
              f"{row['R2_mean']:>10.4f}")

    print("\n  Significance tests (paired t-test vs GBM):")
    gbm_maes = metrics_df[metrics_df["Model"]=="Gradient Boosting"]["MAE"].values
    sig_rows = []
    for mname in metrics_df["Model"].unique():
        if mname == "Gradient Boosting": continue
        other = metrics_df[metrics_df["Model"]==mname]["MAE"].values
        n     = min(len(gbm_maes), len(other))
        if n < 2: continue
        t, p  = stats.ttest_rel(gbm_maes[:n], other[:n])
        mark  = "sig" if p < 0.05 else "ns"
        print(f"  GBM vs {mname:<25} p={p:.4f} ({mark})")
        sig_rows.append({"Comparison": f"GBM vs {mname}",
                         "t_stat": round(t,4), "p_value": round(p,4),
                         "Significant": p < 0.05})

    return metrics_df, summary, pd.DataFrame(sig_rows), sens_df


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — EXPERIMENT 2: CROSS-SEASON GENERALISATION
# ═══════════════════════════════════════════════════════════════

def run_exp2_cross_season(all_data_by_year):
    print("\n\nEXPERIMENT 2 — Cross-season generalisation")
    print("="*50)

    models_dict = get_models()
    cs_rows     = []
    splits      = [
        ("2022->2023",      [2022],      [2023]),
        ("2023->2024",      [2023],      [2024]),
        ("2022+2023->2024", [2022,2023], [2024]),
    ]

    for circuit in all_data_by_year:
        year_data = all_data_by_year[circuit]
        print(f"\n  {circuit}")

        for split_name, train_years, test_years in splits:
            tr_frames = [year_data[y] for y in train_years if y in year_data]
            te_frames = [year_data[y] for y in test_years  if y in year_data]
            if not tr_frames or not te_frames:
                continue

            tr_df = pd.concat(tr_frames, ignore_index=True)
            te_df = pd.concat(te_frames, ignore_index=True)
            tr_df = tr_df.loc[:, ~tr_df.columns.duplicated()].copy()
            te_df = te_df.loc[:, ~te_df.columns.duplicated()].copy()

            for model_name, estimator in models_dict.items():
                if estimator is None:
                    preds = naive_predict(tr_df, te_df)
                else:
                    pipe = build_pipeline(clone_estimator(estimator))
                    pipe.fit(tr_df[ALL_FEATURES], tr_df["LapTimeSec"])
                    preds = pipe.predict(te_df[ALL_FEATURES])

                mae  = mean_absolute_error(te_df["LapTimeSec"], preds)
                rmse = np.sqrt(mean_squared_error(te_df["LapTimeSec"], preds))
                r2   = r2_score(te_df["LapTimeSec"], preds)

                cs_rows.append({
                    "Circuit":    circuit,
                    "Split":      split_name,
                    "Train_years":str(train_years),
                    "Test_year":  str(test_years),
                    "Model":      model_name,
                    "MAE":        round(mae, 4),
                    "RMSE":       round(rmse,4),
                    "R2":         round(r2,  4),
                    "N_train":    len(tr_df),
                    "N_test":     len(te_df),
                })

            print(f"    {split_name}")
            for r in cs_rows[-len(models_dict):]:
                print(f"      {r['Model']:<25} MAE={r['MAE']:.4f}")

    return pd.DataFrame(cs_rows)


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — EXPERIMENT 3: STRATEGIC SENSITIVITY THRESHOLD
# ═══════════════════════════════════════════════════════════════

def run_exp3_threshold(all_data):
    """
    Noise injection to find the strategic error threshold per circuit.
    Trains GBM on full circuit data, then sweeps sigma levels and measures
    how often the strategy recommendation flips vs sigma=0 baseline.
    Threshold = lowest sigma where mean flip rate > 50%.
    """
    print("\n\nEXPERIMENT 3 — Strategic sensitivity threshold")
    print("="*50)

    rng         = np.random.default_rng(RANDOM_STATE)
    thresh_rows = []
    models_dict = get_models()

    for circuit, circuit_df in all_data.items():
        circuit_df = circuit_df.loc[:, ~circuit_df.columns.duplicated()].copy()
        print(f"\n  {circuit}")
        total_laps = int(circuit_df["LapNumber"].max())
        cd         = circuit_defaults_from(circuit_df)
        track_temp = cd.get("TrackTemp", 35.0)
        air_temp   = cd.get("AirTemp",   22.0)

        # Train models for Exp 3 — skip Naive and Random Forest (too slow in sim loop)
        EXP3_SKIP = {"Naive (compound mean)", "Random Forest"}
        trained_models = {}
        for model_name, estimator in models_dict.items():
            if model_name in EXP3_SKIP: continue
            pipe = build_pipeline(clone_estimator(estimator))
            pipe.fit(circuit_df[ALL_FEATURES], circuit_df["LapTimeSec"])
            trained_models[model_name] = pipe

        for model_name, trained in trained_models.items():
            # Baseline recommendation at sigma=0 per snapshot
            baselines = {}
            for snap_pct in RACE_SNAPSHOTS:
                snap_lap           = max(1, int(total_laps * snap_pct))
                base_rec, b_t, s_t = best_strategy(
                    trained, cd, total_laps, snap_lap,
                    "MEDIUM", 12, track_temp, air_temp,
                    noise_sigma=0.0, rng=rng)
                baselines[snap_pct] = {
                    "rec":   base_rec,
                    "delta": s_t - b_t,
                }

            # sigma sweep
            for sigma in NOISE_SIGMAS:
                flip_counts = {s: 0   for s in RACE_SNAPSHOTS}
                delta_lists = {s: []  for s in RACE_SNAPSHOTS}

                for _ in range(N_NOISE_SIMS):
                    for snap_pct in RACE_SNAPSHOTS:
                        snap_lap = max(1, int(total_laps * snap_pct))
                        rec, b_t, s_t = best_strategy(
                            trained, cd, total_laps, snap_lap,
                            "MEDIUM", 12, track_temp, air_temp,
                            noise_sigma=sigma, rng=rng)
                        if rec != baselines[snap_pct]["rec"]:
                            flip_counts[snap_pct] += 1
                        delta_lists[snap_pct].append(s_t - b_t)

                for snap_pct in RACE_SNAPSHOTS:
                    thresh_rows.append({
                        "Circuit":           circuit,
                        "Model":             model_name,
                        "Sigma":             sigma,
                        "Snap_pct":          snap_pct,
                        "Snap_lap":          int(total_laps * snap_pct),
                        "Flip_rate":         round(flip_counts[snap_pct] / N_NOISE_SIMS, 4),
                        "Mean_time_delta_s": round(float(np.mean(delta_lists[snap_pct])), 4),
                        "Baseline_rec":      baselines[snap_pct]["rec"],
                        "Baseline_delta_s":  round(baselines[snap_pct]["delta"], 4),
                    })

            # Print threshold for this model
            tmp = pd.DataFrame(thresh_rows).query(
                f"Circuit=='{circuit}' and Model=='{model_name}'")
            if not tmp.empty:
                agg   = tmp.groupby("Sigma")["Flip_rate"].mean().reset_index()
                above = agg[agg["Flip_rate"] > 0.5]
                thr   = float(above["Sigma"].min()) if not above.empty else ">2.0"
                print(f"    {model_name:<25} threshold ~= {thr}s")

    return pd.DataFrame(thresh_rows)


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — FIGURES
# ═══════════════════════════════════════════════════════════════

def generate_figures(metrics_df, summary, cs_df, thresh_df, sens_df, all_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
    })
    model_order = summary.index.tolist()
    colors      = [PALETTE.get(m, "#999") for m in model_order]

    # Fig 1 — MAE bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    maes = summary["MAE_mean"].values
    stds = summary["MAE_std"].values
    bars = ax.barh(model_order, maes, xerr=stds, color=colors,
                   capsize=4, error_kw=dict(ecolor="#444", lw=1.5))
    for bar, mae, std in zip(bars, maes, stds):
        ax.text(mae+std+0.01, bar.get_y()+bar.get_height()/2,
                f"{mae:.3f}s", va="center", fontsize=9)
    ax.set_xlabel("Mean Absolute Error (seconds)")
    ax.set_title(f"Fig 1 — Lap time prediction MAE "
                 f"(mean ± std, {N_FOLDS}-fold CV, all circuits)")
    ax.set_xlim(0, max(maes)*1.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"fig1_mae_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig1_mae_comparison.png")

    # Fig 2 — MAE violin
    fig, ax = plt.subplots(figsize=(11, 5))
    data_by_model = [metrics_df[metrics_df["Model"]==m]["MAE"].values
                     for m in model_order]
    parts = ax.violinplot(data_by_model, showmedians=True)
    for pc, m in zip(parts["bodies"], model_order):
        pc.set_facecolor(PALETTE.get(m,"#999")); pc.set_alpha(0.7)
    parts["cmedians"].set_color("#000")
    ax.set_xticks(range(1, len(model_order)+1))
    ax.set_xticklabels(model_order, rotation=20, ha="right")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("Fig 2 — MAE distribution across folds and circuits")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"fig2_mae_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig2_mae_violin.png")

    # Fig 3 — MAE heatmap
    per_circuit = (metrics_df.groupby(["Circuit","Model"])["MAE"]
                   .mean().round(4).unstack("Model"))
    plot_cols = [m for m in model_order if m in per_circuit.columns]
    fig, ax   = plt.subplots(figsize=(13, 5))
    sns.heatmap(per_circuit[plot_cols].T, annot=True, fmt=".3f",
                cmap="RdYlGn_r", ax=ax, linewidths=0.5,
                cbar_kws={"label":"MAE (s)"})
    ax.set_title("Fig 3 — MAE heatmap: model x circuit")
    plt.xticks(rotation=25, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR/"fig3_mae_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig3_mae_heatmap.png")

    # Fig 4 — Cross-season generalisation
    if not cs_df.empty:
        split_order = ["2022->2023", "2023->2024", "2022+2023->2024"]
        fig, axes   = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, split in zip(axes, split_order):
            sdf = cs_df[cs_df["Split"]==split]
            if sdf.empty: continue
            pivot = (sdf.groupby(["Circuit","Model"])["MAE"]
                     .mean().unstack("Model"))
            for m in [m for m in model_order if m in pivot.columns]:
                ax.plot(pivot.index, pivot[m], "o-",
                        color=PALETTE.get(m,"#999"), label=m,
                        lw=1.8, markersize=5)
            ax.set_title(split.replace("->"," → "), fontsize=10)
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.3)
            if ax == axes[0]:
                ax.set_ylabel("Test MAE (seconds)")
        axes[-1].legend(loc="upper right", fontsize=7, framealpha=0.9)
        fig.suptitle("Fig 4 — Cross-season generalisation MAE per circuit",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR/"fig4_cross_season.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  fig4_cross_season.png")

    # Fig 5 — Threshold curve
    if not thresh_df.empty:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                                 sharey=True, sharex=True)
        axes_flat = axes.flatten()
        non_naive = [m for m in model_order if m != "Naive (compound mean)"]
        for i, circuit in enumerate(CIRCUITS):
            if i >= len(axes_flat): break
            ax  = axes_flat[i]
            cdf = thresh_df[thresh_df["Circuit"]==circuit]
            if cdf.empty: continue
            for m in non_naive:
                mdf = (cdf[cdf["Model"]==m]
                       .groupby("Sigma")["Flip_rate"].mean().reset_index())
                ax.plot(mdf["Sigma"], mdf["Flip_rate"], "o-",
                        color=PALETTE.get(m,"#999"), label=m,
                        lw=1.8, markersize=4)
            ax.axhline(0.5, color="#999", lw=1, ls="--")
            ax.set_title(circuit.replace(" Grand Prix",""), fontsize=10)
            ax.set_ylim(0, 1); ax.grid(alpha=0.25)
            if i >= 3: ax.set_xlabel("Noise sigma (s)")
            if i % 3 == 0: ax.set_ylabel("Strategy flip rate")
        axes_flat[2].legend(loc="lower right", fontsize=7, ncol=2)
        fig.suptitle("Fig 5 — Strategic error threshold: sigma vs flip rate",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR/"fig5_threshold_curve.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("  fig5_threshold_curve.png")

    # Fig 6 — Degradation curves
    if all_data:
        circuit_name = max(all_data, key=lambda c: len(all_data[c]))
        df           = all_data[circuit_name]
        comp_colors  = {"SOFT":"#E8002D","MEDIUM":"#DDAA00","HARD":"#555555"}
        fig, axes    = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, comp in zip(axes, COMPOUNDS):
            cdf = df[df["Compound"]==comp]
            if cdf.empty: ax.set_title(comp); continue
            med = (cdf.groupby("TyreLife")["LapTimeSec"]
                   .agg(["median","std"]).reset_index())
            med = med[med["TyreLife"]<=40]
            c   = comp_colors[comp]
            ax.plot(med["TyreLife"], med["median"], color=c, lw=2.5)
            ax.fill_between(med["TyreLife"],
                            med["median"]-med["std"],
                            med["median"]+med["std"],
                            alpha=0.15, color=c)
            ax.set_title(comp, fontsize=11, color=c, fontweight="bold")
            ax.set_xlabel("Tyre life (laps)")
            if ax == axes[0]: ax.set_ylabel("Lap time (s)")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Fig 6 — Degradation curves: {circuit_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR/"fig6_degradation_curves.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("  fig6_degradation_curves.png")

    # Fig 7 — Strategy sensitivity heatmap
    if not sens_df.empty:
        import matplotlib as mpl
        rec_cols   = [c for c in sens_df.columns if c.startswith("Rec_")]
        mlabels    = [c.replace("Rec_","").replace("_"," ") for c in rec_cols]
        strat_map  = {"Stay Out":0,"Early 1-Stop":1,"Mid 1-Stop":2,
                      "Late 1-Stop":3,"2-Stop":4,"Error":5}
        rows       = []
        for _, row in sens_df.iterrows():
            for col, ml in zip(rec_cols, mlabels):
                rows.append({"Model":ml,"Circuit":row["Circuit"],
                              "Rec":row[col],
                              "Code":strat_map.get(row[col],5)})
        rdf   = pd.DataFrame(rows)
        pivot = rdf.pivot_table(index="Model", columns="Circuit",
                                values="Code", aggfunc="first")
        ann   = rdf.pivot_table(index="Model", columns="Circuit",
                                values="Rec", aggfunc="first")
        fig, ax = plt.subplots(figsize=(13, 5))
        cmap = mpl.colormaps.get_cmap("tab10").resampled(6)
        sns.heatmap(pivot, annot=ann, fmt="", cmap=cmap, ax=ax,
                    linewidths=0.5, cbar=False, vmin=0, vmax=5)
        ax.set_title("Fig 7 — Strategy recommendations: model x circuit")
        plt.xticks(rotation=25, ha="right"); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR/"fig7_strategy_sensitivity.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("  fig7_strategy_sensitivity.png")


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  F1 TYRE DEGRADATION — FULL RESEARCH PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    print(f"  Circuits : {len(CIRCUITS)}")
    print(f"  Seasons  : {SEASONS}")
    print(f"  CV folds : {N_FOLDS}")
    print(f"  Models   : {list(get_models().keys())}")
    print(f"  Output   : {RESULTS_DIR.resolve()}")
    print("="*60)

    # Load data
    print("\n\nSTEP 1 — Loading race data")
    print("-"*40)
    all_data         = {}
    all_data_by_year = {}

    for circuit in CIRCUITS:
        print(f"\n  {circuit}")
        by_year = {}
        frames  = []
        for year in SEASONS:
            print(f"    {year}...", end=" ", flush=True)
            df = load_race(year, circuit)
            if df is not None:
                by_year[year] = df
                frames.append(df)
                print(f"OK ({len(df)} laps)")
            else:
                print("skipped")
        if frames:
            all_data[circuit]         = pd.concat(frames, ignore_index=True)
            all_data_by_year[circuit] = by_year
            print(f"    Total: {len(all_data[circuit])} laps")
        else:
            print(f"    No data — skipping circuit")

    if not all_data:
        print("\nNo data loaded. Check internet connection and FastF1 cache.")
        return

    # Experiments
    metrics_df, summary, sig_df, sens_df = run_exp1_cv(all_data)
    cs_df                                = run_exp2_cross_season(all_data_by_year)
    thresh_df                            = run_exp3_threshold(all_data)

    # Save
    print("\n\nSAVING RESULTS")
    print("-"*40)
    metrics_df.to_csv(RESULTS_DIR/"metrics.csv",     index=False)
    summary.to_csv(   RESULTS_DIR/"summary.csv")
    sig_df.to_csv(    RESULTS_DIR/"significance.csv",index=False)
    sens_df.to_csv(   RESULTS_DIR/"sensitivity.csv", index=False)
    cs_df.to_csv(     RESULTS_DIR/"cross_season.csv",index=False)
    thresh_df.to_csv( RESULTS_DIR/"threshold.csv",   index=False)
    for circuit, df in all_data.items():
        safe = circuit.replace(" ","_").replace("/","_")
        df.to_csv(RESULTS_DIR/f"laps_{safe}.csv", index=False)
    print(f"  All CSVs saved.")

    # Figures
    print("\n\nGENERATING FIGURES")
    print("-"*40)
    try:
        generate_figures(metrics_df, summary, cs_df, thresh_df, sens_df, all_data)
    except Exception as e:
        print(f"  Figures failed: {e}")
        print("  Install: pip install matplotlib seaborn")

    print("\n" + "="*60)
    print("  DONE")
    print(f"  Results → {RESULTS_DIR.resolve()}")
    print(f"  Figures → {FIGURES_DIR.resolve()}")
    print("="*60 + "\n")

    return metrics_df, summary, cs_df, thresh_df


if __name__ == "__main__":
    main()
