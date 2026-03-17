"""
f1_core.py — Shared ML engine for F1 Strategy + F1 Replay
============================================================
Import this module in both apps. Contains zero Streamlit UI code.

USAGE:
    from f1_core import (
        build_degradation_model, predict_lap_time, simulate_strategy,
        load_race_laps, load_multi_race_data, calc_degradation_rates,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, CIRCUIT_DEFAULTS,
        COMPOUNDS, COMPOUND_COLORS, TEAM_COLORS,
    )
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import fastf1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── Cache ─────────────────────────────────────────────────────
CACHE_DIR = Path.home() / "f1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Shared Constants ──────────────────────────────────────────
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
COMPOUND_COLORS = {
    "SOFT":"#E8002D","MEDIUM":"#FFF200","HARD":"#FFFFFF",
    "INTERMEDIATE":"#43B02A","WET":"#0067FF","UNKNOWN":"#888",
}
TEAM_COLORS = {
    "Red Bull Racing":"#3671C6","Ferrari":"#E8002D","Mercedes":"#27F4D2",
    "McLaren":"#FF8000","Aston Martin":"#229971","Alpine":"#FF87BC",
    "Williams":"#64C4FF","RB":"#6692FF","Kick Sauber":"#52E252","Haas F1 Team":"#B6BABD",
    "Racing Point":"#F596C8","Renault":"#FFF500","AlphaTauri":"#C8C8C8",
    "Alfa Romeo":"#9B0000","Toro Rosso":"#469BFF","Force India":"#F596C8",
    "Sauber":"#9B0000","Haas":"#B6BABD","Alpine F1 Team":"#FF87BC",
}
YEARS = list(range(2025, 2018, -1))

# ── ML globals ────────────────────────────────────────────────
CIRCUIT_DEFAULTS     = {}
MODEL_PIPELINE       = None
NUMERIC_FEATURES     = [
    "TyreLife","CompoundEnc","RaceLapPct","Stint","FuelLoad","IsWarmup",
    "TrackTemp","AirTemp",
    "CircuitThrottlePct","CircuitBrakePct","CircuitAvgSpeed",
]
CATEGORICAL_FEATURES = ["Driver", "Team"]

def get_schedule(year):
    try:
        s = fastf1.get_event_schedule(year, include_testing=False)
        past = s[s["EventDate"] < pd.Timestamp.now(tz="UTC").tz_localize(None)]
        return past[["EventName","Country","Location","RoundNumber"]].to_dict("records")
    except:
        return []

def load_race_laps(year, round_number):
    """
    Load race lap data with all features:
    - Core: TyreLife, Compound, RaceLapPct, Stint, FuelLoad, IsWarmup
    - New:  TrackTemp, AirTemp, Driver, Team
    - New:  Track characteristics from telemetry (ThrottlePct, BrakePct, AvgSpeed)
    """
    try:
        sess = fastf1.get_session(year, round_number, "R")
        sess.load(laps=True, telemetry=True, weather=True, messages=False)
        laps = sess.laps.copy()

        # ── Core lap filtering ─────────────────────────────────
        laps = laps[laps["LapTime"].notna()]
        laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
        laps = laps[
            (laps["LapTimeSec"] > 0) &
            (laps["LapTimeSec"] < 200) &
            (laps["Compound"].isin(["SOFT","MEDIUM","HARD"])) &
            (laps["PitInTime"].isna()) &
            (laps["PitOutTime"].isna())
        ]

        total_laps = sess.total_laps if hasattr(sess, "total_laps") else laps["LapNumber"].max()
        laps["TotalLaps"]  = total_laps
        laps["RaceLapPct"] = laps["LapNumber"] / total_laps
        laps["FuelLoad"]   = (110.0 - laps["LapNumber"] * 1.5).clip(lower=0)
        laps["IsWarmup"]   = (laps["TyreLife"] <= 2).astype(int)

        # ── Weather: merge nearest weather sample per lap ──────
        if not sess.weather_data.empty:
            weather = sess.weather_data[["Time","TrackTemp","AirTemp"]].copy()
            weather = weather.sort_values("Time")
            laps_sorted = laps.sort_values("LapStartTime")
            merged = pd.merge_asof(
                laps_sorted,
                weather.rename(columns={"Time":"LapStartTime"}),
                on="LapStartTime", direction="nearest")
            laps = merged
        else:
            laps["TrackTemp"] = 35.0   # sensible defaults
            laps["AirTemp"]   = 22.0

        # ── Track characteristics from telemetry ───────────────
        # Sample one fast lap per driver to get circuit fingerprint
        throttle_pcts, brake_pcts, avg_speeds = [], [], []
        sampled = (laps.groupby("Driver")
                       .apply(lambda g: g.nsmallest(1, "LapTimeSec"))
                       .reset_index(drop=True))
        circuit_throttle_pct = 65.0   # defaults
        circuit_brake_pct    = 12.0
        circuit_avg_speed    = 210.0
        for _, row in sampled.head(5).iterrows():
            try:
                # Use LapNumber + Driver to locate the exact lap in sess.laps
                # row.name is unreliable after reset_index — LapNumber is stable
                lap_number = row["LapNumber"]
                driver     = row["Driver"]
                lap_row = sess.laps[
                    (sess.laps["LapNumber"] == lap_number) &
                    (sess.laps["Driver"]    == driver)
                ]
                if lap_row.empty: continue
                tel = lap_row.iloc[0].get_car_data()
                if tel.empty: continue
                throttle_pcts.append((tel["Throttle"] > 80).mean() * 100)
                brake_pcts.append(tel["Brake"].astype(float).mean() * 100)
                avg_speeds.append(tel["Speed"].mean())
            except Exception:
                continue
        if throttle_pcts:
            circuit_throttle_pct = float(np.mean(throttle_pcts))
            circuit_brake_pct    = float(np.mean(brake_pcts))
            circuit_avg_speed    = float(np.mean(avg_speeds))

        # These are circuit-level constants — same for every lap
        laps["CircuitThrottlePct"] = circuit_throttle_pct
        laps["CircuitBrakePct"]    = circuit_brake_pct
        laps["CircuitAvgSpeed"]    = circuit_avg_speed

        laps["Event"] = sess.event["EventName"]
        laps["Year"]  = year

        return laps[["Driver","Team","LapNumber","LapTimeSec","Compound",
                      "TyreLife","Stint","RaceLapPct","TotalLaps",
                      "FuelLoad","IsWarmup",
                      "TrackTemp","AirTemp",
                      "CircuitThrottlePct","CircuitBrakePct","CircuitAvgSpeed",
                      "Event","Year"]]
    except Exception as e:
        return pd.DataFrame()

def load_multi_race_data(circuit_name, years_to_use):
    """Load last N races at this circuit for ML training."""
    all_laps = []
    for year in years_to_use:
        schedule = get_schedule(year)
        matches = [e for e in schedule if circuit_name.lower() in e["EventName"].lower()]
        if not matches:
            continue
        ev = matches[0]
        laps = load_race_laps(year, ev["RoundNumber"])
        if not laps.empty:
            all_laps.append(laps)
    return pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()

# ── ML Model ──────────────────────────────────────────────────
# Global state — populated at train time, reused at predict time
CIRCUIT_DEFAULTS = {}   # circuit fingerprint medians
MODEL_PIPELINE   = None # full sklearn Pipeline (preprocessor + estimator)

# Numeric features passed through StandardScaler
NUMERIC_FEATURES = [
    "TyreLife","CompoundEnc","RaceLapPct","Stint","FuelLoad","IsWarmup",
    "TrackTemp","AirTemp",
    "CircuitThrottlePct","CircuitBrakePct","CircuitAvgSpeed",
]
# Categorical features OneHotEncoded — no ordinal relationship
CATEGORICAL_FEATURES = ["Driver", "Team"]

def build_degradation_model(laps_df):
    """
    Train a full sklearn Pipeline:
      Preprocessor:
        - Numeric features  → StandardScaler
        - Driver, Team      → OneHotEncoder (handle_unknown="ignore")
        - Compound          → manual ordinal map (hardness IS ordinal)
      Estimator:
        - GradientBoostingRegressor
    """
    global CIRCUIT_DEFAULTS, MODEL_PIPELINE

    if laps_df.empty or len(laps_df) < 50:
        return None, None, "Not enough data to train model"

    df = laps_df.copy()
    df = df.dropna(subset=["LapTimeSec","TyreLife","Compound","RaceLapPct","Stint",
                            "Driver","Team"])
    df = df[df["Compound"].isin(COMPOUNDS)]

    # Compound stays ordinal — hardness order is meaningful
    df["CompoundEnc"] = df["Compound"].map({"SOFT":0,"MEDIUM":1,"HARD":2})

    # Fill weather/circuit defaults if missing
    defaults = {"TrackTemp":35.0,"AirTemp":22.0,
                "CircuitThrottlePct":65.0,"CircuitBrakePct":12.0,"CircuitAvgSpeed":210.0,
                "FuelLoad":55.0,"IsWarmup":0}
    for col, val in defaults.items():
        if col not in df.columns: df[col] = val
        df[col] = df[col].fillna(val)

    # Store circuit fingerprint for predict-time fallback
    CIRCUIT_DEFAULTS = {
        col: float(df[col].median())
        for col in ["CircuitThrottlePct","CircuitBrakePct","CircuitAvgSpeed",
                    "TrackTemp","AirTemp"]
    }

    # ── Build ColumnTransformer ───────────────────────────────
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            min_samples_leaf=5, subsample=0.85, random_state=42)),
    ])

    feature_df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["LapTimeSec"]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    MODEL_PIPELINE = pipeline

    mae = mean_absolute_error(y_test, pipeline.predict(X_test))
    return pipeline, mae, None

def predict_lap_time(model, tyre_life, compound, race_lap_pct, stint,
                     fuel_load=55.0, is_warmup=0,
                     track_temp=None, air_temp=None,
                     driver=None, team=None):
    """
    Predict lap time via the full sklearn Pipeline.
    Driver + Team are passed as raw strings — OneHotEncoder handles them.
    Unknown drivers/teams fall back gracefully via handle_unknown="ignore".
    """
    if model is None: return None

    comp_enc = {"SOFT":0,"MEDIUM":1,"HARD":2}.get(compound, 1)
    warmup   = 1 if (is_warmup or tyre_life <= 2) else 0
    t_temp   = track_temp if track_temp is not None else CIRCUIT_DEFAULTS.get("TrackTemp", 35.0)
    a_temp   = air_temp   if air_temp   is not None else CIRCUIT_DEFAULTS.get("AirTemp",   22.0)
    th_pct   = CIRCUIT_DEFAULTS.get("CircuitThrottlePct", 65.0)
    br_pct   = CIRCUIT_DEFAULTS.get("CircuitBrakePct",    12.0)
    av_spd   = CIRCUIT_DEFAULTS.get("CircuitAvgSpeed",   210.0)

    # Build a single-row DataFrame matching the pipeline's expected input
    row = pd.DataFrame([{
        "TyreLife":            tyre_life,
        "CompoundEnc":         comp_enc,
        "RaceLapPct":          race_lap_pct,
        "Stint":               stint,
        "FuelLoad":            fuel_load,
        "IsWarmup":            warmup,
        "TrackTemp":           t_temp,
        "AirTemp":             a_temp,
        "CircuitThrottlePct":  th_pct,
        "CircuitBrakePct":     br_pct,
        "CircuitAvgSpeed":     av_spd,
        "Driver":              driver or "UNKNOWN",
        "Team":                team   or "UNKNOWN",
    }])
    return model.predict(row)[0]

def simulate_strategy(model, current_lap, total_laps, current_compound,
                      current_tyre_life, current_stint, strategies,
                      sc_laps=None, traffic_laps=None, traffic_loss=0.5,
                      track_temp=None, air_temp=None, driver=None, team=None):
    """
    Simulate multiple pit strategies from current lap to end.

    Factors modelled:
      - Tyre degradation (via ML model, 13 features)
      - Fuel load, tyre warmup, SC, traffic
      - TrackTemp, AirTemp, Driver, Team, Circuit characteristics
    Returns dict of strategy_name -> result dict.
    """
    sc_laps    = set(sc_laps or [])
    traffic_laps = set(traffic_laps or [])
    results = {}
    remaining = total_laps - current_lap
    # Current fuel at start of simulation
    start_fuel = max(0.0, 110.0 - current_lap * 1.5)
    # SC average lap time multiplier (cars travel ~30% slower)
    SC_SLOWDOWN = 1.30

    for strat_name, pit_laps, compounds in strategies:
        total_time  = 0.0
        tyre_life   = current_tyre_life
        compound    = current_compound
        stint       = current_stint
        pit_stop_penalty = 22.0  # avg pit lane loss in seconds

        pit_queue = list(zip(pit_laps, compounds))

        for lap_offset in range(remaining):
            lap = current_lap + lap_offset

            # ── Pit stop ──────────────────────────────────────
            if pit_queue and pit_queue[0][0] == lap:
                total_time += pit_stop_penalty
                _, compound = pit_queue.pop(0)
                tyre_life   = 1
                stint      += 1
            else:
                tyre_life += 1

            # ── Fuel load this lap ────────────────────────────
            fuel_load = max(0.0, start_fuel - lap_offset * 1.5)

            # ── Warmup flag ───────────────────────────────────
            is_warmup = 1 if tyre_life <= 2 else 0

            # ── Base lap time from model ───────────────────────
            race_lap_pct = lap / total_laps
            lap_time = predict_lap_time(
                model, tyre_life, compound, race_lap_pct,
                stint, fuel_load, is_warmup,
                track_temp=track_temp, air_temp=air_temp,
                driver=driver, team=team)
            if lap_time is None:
                continue

            # ── Safety Car: freeze pace, everyone queues up ───
            if lap in sc_laps:
                lap_time = lap_time * SC_SLOWDOWN

            # ── Traffic penalty ───────────────────────────────
            if lap in traffic_laps:
                lap_time += traffic_loss

            total_time += lap_time

        results[strat_name] = {
            "total_time": total_time,
            "pit_laps":   pit_laps,
            "compounds":  compounds,
        }

    return results


def monte_carlo_simulation(model, current_lap, total_laps, current_compound,
                            current_tyre_life, current_stint, strategies,
                            laps_df, n_simulations=500,
                            sc_prob_per_lap=0.04, traffic_active=False,
                            traffic_loss=0.5, traffic_laps_ahead=5,
                            track_temp=None, air_temp=None,
                            driver=None, team=None):
    """
    Run N Monte Carlo simulations per strategy, sampling randomness from:
      - Lap time noise  (residual std from historical data)
      - SC probability  (random SC deployment each lap)
      - Pit stop time   (sampled from real pit stop distribution)
      - Traffic         (random traffic encounters if enabled)

    Returns dict: strategy -> array of N total race times
    """
    # Estimate lap time noise (std of residuals from historical data)
    if not laps_df.empty and len(laps_df) > 20:
        noise_std = laps_df["LapTimeSec"].std() * 0.12   # ~12% of std = realistic lap variance
        noise_std = np.clip(noise_std, 0.05, 0.8)
    else:
        noise_std = 0.25

    # Pit stop time distribution: mean 22s, std 1.5s (based on real F1 data)
    PIT_MEAN, PIT_STD = 22.0, 1.5

    results = {name: [] for name, _, _ in strategies}
    remaining = total_laps - current_lap
    start_fuel = max(0.0, 110.0 - current_lap * 1.5)
    SC_SLOWDOWN = 1.30

    for _ in range(n_simulations):
        # Draw random SC laps for this simulation run
        sc_this_run = set()
        in_sc = False
        sc_end = 0
        for lap in range(current_lap, total_laps):
            if in_sc:
                sc_this_run.add(lap)
                if lap >= sc_end:
                    in_sc = False
            elif np.random.random() < sc_prob_per_lap:
                in_sc = True
                sc_duration_this = np.random.randint(2, 6)
                sc_end = lap + sc_duration_this
                sc_this_run.add(lap)

        for strat_name, pit_laps, compounds in strategies:
            total_time = 0.0
            tyre_life  = current_tyre_life
            compound   = current_compound
            stint      = current_stint
            pit_queue  = list(zip(pit_laps, compounds))

            # SC opportunity: if SC happens near planned pit, shift pit to SC lap
            adjusted_queue = list(pit_queue)
            for i, (pl, comp) in enumerate(adjusted_queue):
                nearby_sc = [s for s in sc_this_run if abs(s - pl) <= 2]
                if nearby_sc:
                    adjusted_queue[i] = (min(nearby_sc), comp)
            pit_queue = adjusted_queue

            for lap_offset in range(remaining):
                lap = current_lap + lap_offset

                if pit_queue and pit_queue[0][0] == lap:
                    # Sample pit stop time from distribution
                    pit_time = np.random.normal(PIT_MEAN, PIT_STD)
                    total_time += max(18.0, pit_time)
                    _, compound = pit_queue.pop(0)
                    tyre_life   = 1
                    stint      += 1
                else:
                    tyre_life += 1

                fuel_load    = max(0.0, start_fuel - lap_offset * 1.5)
                is_warmup    = 1 if tyre_life <= 2 else 0
                race_lap_pct = lap / total_laps

                lap_time = predict_lap_time(
                    model, tyre_life, compound, race_lap_pct,
                    stint, fuel_load, is_warmup,
                    track_temp=track_temp, air_temp=air_temp,
                    driver=driver, team=team)
                if lap_time is None:
                    continue

                # Add lap time noise
                lap_time += np.random.normal(0, noise_std)

                # SC slowdown
                if lap in sc_this_run:
                    lap_time *= SC_SLOWDOWN

                # Traffic
                if traffic_active and lap_offset < traffic_laps_ahead:
                    lap_time += np.random.uniform(0, traffic_loss * 1.5)

                total_time += max(60.0, lap_time)  # floor at 60s

            results[strat_name].append(total_time)

    return {k: np.array(v) for k, v in results.items()}


def calc_degradation_rates(laps_df):
    """Calculate avg deg rate per compound (seconds per lap of tyre life)."""
    rates = {}
    for comp in COMPOUNDS:
        df = laps_df[laps_df["Compound"]==comp].copy()
        if len(df) < 10: continue
        # Fit linear regression: lap_time ~ tyre_life
        from numpy.polynomial import polynomial as P
        df = df[df["TyreLife"] <= 40].sort_values("TyreLife")
        if len(df) < 5: continue
        coeff = np.polyfit(df["TyreLife"], df["LapTimeSec"], 1)
        rates[comp] = coeff[0]  # seconds per lap of tyre life
    return rates

def optimal_pit_window(model, current_lap, total_laps, current_compound,
                        current_tyre_life, current_stint, next_compound):
    """Find the single best lap to pit using brute force simulation."""
    best_lap, best_time = None, float("inf")
    remaining = total_laps - current_lap

    for pit_offset in range(1, remaining - 5):
        pit_lap = current_lap + pit_offset
        strat = [("test", [pit_lap], [next_compound])]
        result = simulate_strategy(model, current_lap, total_laps, current_compound,
                                   current_tyre_life, current_stint, strat)
        t = result["test"]["total_time"]
        if t < best_time:
            best_time = t
            best_lap = pit_lap

    return best_lap, best_time

# ── Full Grid Simulator ───────────────────────────────────────
def simulate_full_grid(model, race_laps_df, total_laps,
                       track_temp=35.0, air_temp=22.0):
    """
    Simulate the entire grid lap-by-lap using actual pit stop history
    from FastF1 but ML-predicted lap times.

    Returns a dict keyed by lap number, each value a list of car states:
        {driver, team, predicted_pos, actual_pos, gap_to_leader,
         compound, tyre_life, stint, predicted_lap_time, actual_lap_time,
         pit_this_lap, undercut_flag}

    Architecture:
      - Each car follows its ACTUAL pit stop laps + compounds from FastF1
      - Lap times are ML-predicted (not actual) so we see what SHOULD have happened
      - Gaps are recalculated from cumulative predicted time each lap
      - Positions are derived from gap ranking
      - Undercut flag = car behind pitted and rejoined within 3s of car ahead
    """
    if model is None or race_laps_df.empty:
        return {}

    drivers = sorted(race_laps_df["Driver"].unique().tolist())

    # ── Build per-driver stint schedule from actual race data ──
    # stint_schedule[driver] = list of {start_lap, compound}
    stint_schedule = {}
    for drv in drivers:
        drv_laps = race_laps_df[race_laps_df["Driver"] == drv].sort_values("LapNumber")
        stints   = []
        prev_comp, prev_stint = None, None
        for _, row in drv_laps.iterrows():
            comp  = str(row.get("Compound","MEDIUM")).upper()
            stint = row.get("Stint", 1)
            if comp not in COMPOUNDS: comp = "MEDIUM"
            if comp != prev_comp or stint != prev_stint:
                stints.append({
                    "start_lap": int(row["LapNumber"]),
                    "compound":  comp,
                    "stint_num": int(stint) if pd.notna(stint) else len(stints)+1,
                })
                prev_comp, prev_stint = comp, stint
        stint_schedule[drv] = stints if stints else [{"start_lap":1,"compound":"MEDIUM","stint_num":1}]

    # ── Get team mapping ───────────────────────────────────────
    team_map = {}
    if "Team" in race_laps_df.columns:
        for drv in drivers:
            rows = race_laps_df[race_laps_df["Driver"]==drv]
            if not rows.empty:
                team_map[drv] = rows["Team"].iloc[0]

    # ── Initial car states ─────────────────────────────────────
    car_states = {}
    for drv in drivers:
        first_stint = stint_schedule[drv][0]
        # Try to get actual starting position from lap 1
        lap1 = race_laps_df[(race_laps_df["Driver"]==drv) &
                             (race_laps_df["LapNumber"]==1)]
        start_pos = int(lap1["Position"].iloc[0]) if (not lap1.empty and
                    "Position" in lap1.columns and
                    pd.notna(lap1["Position"].iloc[0])) else drivers.index(drv)+1
        car_states[drv] = {
            "cum_time":   0.0,
            "tyre_life":  1,
            "compound":   first_stint["compound"],
            "stint":      first_stint["stint_num"],
            "position":   start_pos,
            "retired":    False,
        }

    # ── Actual lap times lookup for comparison ─────────────────
    actual_times = {}
    for _, row in race_laps_df.iterrows():
        key = (row["Driver"], int(row["LapNumber"]))
        lt  = row.get("LapTimeSec", np.nan)
        actual_times[key] = float(lt) if pd.notna(lt) else None

    # ── Actual positions lookup ────────────────────────────────
    actual_positions = {}
    if "Position" in race_laps_df.columns:
        for _, row in race_laps_df.iterrows():
            pos = row.get("Position")
            if pd.notna(pos):
                actual_positions[(row["Driver"], int(row["LapNumber"]))] = int(pos)

    # ── Main simulation loop ───────────────────────────────────
    grid_history = {}   # lap -> list of car dicts
    PIT_LOSS     = 22.0  # seconds lost in pit lane

    for lap in range(1, total_laps + 1):
        race_lap_pct = lap / total_laps
        fuel_load    = max(0.0, 110.0 - lap * 1.5)
        lap_results  = []

        # Determine which cars pit this lap (from actual data)
        pitting_this_lap = set()
        for drv in drivers:
            schedule = stint_schedule[drv]
            for i, s in enumerate(schedule[1:], 1):
                if s["start_lap"] == lap:
                    pitting_this_lap.add(drv)
                    break

        # Predict lap time for every car
        for drv in drivers:
            state = car_states[drv]
            if state["retired"]:
                continue

            # Check if pitting this lap
            pitting = drv in pitting_this_lap

            # Find current stint compound/number from schedule
            schedule = stint_schedule[drv]
            cur_stint_info = schedule[0]
            for s in schedule:
                if s["start_lap"] <= lap:
                    cur_stint_info = s

            compound   = cur_stint_info["compound"]
            stint_num  = cur_stint_info["stint_num"]
            tyre_life  = state["tyre_life"]
            is_warmup  = 1 if tyre_life <= 2 else 0
            team       = team_map.get(drv)

            # ML predicted lap time
            pred_lt = predict_lap_time(
                model, tyre_life, compound, race_lap_pct,
                stint_num, fuel_load, is_warmup,
                track_temp=track_temp, air_temp=air_temp,
                driver=drv, team=team)
            if pred_lt is None:
                pred_lt = 90.0  # fallback

            # Add pit loss to cumulative time
            pit_cost = PIT_LOSS if pitting else 0.0

            # Update cumulative time
            state["cum_time"] += pred_lt + pit_cost

            # Update tyre state for next lap
            if pitting:
                state["tyre_life"] = 1
                state["compound"]  = compound
                state["stint"]     = stint_num
            else:
                state["tyre_life"] += 1

            lap_results.append({
                "driver":             drv,
                "team":               team or "",
                "compound":           compound,
                "tyre_life":          tyre_life,
                "stint":              stint_num,
                "cum_time":           state["cum_time"],
                "predicted_lap_time": pred_lt,
                "actual_lap_time":    actual_times.get((drv, lap)),
                "actual_pos":         actual_positions.get((drv, lap)),
                "pit_this_lap":       pitting,
                "undercut_flag":      False,
            })

        # Rank by cumulative time → predicted position
        lap_results.sort(key=lambda x: x["cum_time"])
        for i, car in enumerate(lap_results):
            car["predicted_pos"] = i + 1

        # Gap to leader
        if lap_results:
            leader_time = lap_results[0]["cum_time"]
            for car in lap_results:
                car["gap_to_leader"] = round(car["cum_time"] - leader_time, 3)

        # Undercut detection: car pitted and is now within 3s of a car it was behind
        prev_lap_data = grid_history.get(lap - 1, [])
        if prev_lap_data:
            prev_pos = {c["driver"]: c["predicted_pos"] for c in prev_lap_data}
            cur_pos  = {c["predicted_pos"]: c for c in lap_results}
            for car in lap_results:
                if car["pit_this_lap"]:
                    drv = car["driver"]
                    old_p = prev_pos.get(drv, 99)
                    new_p = car["predicted_pos"]
                    # Gained at least 1 position AND within 3s of new car ahead
                    if new_p < old_p:
                        ahead = cur_pos.get(new_p - 1)
                        if ahead and car["gap_to_leader"] - ahead["gap_to_leader"] < 3.0:
                            car["undercut_flag"] = True

        grid_history[lap] = lap_results

    return grid_history


# ── Charts ────────────────────────────────────────────────────