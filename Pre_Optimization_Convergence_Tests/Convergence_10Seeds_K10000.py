"""
Pre-Optimization Convergence Analysis (K = 10,000)

Author: Amna Omer

File Description
----------------
This script extends the pre-optimization convergence analysis by
evaluating three fixed alfalfa irrigation management strategies
using 10,000 sampled states of the world and 10 random seeds.

The analysis:
  1. Defines normal, extreme, and very extreme irrigation strategies.
  2. Samples 13 uncertain parameters using Latin hypercube sampling.
  3. Evaluates each strategy across 10 independent random seeds.
  4. Calculates mean five-year cumulative irrigation and alfalfa yield.
  5. Retains annual yield CV as an additional diagnostic metric.
  6. Produces results for comparison with the smaller sample sizes
     evaluated in Convergence_10Seeds_K5000.py.

Inputs
------
- Imperial County ERA5 daily climate data (2000–2023).
- Alfalfa_Irrigation_Model.py.
- Fixed irrigation strategies and uncertainty bounds.

Outputs
-------
Three CSV files containing strategy-level results, summary statistics,
and individual future-level simulation results.

Execution Requirements
----------------------
- Python with NumPy, pandas, SciPy, and tqdm installed.
- A multicore computing system is recommended.
- Uses up to 14 parallel worker processes.
- Performs 300,000 model evaluations.

Running the Script
------------------
From the repository root:

    python Pre_Optimization_Convergence_Tests/Convergence_10Seeds_K10000.py

The number of workers can be adjusted through max_workers to match
the available CPU resources.
"""

#%% Imports 
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import traceback
from datetime import datetime
from pathlib import Path
import sys
import time


#%% Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

output_dir = BASE_DIR / "Output_convergence_Normal_Extreme_VeryExtreme_K10000_10Seeds"
output_dir.mkdir(exist_ok=True)
sys.path.append(str(PROJECT_ROOT))
climate_csv_path = PROJECT_ROOT / "Input" / "ImperialCounty_ERA5_DailyClimate_2000_2023.csv"
climate_df_big = pd.read_csv(climate_csv_path, parse_dates=["Date"])
print(f" climate_df loaded: {len(climate_df_big)} rows")

#%% import simulation
from Alfalfa_Irrigation_Model import full_alfalfa_simulation
#%% Fixed model structure 
max_cuttings = 12
num_years    = 5
vector_len   = max_cuttings * num_years  

#%% Baseline WP/Ky/Kmitsch vectors 

wp_cut_pattern = np.array([15.5, 16.5, 17.5, 18.5, 18.5, 17.5, 16.5, 15.5, 14.5, 14.0, 13.5, 13.0])
kmitch_cut_pattern = np.array([2.4, 2.7, 3.1, 3.6, 3.6, 3.3, 3.0, 2.8, 2.5, 2.3, 2.1, 2.0])
ky_cut_pattern = np.array([0.90, 0.95, 1.05, 1.15, 1.15, 1.05, 0.95, 0.90, 0.85, 0.85, 0.80, 0.80])
KC_BASE = {
    "kc_ini": 0.40,
    "kc_mid": 1.20,
    "kc_end": 1.15,
    "kc_dormant": 0.35,
    "kc_harvest_day": 0.15,
    "kc_post_harvest": 0.25,
    "kc_pre_harvest_factor": 0.60,
}

wp_base = np.tile(wp_cut_pattern, num_years).astype(float)
Kmitsch_base = np.tile(kmitch_cut_pattern, num_years).astype(float)
ky_base = np.tile(ky_cut_pattern, num_years).astype(float)

assert len(wp_base) == vector_len
assert len(ky_base) == vector_len
assert len(Kmitsch_base) == vector_len

#%% Uncertainty bounds 
UNCERTAINTY_BOUNDS = {
    "kc_scale": [0.7, 1.3],
    "temperature_dormancy_threshold": [1.0, 6.0],
    "p_depletion": [0.41, 0.66],
    "wp_scale": [0.7, 1.3],
    "Kmitsch_scale": [0.7, 1.3],
    "climate_window_u": [0.0, 1.0],
    "AWC": [0.11, 0.20],
    "root_depth_mm": [1000, 2000],
    "K_sat": [0.08, 11.0],
    "theta_s": [0.40, 0.50],
    "theta_r": [0.05, 0.30],
    "n": [1.06, 1.40],
    "pwp": [0.04, 0.30],
}
UNC_KEYS = list(UNCERTAINTY_BOUNDS.keys())

#%% Soil params mapper
def soil_params_from_row(row: dict) -> dict:
    return {
        "theta_s": float(row["theta_s"]),
        "theta_r": float(row["theta_r"]),
        "n": float(row["n"]),
        "K_sat": float(row["K_sat"]),
        "pwp": float(row["pwp"]),
        "AWC": float(row["AWC"]),
    }

#%% Climate windows 
def list_climate_windows(climate_df: pd.DataFrame, window_years: int = 5):
    min_year = int(climate_df["Date"].dt.year.min())
    max_year = int(climate_df["Date"].dt.year.max())
    starts = list(range(min_year, max_year - window_years + 2))
    return starts

def get_climate_window(climate_df: pd.DataFrame, start_year: int, window_years: int = 5):
    end_year = start_year + window_years - 1
    mask = (
        (climate_df["Date"].dt.year >= start_year) &
        (climate_df["Date"].dt.year <= end_year)
    )
    dfw = climate_df.loc[mask].copy()
    dfw = dfw.sort_values("Date").reset_index(drop=True)
    years = list(range(start_year, end_year + 1))
    return dfw, years

CLIMATE_WINDOW_STARTS = list_climate_windows(climate_df_big, window_years=5)


def map_u_to_climate_start(u: float, starts: list[int]) -> int:
    u = float(np.clip(u, 0.0, 0.999999))
    idx = int(np.floor(u * len(starts)))
    return int(starts[idx])

CLIMATE_CACHE = {}

#%% Decision levers order  

LEVER_NAMES = [
    "planting_doy",
    "yield_cutoff",
    "establishment_days",
    "MAD",
    "refill_target",
    "survival_fraction",
    "max_irrigation",
    "initial_smt",
    "min_infiltration",
    "pre_harvest_no_irrigation_days",
    "post_cutting_buffer_days",
    "DI_establishment",
    "DI_winter",
    "DI_spring",
    "DI_summer",
    "DI_fall",
]

assert len(LEVER_NAMES) == 16

#%%  simulator wrapper 
def alfalfa_run_given_levers_and_future(vars_, future_row: dict):
    try:
        planting_day_doy = int(round(vars_[0]))
        yield_cutoff = float(vars_[1])
        establishment_days = int(round(vars_[2]))
        mad = float(vars_[3])
        refill_target = float(vars_[4])
        survival_fraction = float(vars_[5])
        max_irrigation = float(vars_[6])
        init_smt = float(vars_[7])
        min_infiltration = float(vars_[8])
        pre_harvest_days = int(round(vars_[9]))
        post_cutting_days = int(round(vars_[10]))
        
        di_levers = {
            "DI_establishment": float(vars_[11]),
            "DI_winter": float(vars_[12]),
            "DI_spring": float(vars_[13]),
            "DI_summer": float(vars_[14]),
            "DI_fall": float(vars_[15]),
        }
        
        if len(vars_) != 16:
            raise ValueError(f"vars_ length mismatch: expected 16, got {len(vars_)}")

        # uncertainties
        
        kc_scale = float(future_row["kc_scale"])
        temperature_dormancy_threshold = float(future_row["temperature_dormancy_threshold"])

        kc_ini = KC_BASE["kc_ini"] * kc_scale
        kc_mid = KC_BASE["kc_mid"] * kc_scale
        kc_end = KC_BASE["kc_end"] * kc_scale
        kc_dormant = KC_BASE["kc_dormant"] * kc_scale
        kc_harvest_day = KC_BASE["kc_harvest_day"] * kc_scale
        kc_post_harvest = KC_BASE["kc_post_harvest"] * kc_scale
        kc_pre_harvest_factor = KC_BASE["kc_pre_harvest_factor"] * kc_scale

        wp_vector = wp_base * float(future_row["wp_scale"])
        ky_vector = ky_base.copy()   
        Kmitsch_vector = Kmitsch_base * float(future_row["Kmitsch_scale"])
        root_depth_mm = int(round(future_row["root_depth_mm"]))
        p_depletion = float(np.clip(future_row["p_depletion"], 0.1, 0.8))  # FAO recommendation

        start_year = map_u_to_climate_start(future_row["climate_window_u"], CLIMATE_WINDOW_STARTS)
        key = int(start_year)
        if key in CLIMATE_CACHE:
            climate_df, years = CLIMATE_CACHE[key]
        else:
            climate_df, years = get_climate_window(climate_df_big, key, window_years=5)
            CLIMATE_CACHE[key] = (climate_df, years)


        soil_params = soil_params_from_row(future_row)

        planting_date = datetime(years[0], 1, 1) + pd.Timedelta(days=planting_day_doy - 1)
        planting_date = pd.to_datetime(planting_date)

        cutting_kc = [kc_ini, kc_ini, kc_mid, kc_mid, kc_end]
        
        default_ratio = np.array([40, 40, 60, 40]) / 180.0
        est_stages = (default_ratio * establishment_days).astype(int)
        est_stages[-1] = establishment_days - est_stages[:-1].sum()
        
        stage_lengths_by_cutting_number = {0: est_stages.tolist()}
        
        for i in range(1, int(max_cuttings) + 1):
            stage_lengths_by_cutting_number[i] = [5, 10, 10, 5]

        Y = len(years)
        C = int(max_cuttings)
        needed = Y * C
        
        year_arr = np.repeat(np.array(years, dtype=int), C)
        cut_arr  = np.tile(np.arange(1, C + 1, dtype=int), Y)
        
        cutting_param_df = pd.DataFrame({
            "Year": year_arr,
            "Cutting": cut_arr,
            "Water_Productivity": np.asarray(wp_vector[:needed], dtype=float),
            "k_mitsch": np.asarray(Kmitsch_vector[:needed], dtype=float),
            "Ky": np.asarray(ky_vector[:needed], dtype=float),
        })
        
        cut0 = pd.DataFrame([{
            "Year": int(years[0]),
            "Cutting": 0,
            "Water_Productivity": float(wp_vector[0]),
            "Ky": float(ky_vector[0]),
            "k_mitsch": float(Kmitsch_vector[0]),
        }])
        
        cutting_param_df = pd.concat([cut0, cutting_param_df], ignore_index=True)

        df_result, df_cuttings, df_yearly = full_alfalfa_simulation(
            climate_df=climate_df,
            climate_csv_path=None,
            cutting_param_csv_path=None,
            cutting_param_df=cutting_param_df,
            output_csv_path=None,
            cutting_yield_output_csv_path=None,
            yearly_yield_output_csv_path=None,
            start_date=planting_date,
            years=years,
            et0_method="penman",
            cutting_kc=cutting_kc,
            refill_target_fraction=refill_target,
            root_depth_mm=root_depth_mm,
            awc_mm_per_mm=soil_params["AWC"],
            p_depletion=p_depletion,
            MAD=mad,
            initial_smt_fraction=init_smt,
            temperature_dormancy_threshold=temperature_dormancy_threshold,
            pre_harvest_no_irrigation_days=int(round(pre_harvest_days)),
            post_cutting_buffer_days=int(round(post_cutting_days)),
            yield_min_cutting=yield_cutoff,
            min_establishment_days=establishment_days,
            stage_lengths_by_cutting_number=stage_lengths_by_cutting_number,
            kc_dormant=kc_dormant,
            kc_harvest_day=kc_harvest_day,
            kc_post_harvest=kc_post_harvest,
            kc_pre_harvest_factor=kc_pre_harvest_factor,
            establishment_cutting=0,
            survival_threshold_fraction=survival_fraction,
            use_infiltration_excess=True,
            soil_params=soil_params,
            max_irrigation_per_day=max_irrigation,
            min_infiltration=min_infiltration,
            di_levers=di_levers
        )

        yield_total_mg_ha = float(df_yearly["Total_Yield_Mg_ha_Mitsch"].sum())
        irrigation_mm = float(df_result["Irrigation_mm"].sum())
        annual_yields = df_yearly["Total_Yield_Mg_ha_Mitsch"].to_numpy(dtype=float)

        return irrigation_mm, yield_total_mg_ha, annual_yields

    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        return 9999.0, 0.0, np.zeros(num_years, dtype=float)

#%%  Futures samplers 
# --- Soil repair to keep hydraulic parameters physically consistent ---
def repair_soil_bounds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure theta_s > theta_r
    df["theta_s"] = np.maximum(df["theta_s"], df["theta_r"] + 0.05)

    # Ensure theta_r < pwp < theta_s
    df["pwp"] = np.minimum(
        np.maximum(df["pwp"], df["theta_r"] + 0.01),
        df["theta_s"] - 0.01
    )

    # Ensure AWC is physically consistent
    df["AWC"] = np.minimum(df["AWC"], df["theta_s"] - df["pwp"])
    df["AWC"] = np.maximum(df["AWC"], 0.02)

    return df
def generate_design_futures(k: int, seed: int) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=len(UNC_KEYS), seed=seed)
    u = sampler.random(k)
    lows = np.array([UNCERTAINTY_BOUNDS[x][0] for x in UNC_KEYS], dtype=float)
    highs = np.array([UNCERTAINTY_BOUNDS[x][1] for x in UNC_KEYS], dtype=float)
    X = lows + u * (highs - lows)

    df = pd.DataFrame(X, columns=UNC_KEYS)

    df = repair_soil_bounds(df)

    df.insert(0, "future_id", np.arange(k, dtype=int))
    return df

#%%  Compute expected objectives for ONE fixed strategy 
def expected_objectives_for_strategy(
    vars_,
    futures_df: pd.DataFrame,
    strategy_name=None,
    K=None,
    rep_seed=None,
    progress_every_sec=30
):
    """
    Returns:
        mean_irr, mean_yld, mean_cv, df_future_outputs
    where df_future_outputs contains one row per sampled future.
    """
    rows = []
    total_futures = len(futures_df)
    last_progress_time = time.time()

    for i, future in enumerate(futures_df.to_dict("records"), 1):
        now = time.time()
        if now - last_progress_time >= progress_every_sec:
            print(
                f" strategy={strategy_name}, K={K}, seed={rep_seed}: {i}/{total_futures} futures completed",
                flush=True
            )
            last_progress_time = now

        irr, yld_total, annual_yields = alfalfa_run_given_levers_and_future(vars_, future)

        # CV is retained as an additional diagnostic metric;
        # the final optimization uses only cumulative irrigation and yield.

        annual_yields_cv = np.asarray(annual_yields, dtype=float)[1:]

        mu = float(np.mean(annual_yields_cv)) if len(annual_yields_cv) > 0 else 0.0
        sigma = float(np.std(annual_yields_cv, ddof=0)) if len(annual_yields_cv) > 0 else 0.0

        # Diagnostics for CV behavior
        failure_flag = bool(mu <= 1e-9)

        if failure_flag:
            raw_cv = np.inf
            cv = 5.0
            cv_capped_flag = True
        else:
            raw_cv = float(sigma / mu)
            cv = float(np.clip(raw_cv, 0.0, 5.0))
            cv_capped_flag = bool(raw_cv > 5.0)

        row = {
            "future_id": int(future["future_id"]),
            "irr": float(irr),
            "yld": float(yld_total),
            "cv": float(cv),
            "raw_cv": float(raw_cv) if np.isfinite(raw_cv) else np.nan,
            "cv_capped_flag": int(cv_capped_flag),
            "failure_flag": int(failure_flag),
            "mu_yield_cv": float(mu),
            "sigma_yield_cv": float(sigma),
        }

        if strategy_name is not None:
            row["strategy"] = strategy_name
        if K is not None:
            row["K"] = int(K)
        if rep_seed is not None:
            row["rep_seed"] = int(rep_seed)

        for u in UNC_KEYS:
            row[u] = float(future[u])

        rows.append(row)

    df_future_outputs = pd.DataFrame(rows)

    print(
        f" strategy={strategy_name}, K={K}, seed={rep_seed}: finished {total_futures}/{total_futures} futures",
        flush=True
    )

    return (
        float(df_future_outputs["irr"].mean()),
        float(df_future_outputs["yld"].mean()),
        float(df_future_outputs["cv"].mean()),
        df_future_outputs,
    )


# Strategies for convergence test

def make_test_strategies():

    normal = [
        275,   # planting_doy
        0.50,  # yield_cutoff
        180,   # establishment_days
        0.35,  # MAD
        1.00,  # refill_target
        0.30,  # survival_fraction
        35.0,  # max_irrigation
        0.80,  # initial_smt
        20.0,  # min_infiltration
        3,     # pre_harvest_no_irrigation_days
        5,     # post_cutting_buffer_days
        1.00,  # DI_establishment
        1.00,  # DI_winter
        1.00,  # DI_spring
        1.00,  # DI_summer
        1.00,  # DI_fall
    ]

    extreme = [
        305,   # planting_doy
        1.50,  # yield_cutoff
        130,   # establishment_days
        0.75,  # MAD
        0.70,  # refill_target
        0.10,  # survival_fraction
        18.0,  # max_irrigation
        0.60,  # initial_smt
        8.0,   # min_infiltration
        4,     # pre_harvest_no_irrigation_days
        6,     # post_cutting_buffer_days
        0.50,  # DI_establishment
        0.00,  # DI_winter
        0.20,  # DI_spring
        0.30,  # DI_summer
        0.10,  # DI_fall
    ]

    very_extreme = [
        310,   # planting_doy
        1.80,  # yield_cutoff
        125,   # establishment_days
        0.80,  # MAD
        0.55,  # refill_target
        0.05,  # survival_fraction
        16.0,  # max_irrigation
        0.55,  # initial_smt
        6.0,   # min_infiltration
        5,     # pre_harvest_no_irrigation_days
        7,     # post_cutting_buffer_days
        0.40,  # DI_establishment
        0.00,  # DI_winter
        0.10,  # DI_spring
        0.20,  # DI_summer
        0.00,  # DI_fall
    ]

    strategies = {
        "normal": normal,
        "extreme": extreme,
        "very_extreme": very_extreme,
    }

    for name, vars_ in strategies.items():
        assert len(vars_) == len(LEVER_NAMES), \
            f"{name}: expected {len(LEVER_NAMES)} levers, got {len(vars_)}"

    return {
        name: [float(x) for x in vars_]
        for name, vars_ in strategies.items()
    }

def _worker_one(args):
    strategy_name, vars_, K, rep_seed = args
    futures_df = generate_design_futures(K, seed=rep_seed)

    irr, yld, cv, df_future_outputs = expected_objectives_for_strategy(
        vars_,
        futures_df,
        strategy_name=strategy_name,
        K=K,
        rep_seed=rep_seed
    )

    summary_row = {
        "strategy": strategy_name,
        "K": int(K),
        "rep_seed": int(rep_seed),
        "irr_mean": float(irr),
        "yld_mean": float(yld),
        "cv_mean": float(cv),
    }

    return summary_row, df_future_outputs

#%% Main convergence experiment 
def run_convergence_experiment(
    K_list=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
    n_reps=10,
    rep_seeds=None,
    max_workers=None,
    tag="run",
):

    if rep_seeds is None:
        rep_seeds = [1000 + 37*i for i in range(n_reps)]
    else:
        rep_seeds = list(rep_seeds)[:n_reps]

    strategies = make_test_strategies()


    tasks = []
    for strategy_name, vars_ in strategies.items():
        for K in K_list:
            for s in rep_seeds:
                tasks.append((strategy_name, vars_, int(K), int(s)))

    rows = []
    future_rows = []
    desc = f"Convergence test: {len(strategies)} strategies × {len(K_list)} K × {n_reps} reps"
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for summary_row, df_future_outputs in tqdm(
            ex.map(_worker_one, tasks, chunksize=1),
            total=len(tasks),
            desc=desc
        ):
            rows.append(summary_row)
            future_rows.append(df_future_outputs)
    df = pd.DataFrame(rows)
    df_futures_all = pd.concat(future_rows, ignore_index=True)
    out_csv = output_dir / f"convergence_results_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f" Saved: {out_csv}")

    summary = (
        df.groupby(["strategy", "K"])
          .agg(
              irr_mean=("irr_mean", "mean"),
              irr_min=("irr_mean", "min"),
              irr_max=("irr_mean", "max"),
              yld_mean=("yld_mean", "mean"),
              yld_min=("yld_mean", "min"),
              yld_max=("yld_mean", "max"),
              cv_mean=("cv_mean", "mean"),
              cv_min=("cv_mean", "min"),
              cv_max=("cv_mean", "max"),
          )
          .reset_index()
          .sort_values(["strategy", "K"])
    )
    
    diag_summary = (
        df_futures_all.groupby(["strategy", "K"])
          .agg(
              frac_futures_cv_capped=("cv_capped_flag", "mean"),
              frac_futures_yield_failure=("failure_flag", "mean"),
              mean_raw_cv=("raw_cv", "mean"),
              median_raw_cv=("raw_cv", "median"),
          )
          .reset_index()
    )
    
    summary = summary.merge(diag_summary, on=["strategy", "K"], how="left")

    K_ref = max(K_list)
    ref = summary[summary["K"] == K_ref][["strategy", "irr_mean", "yld_mean", "cv_mean"]].rename(
        columns={"irr_mean": "irr_ref", "yld_mean": "yld_ref", "cv_mean": "cv_ref"}
    )
    summary = summary.merge(ref, on="strategy", how="left")
    summary["irr_rel_err"] = np.abs(summary["irr_mean"] - summary["irr_ref"]) / np.maximum(np.abs(summary["irr_ref"]), 1e-9)
    summary["yld_rel_err"] = np.abs(summary["yld_mean"] - summary["yld_ref"]) / np.maximum(np.abs(summary["yld_ref"]), 1e-9)
    summary["cv_rel_err"]  = np.abs(summary["cv_mean"]  - summary["cv_ref"])  / np.maximum(np.abs(summary["cv_ref"]),  1e-9)

    out_sum = output_dir / f"convergence_summary_{tag}.csv"
    summary.to_csv(out_sum, index=False)
    out_futures = output_dir / f"convergence_future_outputs_{tag}.csv"
    df_futures_all.to_csv(out_futures, index=False)
    print(f" Saved future-level outputs: {out_futures}")
    print(f" Saved: {out_sum}")

    return df, summary

#%% Run it
if __name__ == "__main__":

    K_list = (10000,)

    rep_seeds = [
        101, 202, 303, 404, 505,
        606, 707, 808, 909, 1001
    ]

    run_convergence_experiment(
        K_list=K_list,
        n_reps=len(rep_seeds),
        rep_seeds=rep_seeds,
        max_workers=14,
        tag="3strategies_seasonalDI_K10000_10seeds"
    )
