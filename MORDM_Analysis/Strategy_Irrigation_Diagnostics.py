"""
Irrigation Diagnostics for Representative Alfalfa Strategies

Author: Amna Omer

File Description
----------------
Evaluates five representative Pareto irrigation strategies across
the original 300 sampled states of the world to examine routine,
forced, and survival irrigation, seasonal irrigation patterns,
and consistency with the original optimization results.

Inputs
------
- NSGA2_K300_P100_G100_S101/design_futures_k300.csv.
- Strategies_Performance/selected_5_representative_strategies.csv.
- Input/ImperialCounty_ERA5_DailyClimate_2000_2023.csv.
- Alfalfa_Irrigation_Model.py.

Outputs
-------
- Future-level irrigation and yield diagnostics.
- Strategy-level irrigation and yield summaries.
- Irrigation type and seasonal irrigation summaries.
- Comparison with original NSGA-II results.
- Irrigation component balance checks.

Execution Requirements
----------------------
Requires numpy, pandas, and the core alfalfa irrigation model.
The analysis uses 14 parallel worker processes and is
recommended for a multicore Linux computing system.

Running the Script
------------------
From the MORDM_Analysis directory:

python Strategy_Irrigation_Diagnostics.py
"""


import os
import sys
import time
import traceback
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

OPTIMIZATION_OUTPUT_DIR = BASE_DIR / "NSGA2_K300_P100_G100_S101"
DESIGN_FUTURES_FILE = OPTIMIZATION_OUTPUT_DIR / "design_futures_k300.csv"

STRATEGIES_DIR = BASE_DIR / "Strategies_Performance"
SELECTED_STRATEGIES_FILE = STRATEGIES_DIR / "selected_5_representative_strategies.csv"

CLIMATE_FILE = PROJECT_ROOT / "Input" / "ImperialCounty_ERA5_DailyClimate_2000_2023.csv"

OUTPUT_DIR = STRATEGIES_DIR / "Irrigation_Diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT_ROOT))
from Alfalfa_Irrigation_Model import full_alfalfa_simulation



climate_df_big = pd.read_csv(CLIMATE_FILE, parse_dates=["Date"])
print(f"Climate data loaded: {len(climate_df_big):,} days")

max_cuttings = 12
num_years = 5

wp_cut_pattern = np.array([
    15.5, 16.5, 17.5, 18.5,
    18.5, 17.5, 16.5, 15.5,
    14.5, 14.0, 13.5, 13.0
])

kmitch_cut_pattern = np.array([
    2.4, 2.7, 3.1, 3.6,
    3.6, 3.3, 3.0, 2.8,
    2.5, 2.3, 2.1, 2.0
])

ky_cut_pattern = np.array([
    0.90, 0.95, 1.05, 1.15,
    1.15, 1.05, 0.95, 0.90,
    0.85, 0.85, 0.80, 0.80
])

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

DECISION_NAMES = [
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_climate_windows(climate_df, window_years=5):
    min_year = int(climate_df["Date"].dt.year.min())
    max_year = int(climate_df["Date"].dt.year.max())
    return list(range(min_year, max_year - window_years + 2))


def get_climate_window(climate_df, start_year, window_years=5):
    end_year = start_year + window_years - 1

    mask = (
        (climate_df["Date"].dt.year >= start_year)
        & (climate_df["Date"].dt.year <= end_year)
    )

    dfw = (
        climate_df.loc[mask]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    years = list(range(start_year, end_year + 1))
    return dfw, years


def map_u_to_climate_start(u, starts):
    u = float(np.clip(u, 0.0, 0.999999))
    idx = int(np.floor(u * len(starts)))
    return int(starts[idx])


CLIMATE_WINDOW_STARTS = list_climate_windows(
    climate_df_big,
    window_years=5
)

CLIMATE_CACHE = {}


def soil_params_from_row(row):
    return {
        "theta_s": float(row["theta_s"]),
        "theta_r": float(row["theta_r"]),
        "n": float(row["n"]),
        "K_sat": float(row["K_sat"]),
        "pwp": float(row["pwp"]),
        "AWC": float(row["AWC"]),
    }


def build_cutting_param_df(years, wp_vector, ky_vector, kmitsch_vector):
    Y = len(years)
    C = int(max_cuttings)
    needed = Y * C

    cutting_param_df = pd.DataFrame({
        "Year": np.repeat(np.array(years, dtype=int), C),
        "Cutting": np.tile(np.arange(1, C + 1, dtype=int), Y),
        "Water_Productivity": np.asarray(wp_vector[:needed], dtype=float),
        "k_mitsch": np.asarray(kmitsch_vector[:needed], dtype=float),
        "Ky": np.asarray(ky_vector[:needed], dtype=float),
    })

    cut0 = pd.DataFrame([{
        "Year": int(years[0]),
        "Cutting": 0,
        "Water_Productivity": float(wp_vector[0]),
        "Ky": float(ky_vector[0]),
        "k_mitsch": float(kmitsch_vector[0]),
    }])

    return pd.concat([cut0, cutting_param_df], ignore_index=True)


def parse_strategy(row):
    scalar_levers = {
        "planting_doy": int(round(row["planting_doy"])),
        "yield_cutoff": float(row["yield_cutoff"]),
        "establishment_days": int(round(row["establishment_days"])),
        "MAD": float(row["MAD"]),
        "refill_target": float(row["refill_target"]),
        "survival_fraction": float(row["survival_fraction"]),
        "max_irrigation": float(row["max_irrigation"]),
        "initial_smt": float(row["initial_smt"]),
        "min_infiltration": float(row["min_infiltration"]),
        "pre_harvest_no_irrigation_days": int(
            round(row["pre_harvest_no_irrigation_days"])
        ),
        "post_cutting_buffer_days": int(
            round(row["post_cutting_buffer_days"])
        ),
    }

    di_levers = {
        "DI_establishment": float(row["DI_establishment"]),
        "DI_winter": float(row["DI_winter"]),
        "DI_spring": float(row["DI_spring"]),
        "DI_summer": float(row["DI_summer"]),
        "DI_fall": float(row["DI_fall"]),
    }

    return scalar_levers, di_levers


# =============================================================================
# RUN ONE STRATEGY × ONE FUTURE
# =============================================================================

def run_strategy_future(scalar_levers, di_levers, future_row):
    global CLIMATE_CACHE

    kc_scale = float(future_row["kc_scale"])
    temperature_dormancy_threshold = float(
        future_row["temperature_dormancy_threshold"]
    )

    kc_ini = KC_BASE["kc_ini"] * kc_scale
    kc_mid = KC_BASE["kc_mid"] * kc_scale
    kc_end = KC_BASE["kc_end"] * kc_scale
    kc_dormant = KC_BASE["kc_dormant"] * kc_scale
    kc_harvest_day = KC_BASE["kc_harvest_day"] * kc_scale
    kc_post_harvest = KC_BASE["kc_post_harvest"] * kc_scale
    kc_pre_harvest_factor = KC_BASE["kc_pre_harvest_factor"] * kc_scale

    wp_vector = wp_base * float(future_row["wp_scale"])
    ky_vector = ky_base.copy()
    kmitsch_vector = Kmitsch_base * float(future_row["Kmitsch_scale"])

    root_depth_mm = int(round(future_row["root_depth_mm"]))
    p_depletion = float(np.clip(future_row["p_depletion"], 0.1, 0.8))
    soil_params = soil_params_from_row(future_row)

    start_year = map_u_to_climate_start(
        future_row["climate_window_u"],
        CLIMATE_WINDOW_STARTS
    )

    if start_year in CLIMATE_CACHE:
        climate_df, years = CLIMATE_CACHE[start_year]
    else:
        climate_df, years = get_climate_window(
            climate_df_big,
            start_year,
            window_years=5
        )
        CLIMATE_CACHE[start_year] = (climate_df, years)

    planting_day_doy = int(round(scalar_levers["planting_doy"]))

    planting_date = datetime(years[0], 1, 1) + pd.Timedelta(
        days=planting_day_doy - 1
    )
    planting_date = pd.to_datetime(planting_date)

    establishment_days = int(
        round(scalar_levers["establishment_days"])
    )

    default_ratio = np.array([40, 40, 60, 40]) / 180.0
    est_stages = (default_ratio * establishment_days).astype(int)
    est_stages[-1] = establishment_days - est_stages[:-1].sum()

    stage_lengths_by_cutting_number = {
        0: est_stages.tolist()
    }

    for i in range(1, max_cuttings + 1):
        stage_lengths_by_cutting_number[i] = [5, 10, 10, 5]

    cutting_kc = [
        kc_ini,
        kc_ini,
        kc_mid,
        kc_mid,
        kc_end
    ]

    cutting_param_df = build_cutting_param_df(
        years=years,
        wp_vector=wp_vector,
        ky_vector=ky_vector,
        kmitsch_vector=kmitsch_vector,
    )

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
        refill_target_fraction=scalar_levers["refill_target"],
        root_depth_mm=root_depth_mm,
        awc_mm_per_mm=soil_params["AWC"],
        p_depletion=p_depletion,
        MAD=scalar_levers["MAD"],
        initial_smt_fraction=scalar_levers["initial_smt"],
        temperature_dormancy_threshold=temperature_dormancy_threshold,
        pre_harvest_no_irrigation_days=
            scalar_levers["pre_harvest_no_irrigation_days"],
        post_cutting_buffer_days=
            scalar_levers["post_cutting_buffer_days"],
        yield_min_cutting=scalar_levers["yield_cutoff"],
        min_establishment_days=establishment_days,
        stage_lengths_by_cutting_number=stage_lengths_by_cutting_number,
        kc_dormant=kc_dormant,
        kc_harvest_day=kc_harvest_day,
        kc_post_harvest=kc_post_harvest,
        kc_pre_harvest_factor=kc_pre_harvest_factor,
        establishment_cutting=0,
        survival_threshold_fraction=scalar_levers["survival_fraction"],
        use_infiltration_excess=True,
        soil_params=soil_params,
        max_irrigation_per_day=scalar_levers["max_irrigation"],
        min_infiltration=scalar_levers["min_infiltration"],
        di_levers=di_levers
    )

    return (
        df_result,
        df_cuttings,
        df_yearly,
        years,
        planting_date,
        establishment_days,
        start_year
    )


# =============================================================================
# SUMMARIZE ONE FUTURE
# =============================================================================

def assign_period(date, establishment_end):
    date = pd.Timestamp(date)

    if date <= establishment_end:
        return "Establishment"

    if date.month in [12, 1, 2]:
        return "Winter"
    if date.month in [3, 4, 5]:
        return "Spring"
    if date.month in [6, 7, 8]:
        return "Summer"

    return "Fall"


def summarize_future(
    df_result,
    df_yearly,
    planting_date,
    establishment_days
):
    df = df_result.copy()
    df.index = pd.to_datetime(df.index)

    establishment_end = planting_date + pd.Timedelta(
        days=establishment_days - 1
    )

    df["Period"] = [
        assign_period(date, establishment_end)
        for date in df.index
    ]

    total_irr = float(df["Irrigation_mm"].sum())
    routine = float(df["Routine_Irrigation_mm"].sum())
    force = float(df["Force_Irrigation_mm"].sum())
    survival = float(df["Survival_Irrigation_mm"].sum())
    emergency = force + survival

    total_yield = float(
        df_yearly["Total_Yield_Mg_ha_Mitsch"].sum()
    )

    n_years = 5.0

    result = {
        "annual_irrigation_mm": total_irr / n_years,
        "annual_routine_mm": routine / n_years,
        "annual_force_mm": force / n_years,
        "annual_survival_mm": survival / n_years,
        "annual_emergency_mm": emergency / n_years,
        "annual_yield_Mg_ha": total_yield / n_years,
    }

    periods = [
        "Establishment",
        "Winter",
        "Spring",
        "Summer",
        "Fall"
    ]

    irrigation_columns = {
        "total": "Irrigation_mm",
        "routine": "Routine_Irrigation_mm",
        "force": "Force_Irrigation_mm",
        "survival": "Survival_Irrigation_mm",
    }

    for period in periods:
        mask = df["Period"] == period

        for short_name, column in irrigation_columns.items():
            value = float(df.loc[mask, column].sum())
            result[
                f"{period.lower()}_{short_name}_mm_yr"
            ] = value / n_years

    result["mass_balance_error_mm_5yr"] = (
        total_irr - (routine + force + survival)
    )

    return result


# =============================================================================
# MULTIPROCESSING
# =============================================================================

def evaluate_one_strategy_future(
    strategy_name,
    scalar_levers,
    di_levers,
    future_row
):
    try:
        future_id = future_row.get("future_id", "NA")

        (
            df_result,
            df_cuttings,
            df_yearly,
            years,
            planting_date,
            establishment_days,
            climate_start_year
        ) = run_strategy_future(
            scalar_levers=scalar_levers,
            di_levers=di_levers,
            future_row=future_row
        )

        summary = summarize_future(
            df_result=df_result,
            df_yearly=df_yearly,
            planting_date=planting_date,
            establishment_days=establishment_days
        )

        summary.update({
            "Strategy": strategy_name,
            "future_id": future_id,
            "climate_start_year": climate_start_year
        })

        return summary

    except Exception as e:
        print(
            f"\nERROR | {strategy_name} | "
            f"future={future_row.get('future_id', 'NA')} | {e}",
            flush=True
        )
        traceback.print_exc()
        raise


def _diagnostic_worker_wrapper(args):
    return evaluate_one_strategy_future(*args)


def run_parallel_diagnostics(
    strategies,
    design_futures,
    n_workers=14
):
    start_time = time.time()
    all_results = []

    total_strategies = len(strategies)
    futures_per_strategy = len(design_futures)
    total_runs = total_strategies * futures_per_strategy
    completed_total = 0

    print("\n" + "=" * 100)
    print("REPRESENTATIVE STRATEGY IRRIGATION DIAGNOSTICS")
    print("=" * 100)
    print(f"Strategies          : {total_strategies}")
    print(f"Futures per strategy: {futures_per_strategy}")
    print(f"Total simulations   : {total_runs}")
    print(f"Worker processes    : {n_workers}")
    print("=" * 100)

    future_records = design_futures.to_dict("records")
    ctx = mp.get_context("fork")

    with ctx.Pool(processes=n_workers) as pool:

        for strategy_number, (_, strategy_row) in enumerate(
            strategies.iterrows(),
            start=1
        ):
            strategy_name = str(strategy_row["strategy_id"])
            scalar_levers, di_levers = parse_strategy(strategy_row)
            strategy_start = time.time()

            print("\n" + "-" * 100)
            print(
                f"Starting {strategy_name} "
                f"({strategy_number}/{total_strategies})"
            )
            print(
                f"Evaluating {futures_per_strategy} futures "
                f"using {n_workers} processes..."
            )
            print("-" * 100)

            tasks = [
                (
                    strategy_name,
                    scalar_levers,
                    di_levers,
                    future
                )
                for future in future_records
            ]

            strategy_results = []

            for completed_strategy, result in enumerate(
                pool.imap_unordered(
                    _diagnostic_worker_wrapper,
                    tasks,
                    chunksize=1
                ),
                start=1
            ):
                strategy_results.append(result)
                completed_total += 1

                now = time.time()
                strategy_elapsed = now - strategy_start
                total_elapsed = now - start_time

                avg_wall_seconds_per_run = (
                    total_elapsed / completed_total
                )

                remaining_runs = total_runs - completed_total
                eta_seconds = (
                    avg_wall_seconds_per_run * remaining_runs
                )

                if (
                    completed_strategy == 1
                    or completed_strategy % 10 == 0
                    or completed_strategy == futures_per_strategy
                ):
                    strategy_percent = (
                        100.0
                        * completed_strategy
                        / futures_per_strategy
                    )

                    overall_percent = (
                        100.0
                        * completed_total
                        / total_runs
                    )

                    print(
                        f"{strategy_name}: "
                        f"{completed_strategy}/{futures_per_strategy} "
                        f"({strategy_percent:.1f}%) | "
                        f"Overall: {completed_total}/{total_runs} "
                        f"({overall_percent:.1f}%) | "
                        f"Strategy elapsed={strategy_elapsed/60:.2f} min | "
                        f"Total elapsed={total_elapsed/60:.2f} min | "
                        f"ETA={eta_seconds/60:.2f} min",
                        flush=True
                    )

            all_results.extend(strategy_results)

            strategy_elapsed = time.time() - strategy_start

            print(
                f"{strategy_name} COMPLETE | "
                f"{len(strategy_results)} futures | "
                f"runtime={strategy_elapsed/60:.2f} min",
                flush=True
            )

    total_elapsed = time.time() - start_time

    print("\n" + "=" * 100)
    print("ALL DIAGNOSTIC SIMULATIONS COMPLETE")
    print("=" * 100)
    print(f"Total simulations : {len(all_results)}")
    print(f"Worker processes  : {n_workers}")
    print(
        f"Total runtime     : {total_elapsed/60:.2f} minutes "
        f"({total_elapsed/3600:.2f} hours)"
    )
    print("=" * 100)

    return all_results, total_elapsed


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    N_WORKERS = 14
    overall_start_time = time.time()

    print("\n" + "=" * 100)
    print("STARTING REPRESENTATIVE STRATEGY ANALYSIS")
    print("=" * 100)
    print(f"Using {N_WORKERS} multiprocessing workers")

    if not DESIGN_FUTURES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find design futures:\n{DESIGN_FUTURES_FILE}"
        )

    design_futures = pd.read_csv(DESIGN_FUTURES_FILE)

    print(
        f"Design futures loaded: {len(design_futures):,}"
    )

    if len(design_futures) != 300:
        print(
            f"WARNING: Expected 300 futures, "
            f"but found {len(design_futures)}."
        )

    if not SELECTED_STRATEGIES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find representative strategies:\n"
            f"{SELECTED_STRATEGIES_FILE}"
        )

    strategies = pd.read_csv(SELECTED_STRATEGIES_FILE)

    if "strategy_id" not in strategies.columns:
        raise ValueError(
            "Selected strategy file does not contain 'strategy_id'."
        )

    strategy_order = ["S1", "S2", "S3", "S4", "S5"]

    strategies = strategies[
        strategies["strategy_id"].isin(strategy_order)
    ].copy()

    strategies["strategy_id"] = pd.Categorical(
        strategies["strategy_id"],
        categories=strategy_order,
        ordered=True
    )

    strategies = (
        strategies
        .sort_values("strategy_id")
        .reset_index(drop=True)
    )

    if len(strategies) != 5:
        raise ValueError(
            f"Expected 5 representative strategies, "
            f"found {len(strategies)}."
        )

    missing_columns = [
        col
        for col in DECISION_NAMES
        if col not in strategies.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing decision columns: {missing_columns}"
        )

    for col in ["irr_mean", "yld_mean"]:
        if col not in strategies.columns:
            raise ValueError(
                f"Missing '{col}' column required for validation."
            )

    print("\nRepresentative strategies loaded:")
    print(
        strategies[
            ["strategy_id", "irr_mean", "yld_mean"]
        ].to_string(index=False)
    )

    all_results, simulation_elapsed = run_parallel_diagnostics(
        strategies=strategies,
        design_futures=design_futures,
        n_workers=N_WORKERS
    )

    df_future = pd.DataFrame(all_results)

    df_future["Strategy"] = pd.Categorical(
        df_future["Strategy"],
        categories=strategy_order,
        ordered=True
    )

    df_future = (
        df_future
        .sort_values(["Strategy", "future_id"])
        .reset_index(drop=True)
    )

    expected_results = len(strategies) * len(design_futures)

    if len(df_future) != expected_results:
        raise RuntimeError(
            f"Expected {expected_results} results, "
            f"received {len(df_future)}."
        )

    future_file = OUTPUT_DIR / "irrigation_by_future.csv"
    df_future.to_csv(future_file, index=False)

    print(f"\nSaved future-level results:\n{future_file}")




    main_variables = [
        "annual_irrigation_mm",
        "annual_yield_Mg_ha",
        "annual_routine_mm",
        "annual_force_mm",
        "annual_survival_mm",
        "annual_emergency_mm",
    ]

    summary_rows = []

    for strategy, group in df_future.groupby(
        "Strategy",
        sort=False,
        observed=True
    ):
        row = {"Strategy": str(strategy)}

        for variable in main_variables:
            row[f"{variable}_mean"] = group[variable].mean()
            row[f"{variable}_sd"] = group[variable].std(ddof=1)

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    summary_file = (
        OUTPUT_DIR /
        "irrigation_summary_by_strategy.csv"
    )

    df_summary.to_csv(summary_file, index=False)
    print(f"Saved strategy summary:\n{summary_file}")




    original_performance = strategies[
        ["strategy_id", "irr_mean", "yld_mean"]
    ].copy()

    original_performance["Original_Irrigation_mm_yr"] = (
        original_performance["irr_mean"] / 5.0
    )

    original_performance["Original_Yield_Mg_ha_yr"] = (
        original_performance["yld_mean"] / 5.0
    )

    original_performance = original_performance.rename(
        columns={"strategy_id": "Strategy"}
    )

    original_performance["Strategy"] = (
        original_performance["Strategy"].astype(str)
    )

    validation = df_summary.merge(
        original_performance[
            [
                "Strategy",
                "Original_Irrigation_mm_yr",
                "Original_Yield_Mg_ha_yr"
            ]
        ],
        on="Strategy",
        how="left"
    )

    validation["Irrigation_difference_mm_yr"] = (
        validation["annual_irrigation_mm_mean"]
        - validation["Original_Irrigation_mm_yr"]
    )

    validation["Yield_difference_Mg_ha_yr"] = (
        validation["annual_yield_Mg_ha_mean"]
        - validation["Original_Yield_Mg_ha_yr"]
    )

    validation_file = (
        OUTPUT_DIR /
        "validation_against_original_optimization.csv"
    )

    validation.to_csv(validation_file, index=False)

    print("\n" + "=" * 100)
    print("VALIDATION AGAINST ORIGINAL NSGA-II RESULTS")
    print("=" * 100)

    print(
        validation[
            [
                "Strategy",
                "Original_Irrigation_mm_yr",
                "annual_irrigation_mm_mean",
                "Irrigation_difference_mm_yr",
                "Original_Yield_Mg_ha_yr",
                "annual_yield_Mg_ha_mean",
                "Yield_difference_Mg_ha_yr"
            ]
        ]
        .round(4)
        .to_string(index=False)
    )



    type_rows = []

    for strategy, group in df_future.groupby(
        "Strategy",
        sort=False,
        observed=True
    ):
        type_rows.append({
            "Strategy": str(strategy),
            "Total_mean_mm_yr":
                group["annual_irrigation_mm"].mean(),
            "Total_SD_mm_yr":
                group["annual_irrigation_mm"].std(ddof=1),
            "Routine_mean_mm_yr":
                group["annual_routine_mm"].mean(),
            "Routine_SD_mm_yr":
                group["annual_routine_mm"].std(ddof=1),
            "Force_mean_mm_yr":
                group["annual_force_mm"].mean(),
            "Force_SD_mm_yr":
                group["annual_force_mm"].std(ddof=1),
            "Survival_mean_mm_yr":
                group["annual_survival_mm"].mean(),
            "Survival_SD_mm_yr":
                group["annual_survival_mm"].std(ddof=1),
            "Emergency_mean_mm_yr":
                group["annual_emergency_mm"].mean(),
            "Emergency_SD_mm_yr":
                group["annual_emergency_mm"].std(ddof=1),
        })

    df_type = pd.DataFrame(type_rows)

    type_file = OUTPUT_DIR / "irrigation_type_summary.csv"
    df_type.to_csv(type_file, index=False)



    periods = [
        "establishment",
        "winter",
        "spring",
        "summer",
        "fall"
    ]

    season_rows = []

    for strategy, group in df_future.groupby(
        "Strategy",
        sort=False,
        observed=True
    ):
        row = {"Strategy": str(strategy)}

        for period in periods:
            col = f"{period}_total_mm_yr"

            row[f"{period.capitalize()}_mean_mm_yr"] = (
                group[col].mean()
            )

            row[f"{period.capitalize()}_SD_mm_yr"] = (
                group[col].std(ddof=1)
            )

        season_rows.append(row)

    df_season = pd.DataFrame(season_rows)

    season_file = (
        OUTPUT_DIR /
        "seasonal_irrigation_summary.csv"
    )

    df_season.to_csv(season_file, index=False)



    season_type_rows = []

    for strategy, group in df_future.groupby(
        "Strategy",
        sort=False,
        observed=True
    ):
        for period in periods:

            row = {
                "Strategy": str(strategy),
                "Period": period.capitalize()
            }

            for irr_type in [
                "total",
                "routine",
                "force",
                "survival"
            ]:
                col = f"{period}_{irr_type}_mm_yr"

                row[f"{irr_type.capitalize()}_mean_mm_yr"] = (
                    group[col].mean()
                )

                row[f"{irr_type.capitalize()}_SD_mm_yr"] = (
                    group[col].std(ddof=1)
                )

            season_type_rows.append(row)

    df_season_type = pd.DataFrame(season_type_rows)

    season_type_file = (
        OUTPUT_DIR /
        "seasonal_routine_survival_summary.csv"
    )

    df_season_type.to_csv(
        season_type_file,
        index=False
    )

    mass_balance = (
        df_future
        .groupby(
            "Strategy",
            sort=False,
            observed=True
        )["mass_balance_error_mm_5yr"]
        .agg(["mean", "max", "min"])
        .reset_index()
    )

    mass_balance["Strategy"] = (
        mass_balance["Strategy"].astype(str)
    )

    mass_balance.rename(
        columns={
            "mean": "Mean_error_mm_5yr",
            "max": "Maximum_error_mm_5yr",
            "min": "Minimum_error_mm_5yr",
        },
        inplace=True
    )

    mass_balance_file = (
        OUTPUT_DIR /
        "irrigation_mass_balance_check.csv"
    )

    mass_balance.to_csv(
        mass_balance_file,
        index=False
    )


    combined_rows = []

    for strategy, group in df_future.groupby(
        "Strategy",
        sort=False,
        observed=True
    ):
        row = {
            "Strategy": str(strategy),
            "Annual_Irrigation_Mean_mm_yr":
                group["annual_irrigation_mm"].mean(),
            "Annual_Irrigation_SD_mm_yr":
                group["annual_irrigation_mm"].std(ddof=1),
            "Annual_Yield_Mean_Mg_ha_yr":
                group["annual_yield_Mg_ha"].mean(),
            "Annual_Yield_SD_Mg_ha_yr":
                group["annual_yield_Mg_ha"].std(ddof=1),
            "Routine_Mean_mm_yr":
                group["annual_routine_mm"].mean(),
            "Force_Mean_mm_yr":
                group["annual_force_mm"].mean(),
            "Survival_Mean_mm_yr":
                group["annual_survival_mm"].mean(),
            "Emergency_Mean_mm_yr":
                group["annual_emergency_mm"].mean(),
        }

        for period in periods:
            row[f"{period.capitalize()}_Mean_mm_yr"] = (
                group[f"{period}_total_mm_yr"].mean()
            )

        combined_rows.append(row)

    df_combined = pd.DataFrame(combined_rows)

    combined_file = (
        OUTPUT_DIR /
        "representative_strategy_full_summary.csv"
    )

    df_combined.to_csv(
        combined_file,
        index=False
    )


    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 2)

    print("\n" + "=" * 100)
    print("IRRIGATION TYPE SUMMARY")
    print("=" * 100)
    print(df_type.round(2).to_string(index=False))

    print("\n" + "=" * 100)
    print("SEASONAL IRRIGATION SUMMARY")
    print("=" * 100)
    print(df_season.round(2).to_string(index=False))

    print("\n" + "=" * 100)
    print("IRRIGATION COMPONENT BALANCE CHECK")
    print("=" * 100)
    print(mass_balance.round(10).to_string(index=False))

    elapsed = time.time() - overall_start_time

    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(f"Total simulations: {len(df_future):,}")
    print(
        f"Simulation runtime: "
        f"{simulation_elapsed/60:.2f} minutes"
    )
    print(
        f"Total script runtime: "
        f"{elapsed/60:.2f} minutes"
    )
    print(f"\nResults saved to:\n{OUTPUT_DIR}")
    print("\nFiles:")

    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")