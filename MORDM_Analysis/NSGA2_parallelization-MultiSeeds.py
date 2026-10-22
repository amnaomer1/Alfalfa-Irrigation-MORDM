"""
NSGA-II Optimization of Alfalfa Irrigation and Yield (Multiple Seeds)

Author: Amna Omer

File Description
----------------
Runs bi-objective NSGA-II optimization using three random seeds
to assess the consistency of Pareto solutions across independent
optimization runs.

Inputs
------
- Daily ERA5 climate data for Imperial County (2000–2023).
- Alfalfa_Irrigation_Model.py.
- 300 Latin hypercube sampled states of the world.

Outputs
-------
- Shared sampled design futures.
- Population and Pareto solutions for each generation and seed.
- Combined population and Pareto histories for each seed.
- Final Pareto solutions and archives for each seed.
- Combined final Pareto solutions across seeds.
- Optimization run metadata.

Execution Requirements
----------------------
Requires numpy, pandas, scipy, pymoo, and the core alfalfa model.
Each optimization run uses 14 parallel worker processes and is
recommended for a multicore computing system.

Running the Script
------------------
From the MORDM_Analysis directory:

python NSGA2_parallelization-MultiSeeds.py
"""


# %% Imports
import os
import sys
import time
import traceback
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# prevent BLAS/OpenMP oversubscription before importing numpy/scipy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import qmc

# pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.parallelization.starmap import StarmapParallelization

# %% Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

OUTPUT_DIR = BASE_DIR / "NSGA2_K300_P100_G100_MultiSeeds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT_ROOT))

# %% Climate data
climate_csv_path = PROJECT_ROOT / "Input" / "ImperialCounty_ERA5_DailyClimate_2000_2023.csv"
climate_df_big = pd.read_csv(climate_csv_path, parse_dates=["Date"])
print(f" climate_df loaded: {len(climate_df_big)} rows")

# %% Import simulator
from Alfalfa_Irrigation_Model import full_alfalfa_simulation

# =============================================================================
# 1. FIXED MODEL STRUCTURE
# =============================================================================

max_cuttings = 12
num_years = 5
vector_len = max_cuttings * num_years

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
assert len(Kmitsch_base) == vector_len
assert len(ky_base) == vector_len

# =============================================================================
# 2. UNCERTAINTY SPACE
# =============================================================================

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


def soil_params_from_row(row: dict) -> dict:
    return {
        "theta_s": float(row["theta_s"]),
        "theta_r": float(row["theta_r"]),
        "n": float(row["n"]),
        "K_sat": float(row["K_sat"]),
        "pwp": float(row["pwp"]),
        "AWC": float(row["AWC"]),
    }


def repair_soil_bounds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["theta_s"] = np.maximum(df["theta_s"], df["theta_r"] + 0.05)

    df["pwp"] = np.minimum(
        np.maximum(df["pwp"], df["theta_r"] + 0.01),
        df["theta_s"] - 0.01
    )

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


# =============================================================================
# 3. CLIMATE WINDOWS
# =============================================================================

def list_climate_windows(climate_df: pd.DataFrame, window_years: int = 5) -> List[int]:
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


def map_u_to_climate_start(u: float, starts: List[int]) -> int:
    u = float(np.clip(u, 0.0, 0.999999))
    idx = int(np.floor(u * len(starts)))
    return int(starts[idx])


CLIMATE_WINDOW_STARTS = list_climate_windows(climate_df_big, window_years=5)

# per-process cache
CLIMATE_CACHE = {}

# =============================================================================
# 4. DECISION VARIABLES
# =============================================================================

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

XL = np.array([
    245, 0.10, 120, 0.20, 0.50, 0.00, 15.0, 0.50, 5.0, 1, 0,
    0.00, 0.00, 0.00, 0.00, 0.00
], dtype=float)

XU = np.array([
    320, 2.00, 200, 0.80, 1.00, 0.30, 70.0, 0.90, 70.0, 5, 7,
    1.00, 1.00, 1.00, 1.00, 1.00
], dtype=float)


# =============================================================================
# 5. DI HELPERS
# =============================================================================


def build_cutting_param_df(
    years: List[int],
    wp_vector: np.ndarray,
    ky_vector: np.ndarray,
    kmitsch_vector: np.ndarray,
) -> pd.DataFrame:
    Y = len(years)
    C = int(max_cuttings)
    needed = Y * C

    year_arr = np.repeat(np.array(years, dtype=int), C)
    cut_arr = np.tile(np.arange(1, C + 1, dtype=int), Y)

    cutting_param_df = pd.DataFrame({
        "Year": year_arr,
        "Cutting": cut_arr,
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

    cutting_param_df = pd.concat([cut0, cutting_param_df], ignore_index=True)
    return cutting_param_df

# =============================================================================
# 6. CORE SIMULATOR RUN
# =============================================================================

def run_full_simulation_with_di_levers(
    scalar_levers: Dict[str, float],
    di_levers: Dict[str, float],
    future_row: dict
):
    global CLIMATE_CACHE

    if di_levers is None:
        raise ValueError("di_levers cannot be None in date-based DI mode")

    planting_day_doy = int(round(scalar_levers["planting_doy"]))
    yield_cutoff = float(scalar_levers["yield_cutoff"])
    establishment_days = int(round(scalar_levers["establishment_days"]))
    mad = float(scalar_levers["MAD"])
    refill_target = float(scalar_levers["refill_target"])
    survival_fraction = float(scalar_levers["survival_fraction"])
    max_irrigation = float(scalar_levers["max_irrigation"])
    init_smt = float(scalar_levers["initial_smt"])
    min_infiltration = float(scalar_levers["min_infiltration"])
    pre_harvest_days = int(round(scalar_levers["pre_harvest_no_irrigation_days"]))
    post_cutting_days = int(round(scalar_levers["post_cutting_buffer_days"]))


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
    kmitsch_vector = Kmitsch_base * float(future_row["Kmitsch_scale"])

    root_depth_mm = int(round(future_row["root_depth_mm"]))
    p_depletion = float(np.clip(future_row["p_depletion"], 0.1, 0.8))

    start_year = map_u_to_climate_start(future_row["climate_window_u"], CLIMATE_WINDOW_STARTS)

    if start_year in CLIMATE_CACHE:
        climate_df, years = CLIMATE_CACHE[start_year]
    else:
        climate_df, years = get_climate_window(climate_df_big, start_year, window_years=5)
        CLIMATE_CACHE[start_year] = (climate_df, years)


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
        refill_target_fraction=refill_target,
        root_depth_mm=root_depth_mm,
        awc_mm_per_mm=soil_params["AWC"],
        p_depletion=p_depletion,
        MAD=mad,
        initial_smt_fraction=init_smt,
        temperature_dormancy_threshold=temperature_dormancy_threshold,
        pre_harvest_no_irrigation_days=pre_harvest_days,
        post_cutting_buffer_days=post_cutting_days,
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
    return df_result, df_cuttings, df_yearly, years


# =============================================================================
# 7. POLICY EVALUATION
# =============================================================================

def parse_decision_vector(x: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    x = np.asarray(x, dtype=float)

    scalar_levers = {
        "planting_doy": int(round(x[0])),
        "yield_cutoff": float(x[1]),
        "establishment_days": int(round(x[2])),
        "MAD": float(x[3]),
        "refill_target": float(x[4]),
        "survival_fraction": float(x[5]),
        "max_irrigation": float(x[6]),
        "initial_smt": float(x[7]),
        "min_infiltration": float(x[8]),
        "pre_harvest_no_irrigation_days": int(round(x[9])),
        "post_cutting_buffer_days": int(round(x[10])),
    }

    di_levers = {
        "DI_establishment": float(x[11]),
        "DI_winter": float(x[12]),
        "DI_spring": float(x[13]),
        "DI_summer": float(x[14]),
        "DI_fall": float(x[15]),
    }

    scalar_levers["planting_doy"] = int(np.clip(scalar_levers["planting_doy"], 245, 320))
    scalar_levers["establishment_days"] = int(np.clip(scalar_levers["establishment_days"], 120, 200))
    scalar_levers["pre_harvest_no_irrigation_days"] = int(np.clip(scalar_levers["pre_harvest_no_irrigation_days"], 1, 5))
    scalar_levers["post_cutting_buffer_days"] = int(np.clip(scalar_levers["post_cutting_buffer_days"], 0, 7))
    
    
    return scalar_levers, di_levers


def evaluate_one_future_one_pass_di(
    scalar_levers: Dict[str, float],
    di_levers: Dict[str, float],
    future_row: dict
) -> Tuple[float, float]:
    try:
        df_result, _, df_yearly, _ = run_full_simulation_with_di_levers(
            scalar_levers=scalar_levers,
            di_levers=di_levers,
            future_row=future_row
        )

        irrigation_mm = float(df_result["Irrigation_mm"].sum())
        yield_total_mg_ha = float(df_yearly["Total_Yield_Mg_ha_Mitsch"].sum())

        return irrigation_mm, yield_total_mg_ha

    except Exception as e:
        print(f"Evaluation error in future_id={future_row.get('future_id', 'NA')}: {e}")
        traceback.print_exc()
        return 9999.0, 0.0


def evaluate_policy_across_futures(
    x: np.ndarray,
    design_futures: pd.DataFrame
) -> Tuple[float, float]:
    scalar_levers, di_levers = parse_decision_vector(x)

    irr_vals = []
    yld_vals = []

    for future in design_futures.to_dict("records"):
        irr, yld = evaluate_one_future_one_pass_di(
            scalar_levers=scalar_levers,
            di_levers=di_levers,
            future_row=future
        )
        irr_vals.append(float(irr))
        yld_vals.append(float(yld))

    mean_irr = float(np.mean(irr_vals))
    mean_yld = float(np.mean(yld_vals))

    return mean_irr, mean_yld


# =============================================================================
# 8. PYMOO PROBLEM
# =============================================================================

class AlfalfaBiObjectiveProblem(ElementwiseProblem):

    def __init__(
        self,
        design_futures: pd.DataFrame,
        elementwise_runner=None
    ):
        super().__init__(
            n_var=len(DECISION_NAMES),
            n_obj=2,
            n_ieq_constr=2,
            xl=XL,
            xu=XU,
            elementwise_runner=elementwise_runner,
        )

        self.design_futures = design_futures


    def _evaluate(self, x, out, *args, **kwargs):

        mean_irr, mean_yld = evaluate_policy_across_futures(
            x=x,
            design_futures=self.design_futures
        )

        # Objectives:
        # minimize irrigation and maximize yield
        out["F"] = np.array(
            [mean_irr, -mean_yld],
            dtype=float
        )


        MAD = float(x[3])
        refill_target = float(x[4])
        survival_fraction = float(x[5])

        # Refill target must be at least 0.05 above MAD trigger
        g_refill = (
            (1.0 - MAD)
            - refill_target
            + 0.05
        )

        # Survival threshold must be at least 0.05 below MAD trigger
        g_survival = (
            survival_fraction
            - (1.0 - MAD)
            + 0.05
        )

        out["G"] = np.array(
            [
                g_refill,
                g_survival
            ],
            dtype=float
        )

# =============================================================================
# 9. CALLBACK
# =============================================================================

class SaveParetoAndPopulationEveryGeneration(Callback):
    def __init__(self, pareto_dir: Path, population_dir: Path, total_generations: int):
        super().__init__()
        self.pareto_dir = Path(pareto_dir)
        self.population_dir = Path(population_dir)
        self.pareto_records = []
        self.population_records = []
        self.total_generations = int(total_generations)
        self.start_time = time.time()
        self.last_time = self.start_time

    def notify(self, algorithm):
        gen = int(algorithm.n_gen)

        now = time.time()
        elapsed = now - self.start_time
        last_gen_time = now - self.last_time
        self.last_time = now

        avg_time_per_gen = (elapsed - last_gen_time) / (gen - 1) if gen > 1 else last_gen_time
        remaining_gens = self.total_generations - gen
        eta_seconds = remaining_gens * avg_time_per_gen if gen > 0 else 0.0


        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        G = algorithm.pop.get("G")
        
        # A solution is feasible when all constraints are <= 0
        feasible_mask = np.all(G <= 1e-12, axis=1)

        # -------------------------
        # Save full population
        # -------------------------
        pop_rows = []
        for i in range(len(X)):
        
            row = {
                "generation": gen,
                "solution_id": i
            }
        
            for j, name in enumerate(DECISION_NAMES):
                row[name] = float(X[i, j])
        
            # Constraint diagnostics
            row["g_refill"] = float(G[i, 0])
            row["g_survival"] = float(G[i, 1])
            row["feasible"] = bool(feasible_mask[i])
        
            # Objectives
            row["irr_mean"] = float(F[i, 0])
            row["neg_yld_mean"] = float(F[i, 1])
            row["yld_mean"] = float(-F[i, 1])
        
            pop_rows.append(row)

        df_pop = pd.DataFrame(pop_rows)

        pop_file = self.population_dir / f"population_gen_{gen:03d}.csv"
        df_pop.to_csv(pop_file, index=False)
        self.population_records.append(df_pop)


        # -------------------------
        # Save FEASIBLE Pareto front from this generation only
        # -------------------------
        
        feasible_idx = np.where(feasible_mask)[0]
        
        if len(feasible_idx) > 0:
        
            F_feasible = F[feasible_idx]
        
            nd_local = NonDominatedSorting().do(
                F_feasible,
                only_non_dominated_front=True
            )
        
            nd_idx = feasible_idx[nd_local]
        
            df_pareto = df_pop.iloc[nd_idx].copy()
        
        else:
        
            df_pareto = df_pop.iloc[0:0].copy()

        pareto_file = self.pareto_dir / f"pareto_gen_{gen:03d}.csv"
        df_pareto.to_csv(pareto_file, index=False)
        self.pareto_records.append(df_pareto)

        print(
            f"[Gen {gen}/{self.total_generations}] "
            f"elapsed={elapsed/3600:.2f}h | "
            f"last_gen={last_gen_time/60:.1f}m | "
            f"avg_gen={avg_time_per_gen/60:.1f}m | "
            f"ETA={eta_seconds/3600:.2f}h | "
            f"Population={len(df_pop)} | "
            f"Pareto solutions={len(df_pareto)}"
        )
        print(f" Saved population: {pop_file.name}")
        print(f" Saved Pareto front: {pareto_file.name}")

# =============================================================================
# 10. MAIN RUN
# =============================================================================

def run_nsga2_alfalfa_optimization(
    design_future_seed: int = 12345,
    K: int = 300,
    nsga_seed: int = 101,
    pop_size: int = 100,
    n_gen: int = 100,
    n_workers: int = 10,
):
    start_time = time.time()

    # -----------------------------------------------------
    # Separate output folder for this NSGA-II seed
    # -----------------------------------------------------
    seed_dir = OUTPUT_DIR / f"seed_{nsga_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    pareto_dir = seed_dir / "pareto_by_generation"
    pareto_dir.mkdir(parents=True, exist_ok=True)

    population_dir = seed_dir / "population_by_generation"
    population_dir.mkdir(parents=True, exist_ok=True)

    design_futures = generate_design_futures(
        k=K,
        seed=design_future_seed
    )
    design_futures_file = OUTPUT_DIR / f"design_futures_k{K}_seed_{design_future_seed}.csv"
    design_futures.to_csv(design_futures_file, index=False)
    print(f" Saved design futures: {design_futures_file}")

    ctx = mp.get_context("fork")

    with ctx.Pool(processes=n_workers) as pool:
        runner = StarmapParallelization(pool.starmap)

        problem = AlfalfaBiObjectiveProblem(
            design_futures=design_futures,
            elementwise_runner=runner
        )

        algorithm = NSGA2(
            pop_size=pop_size,
            eliminate_duplicates=True,
        )

        callback = SaveParetoAndPopulationEveryGeneration(
            pareto_dir=pareto_dir,
            population_dir=population_dir,
            total_generations=n_gen
        )
        res = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=("n_gen", n_gen),
            seed=nsga_seed,
            save_history=False,
            verbose=True,
            callback=callback,
        )

    df_all_pareto = (
        pd.concat(callback.pareto_records, ignore_index=True)
        if len(callback.pareto_records) > 0 else pd.DataFrame()
    )
    
    df_all_population = (
        pd.concat(callback.population_records, ignore_index=True)
        if len(callback.population_records) > 0 else pd.DataFrame()
    )

    all_pareto_file = seed_dir / "pareto_all_generations.csv"
    all_population_file = seed_dir / "population_all_generations.csv"
    df_all_population.to_csv(all_population_file, index=False)
    print(f"  Saved combined population history: {all_population_file}")
    
    # Build final Pareto archive from all population solutions

    if len(df_all_population) > 0:
    
        # Keep only feasible solutions
        df_archive_input = (
            df_all_population[
                df_all_population["feasible"] == True
            ]
            .drop_duplicates(
                subset=DECISION_NAMES + ["irr_mean", "yld_mean"]
            )
            .reset_index(drop=True)
        )
    
        if len(df_archive_input) > 0:
    
            F_all = df_archive_input[
                ["irr_mean", "neg_yld_mean"]
            ].values
    
            nd_idx_all = NonDominatedSorting().do(
                F_all,
                only_non_dominated_front=True
            )
    
            df_final_archive = (
                df_archive_input.iloc[nd_idx_all].copy()
            )
    
        else:
    
            df_final_archive = pd.DataFrame()
    
        archive_file = (
            seed_dir /
            "final_pareto_archive_from_all_population.csv"
        )
    
        df_final_archive.to_csv(
            archive_file,
            index=False
        )
    
        print(
            f"  Saved final Pareto archive from all populations: "
            f"{archive_file}"
        )
    
        print(
            f"  Final archive size: "
            f"{len(df_final_archive)}"
        )
    df_all_pareto.to_csv(all_pareto_file, index=False)
    print(f" Saved combined Pareto history: {all_pareto_file}")

    final_X = res.X
    final_F = res.F

    if final_X is not None and final_F is not None:
        if final_X.ndim == 1:
            final_X = final_X.reshape(1, -1)
            final_F = final_F.reshape(1, -1)

        rows = []
        for i in range(len(final_X)):
            row = {
                "seed": nsga_seed,
                "solution_id": i
            }
            for j, name in enumerate(DECISION_NAMES):
                row[name] = float(final_X[i, j])

            row["irr_mean"] = float(final_F[i, 0])
            row["neg_yld_mean"] = float(final_F[i, 1])
            row["yld_mean"] = float(-final_F[i, 1])
            rows.append(row)

        df_final = pd.DataFrame(rows)
    else:
        df_final = pd.DataFrame()

    final_file = seed_dir / "final_pareto_generation.csv"
    df_final.to_csv(final_file, index=False)
    print(f" Saved final Pareto set: {final_file}")

    elapsed_min = (time.time() - start_time) / 60.0
    meta = pd.DataFrame([{
        "design_future_seed": design_future_seed,
        "K": K,
        "nsga_seed": nsga_seed,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "n_workers": n_workers,
        "n_variables": len(DECISION_NAMES),
        "elapsed_minutes": elapsed_min,
    }])
    meta_file = seed_dir / "run_metadata.csv"
    meta.to_csv(meta_file, index=False)

    print(f" Saved metadata: {meta_file}")
    print(f" Total runtime: {elapsed_min:.2f} minutes")

    return res, df_all_pareto, df_final

# =============================================================================
# 11. RUN MULTIPLE NSGA-II SEEDS
# =============================================================================

if __name__ == "__main__":

    DESIGN_FUTURE_SEED = 12345

    K = 300
    POP_SIZE = 100
    N_GEN = 100
    N_WORKERS = 14

    # Seed 101 is already being run separately
    NSGA_SEEDS = [102, 103, 104]

    all_final_pareto = []

    for nsga_seed in NSGA_SEEDS:

        print("\n" + "=" * 80)
        print(f"STARTING NSGA-II SEED {nsga_seed}")
        print("=" * 80)

        res, df_all_pareto, df_final = run_nsga2_alfalfa_optimization(
            design_future_seed=DESIGN_FUTURE_SEED,
            K=K,
            nsga_seed=nsga_seed,
            pop_size=POP_SIZE,
            n_gen=N_GEN,
            n_workers=N_WORKERS,
        )

        if len(df_final) > 0:
            all_final_pareto.append(df_final)

        print("\n" + "=" * 80)
        print(f"COMPLETED NSGA-II SEED {nsga_seed}")
        print("=" * 80)

    # -----------------------------------------------------
    # Combine final Pareto fronts from seeds 102, 103, 104
    # -----------------------------------------------------
    if len(all_final_pareto) > 0:

        df_final_all_seeds = pd.concat(
            all_final_pareto,
            ignore_index=True
        )

        final_all_file = (
            OUTPUT_DIR /
            "final_pareto_all_seeds.csv"
        )

        df_final_all_seeds.to_csv(
            final_all_file,
            index=False
        )

        print(
            f" Saved combined final Pareto fronts: "
            f"{final_all_file}"
        )