"""
Sobol Global Sensitivity Analysis of Alfalfa Irrigation Strategies

Author: Amna Omer

File Description
----------------
Performs variance-based global sensitivity analysis for five
representative Pareto irrigation strategies across 13 uncertain
climate, soil, and crop-response parameters. Uses a common Sobol
sampling design to estimate first-order, total-order, and
second-order sensitivity indices for cumulative irrigation
and alfalfa yield.

Inputs
------
- NSGA2_parallelization_OneSeed_100Gen.py.
- Strategies_Performance/selected_5_representative_strategies.csv.
- Daily ERA5 climate data and the core alfalfa irrigation model,
  accessed through the optimization script.

Outputs
-------
- Common Sobol sample of uncertain parameters.
- Future-level irrigation and yield results for each strategy.
- First-order and total-order Sobol sensitivity indices.
- Second-order interaction indices and confidence intervals.
- Strategy-level and combined analysis summaries.

Execution Requirements
----------------------
Requires numpy, pandas, scipy, SALib, tqdm, and dependencies
of the optimization script. Uses 14 parallel worker processes
by default and is recommended for a multicore computing system.

Running the Script
------------------
From the MORDM_Analysis directory:

python Sobol_Analysis.py

The default base sample size is N = 1024. Other sample sizes
can be specified using --N, for example:

python Sobol_Analysis.py --N 2048 --out_dir Sobol_Results/Sobol_Results_N2048
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
# Paths / project import

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.append(str(PROJECT_ROOT))

from NSGA2_parallelization_OneSeed_100Gen import (
    DECISION_NAMES,
    parse_decision_vector,
    run_full_simulation_with_di_levers,
)


# Uncertainty bounds

UNCERTAINTY_BOUNDS = {
    "kc_scale": [0.7, 1.3],
    "temperature_dormancy_threshold": [1.0, 6.0],
    "p_depletion": [0.41, 0.66],
    "wp_scale": [0.7, 1.3],
    "Kmitsch_scale": [0.7, 1.3],
    "climate_window_u": [0.0, 1.0],

    # Raw independent soil uncertainty factors
    "AWC_raw": [0.11, 0.20],

    "root_depth_mm": [1000.0, 2000.0],
    "K_sat": [0.08, 11.0],

    "theta_s_raw": [0.40, 0.50],
    "theta_r_raw": [0.05, 0.30],

    "n": [1.06, 1.40],

    "pwp_raw": [0.04, 0.30],
}
UNC_KEYS = list(UNCERTAINTY_BOUNDS.keys())
D = len(UNC_KEYS)



def load_selected_strategies(strategy_file: Path) -> pd.DataFrame:
    """
    Load the five representative Pareto-optimal strategies selected
    through objective-space clustering.
    """
    if not strategy_file.exists():
        raise FileNotFoundError(
            "Selected strategy file was not found:\n"
            f"{strategy_file}"
        )

    strategies_df = pd.read_csv(strategy_file)
    required_cols = [
        "strategy_id",
        "strategy_label",
        *DECISION_NAMES,
    ]

    missing_cols = [
        col for col in required_cols
        if col not in strategies_df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Selected-strategy file is missing required columns: "
            f"{missing_cols}"
        )
    if len(strategies_df) != 5:
        raise ValueError(
            "Expected exactly five representative strategies, "
            f"but found {len(strategies_df)}."
        )

    # Ensure strategies are ordered S1 to S5
    strategies_df = strategies_df.copy()

    strategies_df["strategy_number"] = (
        strategies_df["strategy_id"]
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    strategies_df = (
        strategies_df
        .sort_values("strategy_number")
        .reset_index(drop=True)
    )

    return strategies_df

def transform_soil_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert independent raw Sobol soil factors into the physically
    valid soil parameters used by the simulation.

    This applies the same soil-repair rules used in the optimization,
    while retaining the original independent raw variables for the
    Sobol decomposition.
    """

    df = df.copy()

    df["theta_r"] = df["theta_r_raw"]

    # Same theta_s repair used in optimization
    df["theta_s"] = np.maximum(
        df["theta_s_raw"],
        df["theta_r"] + 0.05
    )

    # Same pwp repair used in optimization
    df["pwp"] = np.minimum(
        np.maximum(
            df["pwp_raw"],
            df["theta_r"] + 0.01
        ),
        df["theta_s"] - 0.01
    )

    # Same AWC repair used in optimization
    df["AWC"] = np.minimum(
        df["AWC_raw"],
        df["theta_s"] - df["pwp"]
    )

    df["AWC"] = np.maximum(
        df["AWC"],
        0.02
    )

    return df


_WORKER_SCALAR_LEVERS = None
_WORKER_DI_LEVERS = None


def initialize_worker(
    scalar_levers,
    di_levers,
):
    """
    Initialize one worker process with one fixed strategy.
    """
    global _WORKER_SCALAR_LEVERS
    global _WORKER_DI_LEVERS

    _WORKER_SCALAR_LEVERS = scalar_levers
    _WORKER_DI_LEVERS = di_levers


def evaluate_future_worker(future_row):
    """
    Evaluate one Sobol future for the fixed strategy assigned
    to this worker process.
    """

    try:

        df_result, df_cuttings, df_yearly, years = (
            run_full_simulation_with_di_levers(
                scalar_levers=_WORKER_SCALAR_LEVERS,
                di_levers=_WORKER_DI_LEVERS,
                future_row=future_row
            )
        )

        irrigation_mm = float(
            df_result["Irrigation_mm"].sum()
        )

        yield_total_mg_ha = float(
            df_yearly[
                "Total_Yield_Mg_ha_Mitsch"
            ].sum()
        )

        return {
            "future_id": int(
                future_row["future_id"]
            ),
            "irr_mm": irrigation_mm,
            "yld_total_mg_ha": yield_total_mg_ha,
            "ok": 1,
        }

    except Exception as e:

        print(
            f"FAILED future "
            f"{future_row['future_id']} | {e}"
        )

        traceback.print_exc()

        return {
            "future_id": int(
                future_row["future_id"]
            ),
            "irr_mm": np.nan,
            "yld_total_mg_ha": np.nan,
            "ok": 0,
        }

# Sobol problem + sampling

def build_sobol_problem():
    return {
        "num_vars": D,
        "names": list(UNC_KEYS),
        "bounds": [UNCERTAINTY_BOUNDS[k] for k in UNC_KEYS],
    }


def make_sobol_futures(problem, N: int, seed: int) -> pd.DataFrame:
    """
    SALib sobol sampler returns a Saltelli-style design of size:
        R = N * (2D + 2)  (when calc_second_order=True)
    """

    X = sobol_sample.sample(
        problem,
        N,
        calc_second_order=True,
        seed=seed
    )

    df = pd.DataFrame(
        X,
        columns=problem["names"]
    )

    df = transform_soil_parameters(df)

    # -------------------------------------------------
    # Verify transformed soil parameters
    # -------------------------------------------------

    if not np.all(df["theta_s"] > df["theta_r"]):
        raise RuntimeError(
            "Invalid soil sample: theta_s <= theta_r"
        )

    if not np.all(df["pwp"] > df["theta_r"]):
        raise RuntimeError(
            "Invalid soil sample: pwp <= theta_r"
        )

    if not np.all(df["pwp"] < df["theta_s"]):
        raise RuntimeError(
            "Invalid soil sample: pwp >= theta_s"
        )

    if not np.all(
        df["AWC"] <=
        (df["theta_s"] - df["pwp"]) + 1e-12
    ):
        raise RuntimeError(
            "Invalid soil sample: "
            "AWC > theta_s - pwp"
        )

    df.insert(
        0,
        "future_id",
        np.arange(len(df), dtype=int)
    )

    return df



# Sobol analysis 

def sobol_to_df(problem, Y: np.ndarray):
    Si = sobol_analyze.analyze(
        problem,
        Y,
        calc_second_order=True,
        print_to_console=False
    )

    df_main = pd.DataFrame({
        "param": problem["names"],
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    }).sort_values("ST", ascending=False).reset_index(drop=True)

    # Second-order matrix (interactions)
    S2 = Si.get("S2", None)
    S2_conf = Si.get("S2_conf", None)

    return df_main, S2, S2_conf

def save_S2(out_dir: Path, strategy_name: str, obj_name: str, N: int, seed: int, names, S2, S2_conf):

    if S2 is None:
        print(f" No S2 returned for {obj_name}")
        return

    S2m = np.array(S2, dtype=float)
    n = len(names)

    # Masks for upper/lower triangle excluding diagonal
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Count finite entries in each triangle
    lower_finite = int(np.isfinite(S2m[lower_mask]).sum())
    upper_finite = int(np.isfinite(S2m[upper_mask]).sum())
    total_pairs = n * (n - 1) // 2

    print(f"S2 finite pairs (lower): {lower_finite} / {total_pairs} | (upper): {upper_finite} / {total_pairs} for {obj_name}")

    # Choose the triangle that actually contains values
    if upper_finite > lower_finite:
        mask_keep = upper_mask
        triangle_used = "upper"
    else:
        mask_keep = lower_mask
        triangle_used = "lower"

    rows = []
    for i in range(n):
        for j in range(n):
            if mask_keep[i, j]:
                val = S2m[i, j]
                if np.isfinite(val):
                    rows.append({"param_i": names[i], "param_j": names[j], "S2": float(val)})

    print(f"S2 pairs saved ({triangle_used} triangle) for {obj_name}: {len(rows)} / {total_pairs}")

    if len(rows) == 0:
        print(f" No finite S2 pairs to save for {obj_name} at N={N}.")
        return

    s2_long = pd.DataFrame(rows).sort_values("S2", ascending=False)
    s2_path = out_dir / f"sobol_S2_{strategy_name}_{obj_name}_N{N}_seed{seed}.csv"
    s2_long.to_csv(s2_path, index=False)

    # Confidence intervals
    if S2_conf is not None:
        S2c = np.array(S2_conf, dtype=float)
        rowsc = []
        for i in range(n):
            for j in range(n):
                if mask_keep[i, j]:
                    val = S2c[i, j]
                    if np.isfinite(val):
                        rowsc.append({"param_i": names[i], "param_j": names[j], "S2_conf": float(val)})

        s2c_path = out_dir / f"sobol_S2conf_{strategy_name}_{obj_name}_N{N}_seed{seed}.csv"
        
        if len(rowsc) == 0:
            print(f" No finite S2_conf pairs to save for {obj_name} at N={N}.")
        else:
            pd.DataFrame(rowsc).to_csv(s2c_path, index=False)
            print(f" Saved S2 + S2_conf for {obj_name}:\n  - {s2_path}\n  - {s2c_path}")
    else:
        print(f" Saved S2 for {obj_name}:\n  - {s2_path}")

# Main

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--strategies_file",
        type=str,
        default=str(
            BASE_DIR
            / "Strategies_Performance"
            / "selected_5_representative_strategies.csv"
        ),
        help="CSV containing the five selected Pareto strategies"
    )

    ap.add_argument(
        "--N",
        type=int,
        default=1024,
        help=(
            "Sobol base sample size. Runs per strategy are "
            "N*(2D+2) when second-order indices are enabled."
        )
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Sobol sampler seed"
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=14,
        help="Number of parallel worker processes"
    )

    ap.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help=(
            "Number of Sobol simulations assigned to a worker "
            "at one time"
        )
    )


    # Sobol output directory
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(
            BASE_DIR / "Sobol_Results"
        ),
        help="Main output directory"
    )

    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    
    out_dir = out_dir / f"Sobol_Results_N{args.N}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nMain output directory:\n{out_dir}")

    # Load the five selected Pareto strategies
    
    strategy_file = Path(args.strategies_file)
    
    if not strategy_file.is_absolute():
        strategy_file = PROJECT_ROOT / strategy_file
    
    strategies_df = load_selected_strategies(
        strategy_file
    )
    
    print("\nLoaded five representative Pareto strategies:")
    
    display_cols = [
        "strategy_id",
        "strategy_label",
    ]
    
    if "irr_mean" in strategies_df.columns:
        display_cols.append("irr_mean")
    
    if "yld_mean" in strategies_df.columns:
        display_cols.append("yld_mean")
    
    print(
        strategies_df[display_cols].to_string(index=False)
    )
    
    problem = build_sobol_problem()
    
    futures_df = make_sobol_futures(
        problem,
        N=args.N,
        seed=args.seed
    )
    
    runs_per_strategy = len(futures_df)
    number_of_strategies = len(strategies_df)
    total_expected_runs = (
        runs_per_strategy * number_of_strategies
    )
    
    print("\nSobol design summary")
    print("-" * 60)
    print(f"Number of uncertainties D: {D}")
    print(f"Base sample size N: {args.N}")
    print(f"Runs per strategy: {runs_per_strategy:,}")
    print(f"Number of strategies: {number_of_strategies}")
    print(f"Total expected model runs: {total_expected_runs:,}")
    print(f"Parallel workers: {args.workers}")
    print(f"Parallel chunksize: {args.chunksize}")
    print("-" * 60)
    
    common_futures_path = (
        out_dir /
        f"sobol_common_futures_N{args.N}_seed{args.seed}.csv"
    )
    
    futures_df.to_csv(
        common_futures_path,
        index=False
    )
    
    print(
        f"\nSaved common Sobol sample:\n"
        f"{common_futures_path}"
    )

    all_strategy_summaries = []
    overall_start_time = time.perf_counter()

    for strategy_index, strategy_row in strategies_df.iterrows():

        strategy_id = str(
            strategy_row["strategy_id"]
        ).strip()

        strategy_label = str(
            strategy_row["strategy_label"]
        ).strip()

        x = np.array(
            [
                strategy_row[name]
                for name in DECISION_NAMES
            ],
            dtype=float
        )
        
        scalar_levers, di_levers = (
            parse_decision_vector(x)
        )

        strategy_out_dir = (
            out_dir /
            f"Sobol_{strategy_id}"
        )

        strategy_out_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\n" + "=" * 75)
        print(
            f"Starting strategy "
            f"{strategy_index + 1} of {len(strategies_df)}"
        )
        print(f"Strategy ID: {strategy_id}")
        print(f"Strategy label: {strategy_label}")

        if "irr_mean" in strategy_row.index:
            print(
                "Pareto-front mean irrigation: "
                f"{float(strategy_row['irr_mean']):.3f}"
            )

        if "yld_mean" in strategy_row.index:
            print(
                "Pareto-front mean yield: "
                f"{float(strategy_row['yld_mean']):.3f}"
            )

        print(
            f"Runs for this strategy: "
            f"{len(futures_df):,}"
        )

        print(
            f"Output folder:\n"
            f"{strategy_out_dir}"
        )

        print("=" * 75)

        strategy_inputs = pd.DataFrame(
            [{
                "strategy_id": strategy_id,
                "strategy_label": strategy_label,
                **{
                    name: float(strategy_row[name])
                    for name in DECISION_NAMES
                },
            }]
        )

        strategy_inputs_path = (
            strategy_out_dir /
            f"strategy_inputs_{strategy_id}.csv"
        )

        strategy_inputs.to_csv(
            strategy_inputs_path,
            index=False
        )

        print(
            f"\nSaved strategy inputs:\n"
            f"{strategy_inputs_path}"
        )

        strategy_start_time = time.perf_counter()

        rows = []
        failures = 0
        total_runs = len(futures_df)

        future_records = futures_df.to_dict("records")

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(
                scalar_levers,
                di_levers,
            ),
        ) as executor:

            parallel_results = executor.map(
                evaluate_future_worker,
                future_records,
                chunksize=args.chunksize,
            )

            pbar = tqdm(
                parallel_results,
                total=total_runs,
                desc=f"Sobol {strategy_id}",
                unit="run",
                dynamic_ncols=True,
            )

            for result in pbar:

                irr_mm = result["irr_mm"]
                yld_total = result["yld_total_mg_ha"]
                ok = result["ok"]

                valid_result = (
                    ok == 1
                    and np.isfinite(irr_mm)
                    and np.isfinite(yld_total)
                )

                if not valid_result:
                    failures += 1

                rows.append({
                    "future_id": result["future_id"],
                    "strategy_id": strategy_id,
                    "strategy_label": strategy_label,
                    "irr_mm": (
                        float(irr_mm)
                        if np.isfinite(irr_mm)
                        else np.nan
                    ),
                    "yld_total_mg_ha": (
                        float(yld_total)
                        if np.isfinite(yld_total)
                        else np.nan
                    ),
                    "ok": int(ok),
                })

                pbar.set_postfix({
                    "fail": failures,
                    "last_irr": (
                        round(irr_mm, 1)
                        if np.isfinite(irr_mm)
                        else "nan"
                    ),
                    "last_yld": (
                        round(yld_total, 1)
                        if np.isfinite(yld_total)
                        else "nan"
                    ),
                })

                if len(rows) % 200 == 0:

                    checkpoint_path = (
                        strategy_out_dir /
                        f"checkpoint_{strategy_id}_"
                        f"N{args.N}_seed{args.seed}.csv"
                    )

                    pd.DataFrame(rows).to_csv(
                        checkpoint_path,
                        index=False
                    )

                    elapsed = (
                        time.perf_counter()
                        - strategy_start_time
                    )

                    rate = (
                        len(rows) / elapsed
                        if elapsed > 0
                        else np.nan
                    )

                    remaining_seconds = (
                        (total_runs - len(rows)) / rate
                        if np.isfinite(rate) and rate > 0
                        else np.nan
                    )

                    print(
                        f"\n[{strategy_id}: "
                        f"{len(rows):,}/{total_runs:,}] | "
                        f"elapsed={elapsed / 3600:.2f} h | "
                        f"remaining={remaining_seconds / 3600:.2f} h | "
                        f"failures={failures}"
                    )

        strategy_elapsed = (
            time.perf_counter()
            - strategy_start_time
        )

        print(
            f"\nFinished simulations for {strategy_id}"
        )
        print(f"Completed runs: {len(rows):,}")
        print(f"Failures: {failures}")
        print(
            f"Elapsed time: "
            f"{strategy_elapsed / 3600:.2f} hours"
        )

        df_out = (
            pd.DataFrame(rows)
            .sort_values("future_id")
            .reset_index(drop=True)
        )

        raw_path = (
            strategy_out_dir /
            f"sobol_raw_outputs_{strategy_id}_"
            f"N{args.N}_seed{args.seed}.csv"
        )

        df_out.to_csv(
            raw_path,
            index=False
        )

        print(
            f"\nSaved raw outputs:\n"
            f"{raw_path}"
        )

        checkpoint_path = (
            strategy_out_dir /
            f"checkpoint_{strategy_id}_"
            f"N{args.N}_seed{args.seed}.csv"
        )

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        ids = df_out[
            "future_id"
        ].to_numpy()

        expected_ids = np.arange(
            len(df_out)
        )

        if not np.array_equal(
            ids,
            expected_ids
        ):
            raise RuntimeError(
                f"{strategy_id}: future_id is not "
                "0 through n-1 in the correct order."
            )

        if failures > 0:

            print(
                f"\nSobol indices were not calculated "
                f"for {strategy_id} because "
                f"{failures} invalid runs occurred."
            )

            all_strategy_summaries.append({
                "strategy_id": strategy_id,
                "strategy_label": strategy_label,
                "N": args.N,
                "seed": args.seed,
                "workers": args.workers,
                "chunksize": args.chunksize,
                "number_of_runs": len(df_out),
                "failures": failures,
                "status": "failed",
                "elapsed_hours": (
                    strategy_elapsed / 3600
                ),
            })

            continue


        Y_irr = df_out[
            "irr_mm"
        ].to_numpy(dtype=float)

        Y_yld = df_out[
            "yld_total_mg_ha"
        ].to_numpy(dtype=float)

        for output_name, output_values in {
            "irrigation": Y_irr,
            "yield": Y_yld,
        }.items():

            if not np.all(
                np.isfinite(output_values)
            ):
                raise RuntimeError(
                    f"{strategy_id}: non-finite values "
                    f"remain in {output_name}."
                )

            output_variance = float(
                np.var(output_values)
            )

            print(
                f"{strategy_id} {output_name} variance: "
                f"{output_variance:.8f}"
            )

            if output_variance <= 1e-12:
                raise RuntimeError(
                    f"{strategy_id}: {output_name} "
                    "has approximately zero variance."
                )


        (
            df_Si_irr,
            S2_irr,
            S2conf_irr
        ) = sobol_to_df(
            problem,
            Y_irr
        )

        (
            df_Si_yld,
            S2_yld,
            S2conf_yld
        ) = sobol_to_df(
            problem,
            Y_yld
        )

        for result_df in [
            df_Si_irr,
            df_Si_yld,
        ]:
            result_df.insert(
                0,
                "strategy_label",
                strategy_label
            )

            result_df.insert(
                0,
                "strategy_id",
                strategy_id
            )

        si_irr_path = (
            strategy_out_dir /
            f"sobol_Si_{strategy_id}_irr_"
            f"N{args.N}_seed{args.seed}.csv"
        )

        si_yld_path = (
            strategy_out_dir /
            f"sobol_Si_{strategy_id}_yld_"
            f"N{args.N}_seed{args.seed}.csv"
        )


        df_Si_irr.to_csv(
            si_irr_path,
            index=False
        )

        df_Si_yld.to_csv(
            si_yld_path,
            index=False
        )

    

        print(
            f"\nSaved first- and total-order indices "
            f"for {strategy_id}:"
        )
        print(f"  {si_irr_path}")
        print(f"  {si_yld_path}")

        names = problem["names"]

        save_S2(
            strategy_out_dir,
            strategy_id,
            "irr",
            args.N,
            args.seed,
            names,
            S2_irr,
            S2conf_irr
        )

        save_S2(
            strategy_out_dir,
            strategy_id,
            "yld",
            args.N,
            args.seed,
            names,
            S2_yld,
            S2conf_yld
        )


        strategy_summary = {
            "strategy_id": strategy_id,
            "strategy_label": strategy_label,
            "N": args.N,
            "seed": args.seed,
            "workers": args.workers,
            "chunksize": args.chunksize,
            "number_of_runs": len(df_out),
            "failures": failures,
            "status": "complete",
            "elapsed_hours": (
                strategy_elapsed / 3600
            ),
            "mean_irrigation_mm": float(
                np.mean(Y_irr)
            ),
            "std_irrigation_mm": float(
                np.std(Y_irr)
            ),
            "mean_yield_mg_ha": float(
                np.mean(Y_yld)
            ),
            "std_yield_mg_ha": float(
                np.std(Y_yld)
            ),
        }


        if "irr_mean" in strategy_row.index:
            strategy_summary[
                "pareto_irr_mean"
            ] = float(
                strategy_row["irr_mean"]
            )

        if "yld_mean" in strategy_row.index:
            strategy_summary[
                "pareto_yld_mean"
            ] = float(
                strategy_row["yld_mean"]
            )

        all_strategy_summaries.append(
            strategy_summary
        )

        strategy_summary_path = (
            strategy_out_dir /
            f"sobol_summary_{strategy_id}_"
            f"N{args.N}_seed{args.seed}.csv"
        )

        pd.DataFrame(
            [strategy_summary]
        ).to_csv(
            strategy_summary_path,
            index=False
        )

        print(
            f"\nCompleted Sobol analysis for "
            f"{strategy_id}: {strategy_label}"
        )

    overall_elapsed = (
        time.perf_counter()
        - overall_start_time
    )

    all_summary_df = pd.DataFrame(
        all_strategy_summaries
    )

    all_summary_path = (
        out_dir /
        f"sobol_all_5_strategies_summary_"
        f"N{args.N}_seed{args.seed}.csv"
    )

    all_summary_df.to_csv(
        all_summary_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("All five strategy runs have finished.")
    print(
        f"Overall elapsed time: "
        f"{overall_elapsed / 3600:.2f} hours"
    )
    print(
        f"Combined summary saved to:\n"
        f"{all_summary_path}"
    )
    print("=" * 75)

    if not all_summary_df.empty:
        print(
            all_summary_df.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()