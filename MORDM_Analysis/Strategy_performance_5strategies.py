"""
Performance Evaluation of Five Representative Alfalfa Irrigation Strategies

Author: Amna Omer

File Description
----------------
Reevaluates five representative Pareto irrigation strategies across
10,000 sampled states of the world to assess irrigation and yield
performance under uncertainty. An additional evaluation using
50,000 futures was conducted to assess the stability of results
with a larger uncertainty sample.

Inputs
------
- NSGA2_parallelization_OneSeed_100Gen.py.
- Strategies_Performance/selected_5_representative_strategies.csv.
- Daily ERA5 climate data and the core alfalfa irrigation model,
  accessed through the optimization script.

Outputs
-------
- Sampled design futures for the evaluation ensemble.
- Future-level cumulative irrigation, yield, and interannual
  yield variability for each strategy.
- Strategy-level performance summaries and distribution statistics.
- Comparisons with original K = 300 optimization results.

Execution Requirements
----------------------
Requires numpy, pandas, scipy, pymoo, and the core alfalfa model.
The evaluation uses 14 parallel worker processes and is
recommended for a multicore computing system.

Running the Script
------------------
From the MORDM_Analysis directory:

python Strategy_performance_5strategies.py

The default evaluation uses K = 10,000. To reproduce the
additional K = 50,000 convergence check, change K_VALUES
to [50000] in the main settings.
"""

import os
import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Prevent oversubscription

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import model / optimization functions

from NSGA2_parallelization_OneSeed_100Gen import (
    DECISION_NAMES,
    parse_decision_vector,
    run_full_simulation_with_di_levers,
    generate_design_futures
)

# Worker function
def run_one_strategy_future(row):

    try:

        df_result, df_cuttings, df_yearly, years = (
            run_full_simulation_with_di_levers(
                scalar_levers=row["scalar_levers"],
                di_levers=row["di_levers"],
                future_row=row["future_row"]
            )
        )

        total_irrigation = float(
            df_result["Irrigation_mm"].sum()
        )

        annual_yields = (
            df_yearly["Total_Yield_Mg_ha_Mitsch"]
            .astype(float)
        )


        total_yield = float(
            annual_yields.sum()
        )

        mean_annual_yield = annual_yields.mean()

        if mean_annual_yield > 0 and len(annual_yields) > 1:

            cv_yield = float(
                annual_yields.std(ddof=1)
                / mean_annual_yield
            )

        else:

            cv_yield = np.nan


        return {

            "strategy_id":
                row["strategy_id"],

            "strategy_label":
                row["strategy_label"],

            "future_id":
                row["future_id"],

            "cumulative_irrigation_mm":
                total_irrigation,

            "cumulative_yield_Mg_ha":
                total_yield,

            "cv_interannual_yield":
                cv_yield,

            # Original K=300 optimization performance
            "original_irr_mean":
                row["original_irr_mean"],

            "original_yld_mean":
                row["original_yld_mean"],

            "status":
                "success"
        }


    except Exception as e:

        print(
            f"FAILED: "
            f"{row['strategy_id']} | "
            f"future_id={row['future_id']} | "
            f"{e}"
        )

        traceback.print_exc()

        return {

            "strategy_id":
                row["strategy_id"],

            "strategy_label":
                row["strategy_label"],

            "future_id":
                row["future_id"],

            "cumulative_irrigation_mm":
                np.nan,

            "cumulative_yield_Mg_ha":
                np.nan,

            "cv_interannual_yield":
                np.nan,

            "original_irr_mean":
                row["original_irr_mean"],

            "original_yld_mean":
                row["original_yld_mean"],

            "status":
                "failed"
        }



# =====================================================
# Main script
# =====================================================
if __name__ == "__main__":

    K_VALUES = [10000]
    # Use a different seed from the K=300 optimization
    # to generate an independent post-optimization ensemble.

    DESIGN_FUTURE_SEED = 54321

    MAX_WORKERS = 14

    input_dir = Path(__file__).resolve().parent / "Strategies_Performance"

    selected_file = input_dir / "selected_5_representative_strategies.csv"


    
    if not selected_file.exists():
    
        raise FileNotFoundError(
            f"\nSelected strategy file not found:\n"
            f"{selected_file.resolve()}"
        )

    
    selected = pd.read_csv(
        selected_file
    )
    
    
    for K_NEW in K_VALUES:
    
        output_dir = input_dir / f"K{K_NEW}"
        output_dir.mkdir(parents=True, exist_ok=True)
    
        print("\n" + "=" * 80)
        print(f"STARTING K = {K_NEW:,}")
        print("=" * 80)
    
        design_futures_file = (
            output_dir
            / f"design_futures_k{K_NEW}_seed{DESIGN_FUTURE_SEED}.csv"
        )
    
        if design_futures_file.exists():
    
            print(
                "\nLoading existing design futures:"
            )
    
            print(
                design_futures_file.resolve()
            )
    
            design_futures = pd.read_csv(
                design_futures_file
            )
    
        else:
    
            print(
                f"\nGenerating {K_NEW:,} "
                f"independent design futures..."
            )
    
            print(
                f"Seed = {DESIGN_FUTURE_SEED}"
            )
    
            design_futures = (
                generate_design_futures(
                    k=K_NEW,
                    seed=DESIGN_FUTURE_SEED
                )
            )
    
            design_futures.to_csv(
                design_futures_file,
                index=False
            )
    
            print(
                "\nSaved design futures to:"
            )
    
            print(
                design_futures_file.resolve()
            )
    
        print(
            f"\nLoaded design futures: "
            f"{len(design_futures):,}"
        )
    
        print(
            f"Loaded selected strategies: "
            f"{len(selected)}"
        )
    
        required_selected_cols = (
            DECISION_NAMES
            + [
                "strategy_id",
                "strategy_label",
                "irr_mean",
                "yld_mean"
            ]
        )
    
        missing_cols = [
            c
            for c in required_selected_cols
            if c not in selected.columns
        ]
    
        if missing_cols:
    
            raise ValueError(
                "\nMissing columns in selected "
                "strategy file:\n"
                f"{missing_cols}"
            )
    
    
        if "future_id" not in design_futures.columns:
    
            raise ValueError(
                f"\n{design_futures_file.name} "
                "must include a future_id column."
            )
    
        if selected["strategy_id"].duplicated().any():
    
            duplicates = (
                selected.loc[
                    selected["strategy_id"].duplicated(),
                    "strategy_id"
                ]
                .tolist()
            )
    
            raise ValueError(
                "\nDuplicate strategy IDs found:\n"
                f"{duplicates}"
            )
    
    
        print(
            "\nStrategies:"
        )
    
        print(
            selected[
                [
                    "strategy_id",
                    "strategy_label",
                    "irr_mean",
                    "yld_mean"
                ]
            ].to_string(index=False)
        )
    
        run_rows = []
    
    
        for _, strategy in selected.iterrows():

            x = np.array(
                [
                    strategy[name]
                    for name in DECISION_NAMES
                ],
                dtype=float
            )

            scalar_levers, di_levers = (
                parse_decision_vector(x)
            )

            for _, future in design_futures.iterrows():
    
                run_rows.append({
    
                    "strategy_id":
                        strategy["strategy_id"],
    
                    "strategy_label":
                        strategy["strategy_label"],
    
                    "future_id":
                        int(future["future_id"]),
    
                    # Original optimization performance
                    "original_irr_mean":
                        float(strategy["irr_mean"]),
    
                    "original_yld_mean":
                        float(strategy["yld_mean"]),
    
                    "scalar_levers":
                        scalar_levers,
    
                    "di_levers":
                        di_levers,
    
                    "future_row":
                        future.to_dict()
                })
    
    
        total_runs = len(run_rows)
    
        expected_runs = (
            len(selected)
            * len(design_futures)
        )
    
    
        print(
            f"\nTotal simulations = "
            f"{total_runs:,}"
        )
    
        print(
            f"Expected rows = "
            f"{len(selected)} strategies × "
            f"{len(design_futures):,} futures "
            f"= {expected_runs:,}"
        )
    
    
        if total_runs != expected_runs:
    
            raise RuntimeError(
                "\nUnexpected number of "
                "Strategy × Future runs."
            )
    
        results = []
    
        start_time = time.time()
    
    
        print(
            f"\nStarting Strategy × Future "
            f"simulations with "
            f"{MAX_WORKERS} workers..."
        )
    
    
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
    
    
            futures = {
    
                executor.submit(
                    run_one_strategy_future,
                    row
                ): row
    
                for row in run_rows
            }
    
    
            for completed_i, future in enumerate(
                as_completed(futures),
                start=1
            ):
    
    
                result = future.result()
    
                results.append(
                    result
                )

                # Runtime / ETA

                elapsed = (
                    time.time()
                    - start_time
                )
    
                avg_per_run = (
                    elapsed
                    / completed_i
                )
    
                remaining = (
                    total_runs
                    - completed_i
                )
    
                eta = (
                    remaining
                    * avg_per_run
                )
    
    
                print(
                    f"Done "
                    f"{completed_i:,}/"
                    f"{total_runs:,} | "
                    f"{result['strategy_id']} | "
                    f"future="
                    f"{result['future_id']} | "
                    f"status="
                    f"{result['status']} | "
                    f"elapsed="
                    f"{elapsed/60:.1f} min | "
                    f"ETA="
                    f"{eta/60:.1f} min"
                )
    
        results_df = pd.DataFrame(
            results
        )
    
    
        results_df = (
            results_df
            .sort_values(
                [
                    "strategy_id",
                    "future_id"
                ]
            )
            .reset_index(drop=True)
        )
    
        n_failed_total = int(
            (
                results_df["status"]
                == "failed"
            ).sum()
        )
    
        n_success_total = int(
            (
                results_df["status"]
                == "success"
            ).sum()
        )
    
    
        print(
            "\n======================================"
        )
    
        print(
            "SIMULATION STATUS"
        )
    
        print(
            "======================================"
        )
    
        print(
            f"Successful runs: "
            f"{n_success_total:,}"
        )
    
        print(
            f"Failed runs: "
            f"{n_failed_total:,}"
        )
    
    
        if n_failed_total == 0:
    
            print(
                "\nAll Strategy × Future simulations "
                "completed successfully."
            )
    
        else:
    
            print(
                "\nWARNING: Some simulations failed."
            )
    
            print(
                "Inspect failed runs before using "
                "the robustness results."
            )
    
        output_file = (
            output_dir
            / (
                f"strategy_performance_"
                f"across_futures_"
                f"k{K_NEW}_"
                f"seed{DESIGN_FUTURE_SEED}.csv"
            )
        )
    
    
        results_df.to_csv(
            output_file,
            index=False
        )
    
    
        print(
            "\nSaved detailed strategy "
            "performance file to:"
        )
    
        print(
            output_file.resolve()
        )
    
        summary_df = (
    
            results_df
    
            .groupby(
                [
                    "strategy_id",
                    "strategy_label"
                ]
            )
    
            .agg(
    
                n_futures=(
                    "future_id",
                    "count"
                ),
    
                n_success=(
                    "status",
                    lambda x:
                        (x == "success").sum()
                ),
    
                n_failed=(
                    "status",
                    lambda x:
                        (x == "failed").sum()
                ),
    
                irrigation_mean=(
                    "cumulative_irrigation_mm",
                    "mean"
                ),
    
                irrigation_sd=(
                    "cumulative_irrigation_mm",
                    "std"
                ),
    
                irrigation_p05=(
                    "cumulative_irrigation_mm",
                    lambda x:
                        x.quantile(0.05)
                ),
    
                irrigation_median=(
                    "cumulative_irrigation_mm",
                    "median"
                ),
    
                irrigation_p95=(
                    "cumulative_irrigation_mm",
                    lambda x:
                        x.quantile(0.95)
                ),
    
                yield_mean=(
                    "cumulative_yield_Mg_ha",
                    "mean"
                ),
    
                yield_sd=(
                    "cumulative_yield_Mg_ha",
                    "std"
                ),
    
                yield_p05=(
                    "cumulative_yield_Mg_ha",
                    lambda x:
                        x.quantile(0.05)
                ),
    
                yield_median=(
                    "cumulative_yield_Mg_ha",
                    "median"
                ),
    
                yield_p95=(
                    "cumulative_yield_Mg_ha",
                    lambda x:
                        x.quantile(0.95)
                ),
    
                cv_mean=(
                    "cv_interannual_yield",
                    "mean"
                ),
    
                cv_sd=(
                    "cv_interannual_yield",
                    "std"
                )
            )
    
            .reset_index()
        )
    
        optimization_reference = (
    
            selected[
                [
                    "strategy_id",
                    "irr_mean",
                    "yld_mean"
                ]
            ]
    
            .rename(
                columns={
                    "irr_mean":
                        "optimization_irr_mean_k300",
                    "yld_mean":
                        "optimization_yld_mean_k300"
                }
            )
        )
    
    
        summary_df = summary_df.merge(
            optimization_reference,
            on="strategy_id",
            how="left"
        )
    
        summary_df[
            "irr_mean_difference_mm"
        ] = (
            summary_df["irrigation_mean"]
            - summary_df[
                "optimization_irr_mean_k300"
            ]
        )
    
    
        summary_df[
            "yield_mean_difference_Mg_ha"
        ] = (
            summary_df["yield_mean"]
            - summary_df[
                "optimization_yld_mean_k300"
            ]
        )
    
    
        summary_df[
            "irr_mean_difference_percent"
        ] = (
    
            summary_df[
                "irr_mean_difference_mm"
            ]
    
            / summary_df[
                "optimization_irr_mean_k300"
            ]
    
            * 100
        )
    
    
        summary_df[
            "yield_mean_difference_percent"
        ] = (
    
            summary_df[
                "yield_mean_difference_Mg_ha"
            ]
    
            / summary_df[
                "optimization_yld_mean_k300"
            ]
    
            * 100
        )

        summary_file = (
            output_dir
            / (
                f"strategy_performance_summary_"
                f"k{K_NEW}_"
                f"seed{DESIGN_FUTURE_SEED}.csv"
            )
        )
    
    
        summary_df.to_csv(
            summary_file,
            index=False
        )
    
    
        print(
            "\nSaved strategy summary file to:"
        )
    
        print(
            summary_file.resolve()
        )
    
        print(
            "\n======================================"
        )
    
        print(
            "STRATEGY PERFORMANCE SUMMARY"
        )
    
        print(
            "======================================"
        )
    
    
        display_cols = [
    
            "strategy_id",
            "strategy_label",
    
            "n_success",
            "n_failed",
    
            "optimization_irr_mean_k300",
            "irrigation_mean",
            "irr_mean_difference_percent",
    
            "optimization_yld_mean_k300",
            "yield_mean",
            "yield_mean_difference_percent",
    
            "yield_p05",
            "yield_median",
            "yield_p95",
    
            "cv_mean"
        ]
    
    
        print(
            summary_df[
                display_cols
            ].to_string(
                index=False
            )
        )
    
        total_time = (
            time.time()
            - start_time
        )
    
    
        print(
            "\n======================================"
        )
    
        print(
            f"Total runtime = "
            f"{total_time/60:.2f} minutes "
            f"({total_time/3600:.2f} hours)"
        )
    
        print(
            "======================================"
        )