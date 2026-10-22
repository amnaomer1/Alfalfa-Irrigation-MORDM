"""
Evaluation of Selected Alfalfa Irrigation Strategies Across Sample Sizes

Author: Amna Omer

File Description
----------------
Reevaluates five representative Pareto irrigation strategies across
independent uncertainty ensembles of K = 1,000, 5,000, and 10,000
states of the world to assess the stability of irrigation and yield
outcomes relative to the original K = 300 optimization results.

Inputs
------
- NSGA2_parallelization_OneSeed_100Gen.py.
- Strategies_Performance/selected_5_representative_strategies.csv.
- Daily ERA5 climate data and the core alfalfa irrigation model,
  accessed through the optimization script.

Outputs
-------
- Sampled states of the world for each ensemble size.
- Future-level irrigation and yield results for all five strategies.
- Strategy-level summary statistics and comparisons with K = 300.
- Combined summary and plot-ready comparison tables.
- Intermediate checkpoint files.

Execution Requirements
----------------------
Requires numpy, pandas, scipy, pymoo, and the core alfalfa model.
The evaluation uses 14 parallel worker processes and is
recommended for a multicore computing system.

Running the Script
------------------
From the MORDM_Analysis directory:

python Reevaluate_Selected_Strategies_K_1000_5000_10000.py
"""


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



import sys
import time
import traceback
import importlib.util
import multiprocessing as mp

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd



BASE_DIR = Path(__file__).resolve().parent

N_WORKERS = 14

# Independent LHS ensemble for each sample size
K_SETTINGS = {
    1000: 61001,
    5000: 65001,
    10000: 610001,
}

# Enough chunks to keep 14 workers busy.
# 28 chunks/strategy -> 140 tasks for each K
N_CHUNKS_PER_STRATEGY = 28

CHECKPOINT_EVERY_TASKS = 10



OPTIMIZATION_SCRIPT = (
    BASE_DIR /
    "NSGA2_parallelization_OneSeed_100Gen.py"
)

SELECTED_STRATEGIES_FILE = (
    BASE_DIR /
    "Strategies_Performance" /
    "selected_5_representative_strategies.csv"
)


OUTPUT_DIR = (
    BASE_DIR /
    "Reevaluation_Selected_Strategies_K1000_5000_10000"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if not OPTIMIZATION_SCRIPT.exists():
    raise FileNotFoundError(
        f"Optimization script not found:\n"
        f"{OPTIMIZATION_SCRIPT.resolve()}"
    )

spec = importlib.util.spec_from_file_location(
    "alfalfa_opt",
    OPTIMIZATION_SCRIPT
)

alfalfa_opt = importlib.util.module_from_spec(spec)

sys.modules["alfalfa_opt"] = alfalfa_opt

spec.loader.exec_module(alfalfa_opt)

print("\n Imported original optimization code")
print(f"   {OPTIMIZATION_SCRIPT.name}")


required_objects = [
    "DECISION_NAMES",
    "UNC_KEYS",
    "generate_design_futures",
    "parse_decision_vector",
    "evaluate_one_future_one_pass_di",
]

for obj in required_objects:
    if not hasattr(alfalfa_opt, obj):
        raise AttributeError(
            f"Original optimization script "
            f"does not contain: {obj}"
        )

DECISION_NAMES = alfalfa_opt.DECISION_NAMES
UNC_KEYS = alfalfa_opt.UNC_KEYS



strategies_df = pd.read_csv(
    SELECTED_STRATEGIES_FILE
)

required_cols = (
    DECISION_NAMES +
    ["irr_mean", "yld_mean"]
)

missing = [
    c for c in required_cols
    if c not in strategies_df.columns
]

if missing:
    raise ValueError(
        f"Selected strategy file is missing:\n{missing}"
    )

if "strategy_id" not in strategies_df.columns:
    strategies_df["strategy_id"] = [
        f"S{i+1}"
        for i in range(len(strategies_df))
    ]

strategy_order = [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
]

strategies_df["strategy_id"] = pd.Categorical(
    strategies_df["strategy_id"],
    categories=strategy_order,
    ordered=True
)

strategies_df = (
    strategies_df
    .sort_values("strategy_id")
    .reset_index(drop=True)
)

if len(strategies_df) != 5:
    raise ValueError(
        f"Expected 5 strategies, "
        f"found {len(strategies_df)}"
    )


print("\n==============================================")
print("Selected strategies")
print("==============================================")

print(
    strategies_df[
        [
            "strategy_id",
            "irr_mean",
            "yld_mean",
        ]
    ].to_string(index=False)
)


def evaluate_strategy_chunk(task):

    (
        K,
        strategy_id,
        strategy_label,
        x_values,
        irr_K300,
        yld_K300,
        chunk_id,
        future_records,
    ) = task

    start = time.time()

    try:

        x = np.asarray(
            x_values,
            dtype=float
        )

        scalar_levers, di_levers = (
            alfalfa_opt.parse_decision_vector(x)
        )

        rows = []

        for future in future_records:

            irr, yld = (
                alfalfa_opt.evaluate_one_future_one_pass_di(
                    scalar_levers=scalar_levers,
                    di_levers=di_levers,
                    future_row=future
                )
            )

            row = {
                "K": int(K),
                "strategy_id": strategy_id,
                "strategy_label": strategy_label,
                "chunk_id": int(chunk_id),
                "future_id": int(
                    future["future_id"]
                ),

                "irrigation_mm": float(irr),
                "yield_Mg_ha": float(yld),

                # Original K300 objectives
                "irr_mean_K300": float(
                    irr_K300
                ),
                "yld_mean_K300": float(
                    yld_K300
                ),
            }

            # Preserve uncertainty values for
            # robustness and PRIM analyses
            for name in UNC_KEYS:
                row[name] = float(
                    future[name]
                )

            rows.append(row)

        elapsed_minutes = (
            time.time() - start
        ) / 60.0

        return {
            "K": K,
            "strategy_id": strategy_id,
            "chunk_id": chunk_id,
            "rows": rows,
            "elapsed_minutes": elapsed_minutes,
            "error": None,
        }

    except Exception as e:

        return {
            "K": K,
            "strategy_id": strategy_id,
            "chunk_id": chunk_id,
            "rows": [],
            "elapsed_minutes": None,
            "error": (
                f"{e}\n"
                f"{traceback.format_exc()}"
            ),
        }



def run_one_K(K, sow_seed):

    print("\n\n")
    print("=" * 70)
    print(f"STARTING K={K:,}")
    print("=" * 70)

    K_DIR = (
        OUTPUT_DIR /
        f"K{K}"
    )

    K_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        f"\nGenerating K={K:,} "
        f"independent LHS ensemble"
    )

    print(
        f"SOW seed = {sow_seed}"
    )

    futures_df = (
        alfalfa_opt.generate_design_futures(
            k=K,
            seed=sow_seed
        )
    )

    sow_file = (
        K_DIR /
        f"SOWs_K{K}_seed{sow_seed}.csv"
    )

    futures_df.to_csv(
        sow_file,
        index=False
    )

    print(
        f" Saved SOW ensemble:\n"
        f"{sow_file}"
    )



    future_indices = np.array_split(
        np.arange(len(futures_df)),
        N_CHUNKS_PER_STRATEGY
    )

    tasks = []

    for _, strategy in strategies_df.iterrows():

        strategy_id = str(
            strategy["strategy_id"]
        )

        strategy_label = (
            str(strategy["strategy_label"])
            if "strategy_label"
               in strategies_df.columns
            else strategy_id
        )

        x_values = [
            float(strategy[name])
            for name in DECISION_NAMES
        ]

        irr_K300 = float(
            strategy["irr_mean"]
        )

        yld_K300 = float(
            strategy["yld_mean"]
        )

        for chunk_id, idx in enumerate(
            future_indices
        ):

            future_records = (
                futures_df
                .iloc[idx]
                .to_dict("records")
            )

            tasks.append(
                (
                    K,
                    strategy_id,
                    strategy_label,
                    x_values,
                    irr_K300,
                    yld_K300,
                    chunk_id,
                    future_records,
                )
            )


    print("\nParallel setup")
    print("--------------------------------")

    print("Strategies              = 5")

    print(
        f"SOWs/strategy           = {K:,}"
    )

    print(
        f"Total model simulations = "
        f"{5*K:,}"
    )

    print(
        f"Chunks/strategy         = "
        f"{N_CHUNKS_PER_STRATEGY}"
    )

    print(
        f"Total tasks             = "
        f"{len(tasks)}"
    )

    print(
        f"Workers                 = "
        f"{N_WORKERS}"
    )


    start_K = time.time()

    all_rows = []

    completed_tasks = 0

    ctx = mp.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        mp_context=ctx
    ) as executor:

        submitted = {
            executor.submit(
                evaluate_strategy_chunk,
                task
            ): (
                task[1],
                task[6]
            )
            for task in tasks
        }

        for future in as_completed(
            submitted
        ):

            strategy_id, chunk_id = (
                submitted[future]
            )

            result = future.result()

            if result["error"] is not None:

                raise RuntimeError(
                    f"\nFailure in "
                    f"{strategy_id}, "
                    f"chunk {chunk_id}:\n"
                    f"{result['error']}"
                )

            all_rows.extend(
                result["rows"]
            )

            completed_tasks += 1

            elapsed_hours = (
                time.time() - start_K
            ) / 3600.0

            if elapsed_hours > 0:
                rate = (
                    completed_tasks /
                    elapsed_hours
                )

                remaining = (
                    len(tasks) -
                    completed_tasks
                )

                eta_hours = (
                    remaining /
                    rate
                )

            else:
                eta_hours = np.nan


            print(
                f" K={K:,} "
                f"{completed_tasks}/{len(tasks)} tasks "
                f"| {strategy_id} "
                f"chunk {chunk_id:02d} "
                f"| elapsed={elapsed_hours:.2f} h "
                f"| ETA={eta_hours:.2f} h",
                flush=True
            )


            if (
                completed_tasks %
                CHECKPOINT_EVERY_TASKS == 0
                or
                completed_tasks == len(tasks)
            ):

                checkpoint_df = (
                    pd.DataFrame(
                        all_rows
                    )
                )

                checkpoint_file = (
                    K_DIR /
                    f"K{K}_checkpoint.csv"
                )

                checkpoint_df.to_csv(
                    checkpoint_file,
                    index=False
                )

                print(
                    f"   checkpoint saved "
                    f"({len(checkpoint_df):,} rows)",
                    flush=True
                )

    results_df = pd.DataFrame(
        all_rows
    )

    results_df[
        "strategy_id"
    ] = pd.Categorical(
        results_df["strategy_id"],
        categories=strategy_order,
        ordered=True
    )

    results_df = (
        results_df
        .sort_values(
            [
                "strategy_id",
                "future_id",
            ]
        )
        .reset_index(drop=True)
    )


    expected_rows = 5 * K

    if len(results_df) != expected_rows:

        raise ValueError(
            f"K={K}: expected "
            f"{expected_rows:,} rows, "
            f"obtained "
            f"{len(results_df):,}"
        )


    future_file = (
        K_DIR /
        f"selected_5_K{K}_future_outputs.csv"
    )

    results_df.to_csv(
        future_file,
        index=False
    )

    summary = (

        results_df

        .groupby(
            "strategy_id",
            observed=True
        )

        .agg(

            n_sows=(
                "future_id",
                "count"
            ),

            irrigation_mean=(
                "irrigation_mm",
                "mean"
            ),

            irrigation_median=(
                "irrigation_mm",
                "median"
            ),

            irrigation_std=(
                "irrigation_mm",
                "std"
            ),

            irrigation_p5=(
                "irrigation_mm",
                lambda x:
                np.quantile(x, 0.05)
            ),

            irrigation_q1=(
                "irrigation_mm",
                lambda x:
                np.quantile(x, 0.25)
            ),

            irrigation_q3=(
                "irrigation_mm",
                lambda x:
                np.quantile(x, 0.75)
            ),

            irrigation_p95=(
                "irrigation_mm",
                lambda x:
                np.quantile(x, 0.95)
            ),

            irrigation_min=(
                "irrigation_mm",
                "min"
            ),

            irrigation_max=(
                "irrigation_mm",
                "max"
            ),

            yield_mean=(
                "yield_Mg_ha",
                "mean"
            ),

            yield_median=(
                "yield_Mg_ha",
                "median"
            ),

            yield_std=(
                "yield_Mg_ha",
                "std"
            ),

            yield_p5=(
                "yield_Mg_ha",
                lambda x:
                np.quantile(x, 0.05)
            ),

            yield_q1=(
                "yield_Mg_ha",
                lambda x:
                np.quantile(x, 0.25)
            ),

            yield_q3=(
                "yield_Mg_ha",
                lambda x:
                np.quantile(x, 0.75)
            ),

            yield_p95=(
                "yield_Mg_ha",
                lambda x:
                np.quantile(x, 0.95)
            ),

            yield_min=(
                "yield_Mg_ha",
                "min"
            ),

            yield_max=(
                "yield_Mg_ha",
                "max"
            ),
        )

        .reset_index()
    )


    # Add K
    summary.insert(
        1,
        "K",
        K
    )


    # Original K300 values
    original = (

        strategies_df[
            [
                "strategy_id",
                "irr_mean",
                "yld_mean",
            ]
        ]

        .rename(
            columns={
                "irr_mean":
                    "irrigation_mean_K300",

                "yld_mean":
                    "yield_mean_K300",
            }
        )
    )


    summary = summary.merge(
        original,
        on="strategy_id",
        how="left"
    )


    summary[
        "irrigation_change_from_K300"
    ] = (
        summary["irrigation_mean"]
        -
        summary["irrigation_mean_K300"]
    )


    summary[
        "irrigation_abs_pct_change_from_K300"
    ] = (

        100
        *
        np.abs(
            summary[
                "irrigation_change_from_K300"
            ]
        )
        /
        np.maximum(
            np.abs(
                summary[
                    "irrigation_mean_K300"
                ]
            ),
            1e-12
        )
    )


    summary[
        "yield_change_from_K300"
    ] = (
        summary["yield_mean"]
        -
        summary["yield_mean_K300"]
    )


    summary[
        "yield_abs_pct_change_from_K300"
    ] = (

        100
        *
        np.abs(
            summary[
                "yield_change_from_K300"
            ]
        )
        /
        np.maximum(
            np.abs(
                summary[
                    "yield_mean_K300"
                ]
            ),
            1e-12
        )
    )


    summary_file = (
        K_DIR /
        f"selected_5_K{K}_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )


    runtime_hours = (
        time.time() - start_K
    ) / 3600.0


    print("\n")
    print("=" * 70)

    print(
        f" K={K:,} COMPLETE"
    )

    print(
        f"Runtime = "
        f"{runtime_hours:.2f} h"
    )

    print(
        f"Saved full output:\n"
        f"{future_file}"
    )

    print(
        f"Saved summary:\n"
        f"{summary_file}"
    )

    print("=" * 70)


    return summary


overall_start = time.time()

all_summaries = []


for K, seed in K_SETTINGS.items():

    summary_K = run_one_K(
        K=K,
        sow_seed=seed
    )

    all_summaries.append(
        summary_K
    )



combined = pd.concat(
    all_summaries,
    ignore_index=True
)


combined = combined.sort_values(
    [
        "strategy_id",
        "K",
    ]
).reset_index(drop=True)


combined_file = (
    OUTPUT_DIR /
    "selected_5_K1000_K5000_K10000_combined_summary.csv"
)


combined.to_csv(
    combined_file,
    index=False
)


# Original K300 rows
K300_rows = (

    strategies_df[
        [
            "strategy_id",
            "irr_mean",
            "yld_mean",
        ]
    ]

    .rename(
        columns={
            "irr_mean":
                "irrigation_mean",

            "yld_mean":
                "yield_mean",
        }
    )
)


K300_rows["K"] = 300


plot_data = pd.concat(
    [
        K300_rows[
            [
                "strategy_id",
                "K",
                "irrigation_mean",
                "yield_mean",
            ]
        ],

        combined[
            [
                "strategy_id",
                "K",
                "irrigation_mean",
                "yield_mean",
            ]
        ],
    ],
    ignore_index=True
)


plot_data = plot_data.sort_values(
    [
        "strategy_id",
        "K",
    ]
).reset_index(drop=True)


plot_file = (
    OUTPUT_DIR /
    "K300_K1000_K5000_K10000_plot_data.csv"
)


plot_data.to_csv(
    plot_file,
    index=False
)



compact = combined[
    [
        "strategy_id",
        "K",

        "irrigation_mean",
        "irrigation_abs_pct_change_from_K300",

        "yield_mean",
        "yield_abs_pct_change_from_K300",
    ]
].copy()


print("\n\n")
print("=" * 80)
print("SELECTED STRATEGY CONVERGENCE SUMMARY")
print("=" * 80)

print(
    compact.to_string(
        index=False
    )
)


total_hours = (
    time.time() -
    overall_start
) / 3600.0


print("\n")
print("=" * 80)

print(" ALL ENSEMBLE SIZES COMPLETE")

print(
    f"Total runtime = "
    f"{total_hours:.2f} hours"
)

print(
    f"\nCombined summary:\n"
    f"{combined_file}"
)

print(
    f"\nPlot-ready file:\n"
    f"{plot_file}"
)

print("=" * 80)