"""
PRIM Scenario Discovery for Alfalfa Irrigation Strategy S3

This script applies the Patient Rule Induction Method (PRIM) to identify
combinations of uncertain conditions associated with poor performance
of the selected balanced irrigation strategy (S3).

The analysis evaluates 10,000 states of the world (SOWs) using two
satisficing criteria:
    - Irrigation application <= 80% of baseline irrigation.
    - Alfalfa yield >= 80% of baseline yield.

A future is classified as a failure when either criterion is not met.
PRIM is applied to the uncertain parameters to identify regions with
high failure density. The highest-density two-dimensional box is selected
as a parsimonious vulnerability region, and its performance and parameter
restrictions are evaluated.

The script exports:
    - Full PRIM peeling trajectory.
    - Comparison of selected candidate boxes.
    - Parameter limits of the selected PRIM box.
    - Future-level performance outcomes and selected-box membership.

Inputs:
    Strategies_Performance/K10000/
        strategy_performance_across_futures_k10000_seed54321.csv
        design_futures_k10000_seed54321.csv

Outputs:
    PRIM_Results/
        prim_peeling_trajectory_S3.csv
        prim_candidate_boxes_S3.csv
        prim_selected_box_limits_S3.csv
        prim_sow_outcomes_S3.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from ema_workbench.analysis import prim


# Settings

RESULTS_DIR = Path("Strategies_Performance/K10000")
OUTPUT_DIR = Path("PRIM_Results")
STRATEGY = "S3"

BASELINE_IRRIGATION = 5834.781
BASELINE_YIELD = 105.978
IRRIGATION_THRESHOLD = 0.80 * BASELINE_IRRIGATION
YIELD_THRESHOLD = 0.80 * BASELINE_YIELD

UNCERTAINTIES = [
    "kc_scale", "temperature_dormancy_threshold", "p_depletion",
    "wp_scale", "Kmitsch_scale", "climate_window_u", "AWC",
    "root_depth_mm", "K_sat", "theta_s", "theta_r", "n", "pwp",
]

PARAMETER_LABELS = {
    "kc_scale": "Crop coefficient scale",
    "temperature_dormancy_threshold": "Dormancy temperature threshold",
    "p_depletion": "Depletion fraction, p",
    "wp_scale": "Water productivity scale",
    "Kmitsch_scale": "Yield-response curvature scale",
    "climate_start_year": "Climate-window start year",
    "AWC": "Available water capacity",
    "root_depth_mm": "Root depth",
    "K_sat": "Saturated hydraulic conductivity",
    "theta_s": "Saturated water content",
    "theta_r": "Residual water content",
    "n": "Van Genuchten n",
    "pwp": "Permanent wilting point",
}


def load_futures():
    performance = pd.read_csv(
        RESULTS_DIR / "strategy_performance_across_futures_k10000_seed54321.csv"
    )
    futures = pd.read_csv(
        RESULTS_DIR / "design_futures_k10000_seed54321.csv"
    )

    performance_required = {
        "future_id", "strategy_id", "cumulative_irrigation_mm",
        "cumulative_yield_Mg_ha",
    }
    futures_required = {"future_id", *UNCERTAINTIES}

    for name, frame, required in [
        ("Performance", performance, performance_required),
        ("Design futures", futures, futures_required),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} file is missing columns: {sorted(missing)}")

    if "status" in performance.columns:
        performance = performance.loc[
            performance["status"].astype(str).str.lower().eq("success")
        ].copy()

    performance = performance.loc[
        performance["strategy_id"].eq(STRATEGY)
    ].copy()

    if futures["future_id"].duplicated().any():
        raise ValueError("Duplicate future IDs in design futures.")

    df = performance.merge(
        futures[["future_id", *UNCERTAINTIES]],
        on="future_id",
        how="left",
        validate="one_to_one"
    )

    required = [
        "cumulative_irrigation_mm",
        "cumulative_yield_Mg_ha",
        *UNCERTAINTIES
    ]

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[required].isna().any().any():
        raise ValueError("Missing or nonnumeric simulation results or uncertainties.")

    if len(df) != 10000:
        raise ValueError(f"Expected 10,000 successful SOWs; found {len(df):,}.")

    # Represent climate-window start years as floats for PRIM.
    df["climate_start_year"] = (
        2000 + np.floor(df["climate_window_u"].clip(0, 0.999999) * 20)
    ).astype(float)

    df["meets_yield"] = df["cumulative_yield_Mg_ha"] >= YIELD_THRESHOLD
    df["meets_irrigation"] = df["cumulative_irrigation_mm"] <= IRRIGATION_THRESHOLD
    df["combined_failure"] = ~(df["meets_yield"] & df["meets_irrigation"])

    df["failure_type"] = np.select(
        [
            df["meets_yield"] & df["meets_irrigation"],
            ~df["meets_yield"] & df["meets_irrigation"],
            df["meets_yield"] & ~df["meets_irrigation"],
            ~df["meets_yield"] & ~df["meets_irrigation"],
        ],
        [
            "Meets both criteria",
            "Fails yield only",
            "Fails irrigation only",
            "Fails both criteria",
        ],
        default="Unknown",
    )

    return df


def get_box_limits(box, box_id, X):
    """Extract box limits and identify restricted parameter ranges."""
    limits = pd.DataFrame(box.box_lims[box_id]).copy()

    if {"min", "max"}.issubset(limits.index):
        lower, upper = limits.loc["min"], limits.loc["max"]
    elif {"min", "max"}.issubset(limits.columns):
        lower, upper = limits["min"], limits["max"]
    else:
        lower, upper = limits.iloc[0], limits.iloc[1]

    result = pd.DataFrame({
        "parameter": X.columns,
        "box_min": [float(lower[p]) for p in X.columns],
        "box_max": [float(upper[p]) for p in X.columns],
        "full_min": [float(X[p].min()) for p in X.columns],
        "full_max": [float(X[p].max()) for p in X.columns],
    })

    result["lower_restricted"] = result["box_min"] > result["full_min"] + 1e-10
    result["upper_restricted"] = result["box_max"] < result["full_max"] - 1e-10
    result["is_restricted"] = (
        result["lower_restricted"] | result["upper_restricted"]
    )

    return result


def describe_conditions(limits):
    conditions = []

    for row in limits.loc[limits["is_restricted"]].itertuples(index=False):
        label = PARAMETER_LABELS.get(row.parameter, row.parameter)

        if row.lower_restricted and row.upper_restricted:
            condition = f"{label}: {row.box_min:.3f}–{row.box_max:.3f}"
        elif row.lower_restricted:
            condition = f"{label} ≥ {row.box_min:.3f}"
        else:
            condition = f"{label} ≤ {row.box_max:.3f}"

        conditions.append(condition)

    return conditions


def main():
    df = load_futures()

    prim_columns = [
        p for p in UNCERTAINTIES if p != "climate_window_u"
    ] + ["climate_start_year"]

    X = df[prim_columns].copy()
    y = df["combined_failure"].astype(int).to_numpy()

    # Run PRIM using the original scenario-discovery settings.
    algorithm = prim.Prim(
        X,
        y,
        threshold=0.50,
        peel_alpha=0.05,
        paste_alpha=0.05,
        mass_min=0.05,
        threshold_type=prim.ABOVE,
    )

    box = algorithm.find_box()

    if box is None:
        raise RuntimeError("PRIM did not identify a box.")

    trajectory = box.peeling_trajectory.copy().reset_index(drop=True)
    trajectory["box_id"] = np.arange(len(trajectory))

    # Select the highest-density two-dimensional box.
    two_dimensional = trajectory.loc[trajectory["res_dim"].eq(2)]

    if two_dimensional.empty:
        raise RuntimeError("PRIM did not identify a two-dimensional box.")

    selected_box_id = int(
        two_dimensional.loc[
            two_dimensional["density"].idxmax(), "box_id"
        ]
    )

    selected = trajectory.iloc[selected_box_id]
    limits = get_box_limits(box, selected_box_id, X)
    restricted = limits.loc[limits["is_restricted"], "parameter"].tolist()

    if len(restricted) != 2:
        raise RuntimeError(
            f"Selected Box {selected_box_id} has "
            f"{len(restricted)} restricted parameters instead of two."
        )

    box.select(selected_box_id)

    # Compare the selected box with nearby and higher-dimensional candidates.
    candidate_boxes = sorted(set([
        selected_box_id,
        selected_box_id + 1,
        43,
        44,
    ]))

    candidate_boxes = [
        box_id for box_id in candidate_boxes
        if box_id < len(trajectory)
    ]

    rows = []

    for box_id in candidate_boxes:
        candidate = trajectory.iloc[box_id]
        conditions = describe_conditions(get_box_limits(box, box_id, X))

        rows.append({
            "Box": box_id,
            "Dimensions": len(conditions),
            "Density": float(candidate["density"]),
            "Coverage": float(candidate["coverage"]),
            "Mass": float(candidate["mass"]),
            "Restricted parameter ranges": "; ".join(conditions),
        })

    candidates = pd.DataFrame(rows)

    # Identify SOWs inside the selected PRIM box.
    df["inside_prim_box"] = True

    for row in limits.loc[limits["is_restricted"]].itertuples(index=False):
        if row.lower_restricted:
            df["inside_prim_box"] &= df[row.parameter] >= row.box_min

        if row.upper_restricted:
            df["inside_prim_box"] &= df[row.parameter] <= row.box_max

    inside = df["inside_prim_box"]
    n_inside = int(inside.sum())
    n_failures = int(y.sum())
    failures_inside = int(df.loc[inside, "combined_failure"].sum())

    if n_inside == 0 or n_failures == 0:
        raise RuntimeError("Cannot calculate PRIM box statistics.")

    observed_density = failures_inside / n_inside
    observed_coverage = failures_inside / n_failures
    observed_mass = n_inside / len(df)

    for metric, observed in [
        ("density", observed_density),
        ("coverage", observed_coverage),
        ("mass", observed_mass),
    ]:
        if not np.isclose(
            observed, float(selected[metric]), atol=1e-6
        ):
            raise RuntimeError(
                f"Box {selected_box_id} {metric} mismatch: "
                f"observed {observed:.6f}, "
                f"PRIM {selected[metric]:.6f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trajectory.to_csv(
        OUTPUT_DIR / f"prim_peeling_trajectory_{STRATEGY}.csv",
        index=False
    )

    candidates.to_csv(
        OUTPUT_DIR / f"prim_candidate_boxes_{STRATEGY}.csv",
        index=False
    )

    limits.to_csv(
        OUTPUT_DIR / f"prim_selected_box_limits_{STRATEGY}.csv",
        index=False
    )

    df.to_csv(
        OUTPUT_DIR / f"prim_sow_outcomes_{STRATEGY}.csv",
        index=False
    )

    print(f"\nStrategy: {STRATEGY}")
    print(f"Number of SOWs: {len(df):,}")
    print(f"Number of failures: {n_failures:,}")
    print(f"Overall failure rate: {n_failures / len(df):.3f}")

    print(f"\nSelected PRIM box: {selected_box_id}")
    print(f"Restricted dimensions: {len(restricted)}")
    print(f"Density: {observed_density:.3f}")
    print(f"Coverage: {observed_coverage:.3f}")
    print(f"Mass: {observed_mass:.3f}")

    print("\nRestricted conditions:")
    for condition in describe_conditions(limits):
        print(f"  {condition}")

    print("\nCandidate box comparison:")
    with pd.option_context("display.max_colwidth", None, "display.width", 220):
        print(candidates.round(3).to_string(index=False))

    print(f"\nSaved four CSV files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
