"""
Select Five Representative Irrigation Strategies
================================================

Selects five representative irrigation strategies from the final NSGA-II
Pareto archive using K-means clustering in standardized objective space.

The clustering uses mean cumulative irrigation and mean cumulative yield.
Within each cluster, the Pareto solution closest to the cluster centroid
is selected as the representative strategy. The five selected strategies
are ordered by increasing irrigation use and labeled S1–S5.

Inputs
------
NSGA2_K300_P100_G100_S101/
    final_pareto_archive_from_all_population.csv

Outputs
-------
Strategies_Performance/
    final_archive_with_5_objective_clusters.csv
    selected_5_representative_strategies.csv

Run from any directory:
    python /path/to/MORDM_Analysis/Select_Representative_Strategies.py
"""

from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Configuration
BASE_DIR = Path(__file__).resolve().parent
PARETO_FILE = (
    BASE_DIR
    / "NSGA2_K300_P100_G100_S101"
    / "final_pareto_archive_from_all_population.csv"
)
OUTPUT_DIR = BASE_DIR / "Strategies_Performance"

N_CLUSTERS = 5
RANDOM_STATE = 42
N_INIT = 20
OBJECTIVES = ["irr_mean", "yld_mean"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PARETO_FILE)

    missing_cols = [col for col in OBJECTIVES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Cluster Pareto solutions in standardized objective space
    X_scaled = StandardScaler().fit_transform(df[OBJECTIVES])

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    df["cluster_raw"] = kmeans.fit_predict(X_scaled)

    distances = cdist(X_scaled, kmeans.cluster_centers_)

    df["distance_to_cluster_center"] = [
        distances[i, int(df.iloc[i]["cluster_raw"])]
        for i in range(len(df))
    ]

    # Select the Pareto solution nearest each cluster centroid
    representative_rows = []

    for cluster in sorted(df["cluster_raw"].unique()):
        cluster_df = df[df["cluster_raw"] == cluster]
        representative = cluster_df.loc[
            cluster_df["distance_to_cluster_center"].idxmin()
        ]
        representative_rows.append(representative)

    representatives_df = pd.DataFrame(representative_rows)

    # Order representative strategies by increasing irrigation use
    representatives_df = (
        representatives_df
        .sort_values("irr_mean")
        .reset_index(drop=True)
    )

    representatives_df["strategy_label"] = [
        "S1 Low irrigation",
        "S2 Low-moderate irrigation",
        "S3 Balanced strategy",
        "S4 Moderate-high production",
        "S5 High production",
    ]

    representatives_df["strategy_id"] = [
        f"S{i + 1}" for i in range(len(representatives_df))
    ]

    cluster_order_map = {
        row["cluster_raw"]: i + 1
        for i, row in representatives_df.iterrows()
    }

    df["cluster_ordered"] = df["cluster_raw"].map(cluster_order_map)

    # Export clustered archive and selected strategies
    df.to_csv(
        OUTPUT_DIR / "final_archive_with_5_objective_clusters.csv",
        index=False,
    )

    representatives_df.to_csv(
        OUTPUT_DIR / "selected_5_representative_strategies.csv",
        index=False,
    )

    print("\nSelected representative strategies:")
    print(
        representatives_df[
            ["strategy_id", "strategy_label", "irr_mean", "yld_mean"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
