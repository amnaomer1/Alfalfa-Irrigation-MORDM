
# Balancing Water Conservation and Crop Production for Robust Alfalfa Irrigation under Uncertainty

## 1. Overview

This repository contains the Python code and analysis workflows developed to evaluate alfalfa irrigation management strategies under weather and biophysical uncertainty in California's Imperial Valley.

The study applies a Multi-Objective Robust Decision Making (MORDM) framework to explore tradeoffs between irrigation water use and crop production. A daily, field-scale alfalfa simulation model represents soil water dynamics, irrigation decisions, crop evapotranspiration, and yield responses. The model is coupled with the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to identify irrigation management strategies that balance water conservation and alfalfa production.

The repository includes scripts and Jupyter notebooks used to generate the figures and tables presented in the manuscript and supplementary material.

## 2. Repository Structure

The repository is organized into the following main components:

```text
Alfalfa_MORDM_GitHub/
│
├── Alfalfa_Irrigation_Model.py
│
├── Input/
│   └── Model input datasets
│
├── Baseline_and_Model_Evaluation/
│   ├── Baseline_Scenario.ipynb
│   └── Model_Evaluation.ipynb
│
├── Pre_Optimization_Convergence_Tests/
│   ├── Convergence_10Seeds_K5000.py
│   ├── Convergence_10Seeds_K10000.py
│   └── convergence plots.ipynb
│
└── MORDM_Analysis/
    ├── NSGA2_parallelization_OneSeed_100Gen.py
    ├── NSGA2_parallelization_OneSeed_200Gen.py
    ├── NSGA2_parallelization-MultiSeeds.py
    ├── Reoptimization_NSGA2_K700.py
    ├── Reoptimization_NSGA2_K1000.py
    ├── Select_Representative_Strategies.py
    ├── Strategy_performance_5strategies.py
    ├── Reevaluate_Selected_Strategies_K_1000_5000_10000.py
    ├── Strategy_Irrigation_Diagnostics.py
    ├── Sobol_Analysis.py
    ├── PRIM_Scenario_Discovery.py
    ├── Manuscript_Figures_and_Tables.ipynb
    └── Supplementary_Analysis_and_Figures.ipynb
```

The main simulation model is defined in `Alfalfa_Irrigation_Model.py`. The `Input/` directory contains datasets used by the model, while the remaining directories organize model evaluation, convergence testing, optimization, robustness assessment, sensitivity analysis, scenario discovery, and visualization.

## 3. Data Sources and Inputs

The study uses climate, soil, crop, and model evaluation datasets to simulate alfalfa irrigation and production in California's Imperial Valley.

The main input categories include:

- **Climate data:** Daily meteorological inputs used to calculate reference evapotranspiration and represent weather conditions.
- **Soil data:** Parameters describing soil water storage, hydraulic properties, and root-zone water availability.
- **Crop parameters:** Alfalfa growth, evapotranspiration, and yield-response parameters.
- **Model evaluation data:** Independent datasets used to evaluate simulated evapotranspiration and crop water use.
- **Uncertainty ranges:** Parameter bounds used to generate alternative states of the world for optimization, robustness evaluation, and sensitivity analysis.

The model uses daily climate inputs derived from ERA5, including the file `Input/ImperialCounty_ERA5_DailyClimate_2000_2023.csv`.

Additional data sources and processing procedures are documented in the accompanying manuscript and supplementary material.

## 4. Analysis Workflow and Reproducibility

The following table summarizes the scripts and notebooks used in this study, their functions, required dependencies, computational resources, and reference runtimes. Stages represent the logical analysis workflow; some stages can run independently once their required inputs are available.

| Stage | Script or notebook | Description | Required inputs / dependencies | Allocated CPUs | Reference runtime |
|---|---|---|---|---|---|
| 1 | `Baseline_Scenario.ipynb` | Runs the baseline alfalfa irrigation simulation and generates diagnostic figures. | Core model and climate inputs | — | — |
| 2 | `Model_Evaluation.ipynb` | Evaluates simulated ET₀, PETc, AETc, and alfalfa yield against reference datasets. | Core model, climate inputs, and evaluation datasets | — | — |
| 3 | `Convergence_10Seeds_K5000.py` | Evaluates convergence across three fixed irrigation strategies, seven sample sizes (K=100–5,000), and 10 random seeds. | Core model and climate inputs | 48 | 1 h 47 min |
| 4 | `Convergence_10Seeds_K10000.py` | Extends the convergence analysis to K=10,000 across three strategies and 10 random seeds. | Core model and climate inputs | 48 | 2 h 12 min |
| 5 | `convergence plots.ipynb` | Generates convergence plots for irrigation and yield means and distribution percentiles. | Stages 3–4 | — | — |
| 6 | `NSGA2_parallelization_OneSeed_100Gen.py` | Runs the main NSGA-II optimization (100 individuals, 100 generations, K=300) to identify Pareto-efficient irrigation strategies. | Core model and climate inputs | 48 | 21 h 27 min |
| 7 | `NSGA2_parallelization_OneSeed_200Gen.py` | Extends NSGA-II optimization to 200 generations to assess Pareto-front convergence. | Core model and climate inputs | 48 | 43 h 19 min |
| 8 | `NSGA2_parallelization-MultiSeeds.py` | Runs NSGA-II with three additional random seeds (102–104) to assess optimization consistency. | Core model and climate inputs | 48 | 64 h 5 min |
| 9 | `Reoptimization_NSGA2_K700.py` | Repeats optimization with K=700 to assess sensitivity to uncertainty sample size. | Core model and climate inputs | 48 | 49 h 55 min 36 s |
| 10 | `Reoptimization_NSGA2_K1000.py` | Repeats optimization with K=1,000 to assess sensitivity to uncertainty sample size. | Core model and climate inputs | 48 | 71 h 50 min 18 s |
| 11 | `Select_Representative_Strategies.py` | Selects five representative strategies (S1–S5) using K-means clustering of the final Pareto archive. | Stage 6 final Pareto archive | — | — |
| 12 | `Strategy_performance_5strategies.py` | Evaluates five strategies across K=10,000 and optionally K=50,000 futures to assess irrigation, yield, robustness, and outcome stability. | Stage 11 CSV; Stage 6 optimization functions | 48 | 15 min 33 s (K=10,000); 1 h 17 min 18 s (K=50,000) |
| 13 | `Reevaluate_Selected_Strategies_K_1000_5000_10000.py` | Reevaluates five strategies across K=1,000, 5,000, and 10,000 to assess the stability of irrigation and yield outcomes. | Stage 11 CSV; Stage 6 optimization functions | 48 | 25 min 58 s |
| 14 | `Strategy_Irrigation_Diagnostics.py` | Quantifies routine, forced, and survival irrigation and examines seasonal irrigation patterns across 300 futures. | Stage 11 CSV; Stage 6 design futures; core model and climate inputs | 48 | 35 s |
| 15 | `Sobol_Analysis.py` | Calculates first-order, total-order, and second-order Sobol sensitivity indices for irrigation and yield across 13 uncertain parameters. | Stage 11 CSV; Stage 6 optimization functions | 48 | N=512: 22 min 25 s; N=1,024: 44 min 41 s; N=2,048: 1 h 28 min 39 s |
| 16 | `PRIM_Scenario_Discovery.py` | Identifies vulnerability regions for Strategy S3 using PRIM across 10,000 sampled futures. | Stage 12 K=10,000 outputs | — | — |
| 17 | `Manuscript_Figures_and_Tables.ipynb` | Reproduces manuscript Figures 2–5 and Table 3 using previously generated analysis outputs. | Optimization, performance, Sobol, and PRIM outputs | — | — |
| 18 | `Supplementary_Analysis_and_Figures.ipynb` | Reproduces supplementary Figures S13–S21 and Tables S11–S13 using previously generated analysis outputs. | Convergence, optimization, robustness, Sobol, and PRIM outputs | — | — |

**Notes:** Runtimes are measured wall-clock durations obtained from SLURM accounting records on Penn State's Roar HPC system. Allocated CPUs indicate scheduler allocations, not necessarily the number of parallel workers used by each script. A dash indicates that a reference runtime or CPU allocation was not recorded.

## 5. How to Reproduce the Results

The repository supports two approaches to reproducing the study results, depending on whether the required precomputed outputs are available.

### 5.1. Reproducing manuscript figures and tables

When the required analysis outputs are available, the manuscript and supplementary figures can be regenerated without rerunning the computationally intensive simulations.

The relevant notebooks are:

- `MORDM_Analysis/Manuscript_Figures_and_Tables.ipynb`
- `MORDM_Analysis/Supplementary_Analysis_and_Figures.ipynb`
- `Pre_Optimization_Convergence_Tests/convergence plots.ipynb`
- `Baseline_and_Model_Evaluation/Baseline_Scenario.ipynb`
- `Baseline_and_Model_Evaluation/Model_Evaluation.ipynb`

### 5.2. Reproducing the complete analysis

To reproduce the analysis from the model inputs, follow the execution order and dependencies described in Section 4.

**Note:** The two large convergence future-level CSV files (K=100–5,000 and K=10,000) are not included in this GitHub repository because they exceed GitHub's regular file-size limit. The smaller convergence summaries are included.

## 6. Expected Outputs

### Main Output Files and Directories

The following table summarizes the main output directories generated by the analysis workflow. These directories contain simulation results, optimization archives, strategy evaluations, sensitivity indices, and scenario discovery outputs used to reproduce the manuscript and supplementary analyses.

| Output directory | Main contents | Purpose / downstream use |
|---|---|---|
| `Baseline_and_Model_Evaluation/Evaluation_Output/` | Combined daily ET₀, PETc, and AETc results from model evaluation simulations. | Supports model evaluation and comparison with reference datasets. |
| `Pre_Optimization_Convergence_Tests/Output_convergence_Normal_Extreme_VeryExtreme_K5000_10Seeds/` | Future-level simulation results, strategy-level summaries, and convergence statistics for K=100–5,000 across 10 random seeds. | Evaluates uncertainty-sampling convergence before optimization. |
| `Pre_Optimization_Convergence_Tests/Output_convergence_Normal_Extreme_VeryExtreme_K10000_10Seeds/` | Future-level results and convergence summaries for K=10,000 across 10 random seeds. | Extends the pre-optimization convergence assessment and supports convergence plots. |
| `MORDM_Analysis/NSGA2_K300_P100_G100_S101/` | K=300 sampled futures, generation-level populations, Pareto solutions, final Pareto archive, and optimization metadata. | Main optimization output. Provides the Pareto archive for representative strategy selection and sampled futures used in subsequent analyses. |
| `MORDM_Analysis/NSGA2_K300_P100_G200_S101/` | Generation-level optimization results, final Pareto solutions, and metadata from the 200-generation run. | Evaluates Pareto-front convergence with additional generations. |
| `MORDM_Analysis/NSGA2_K300_P100_G100_MultiSeeds/` | Shared sampled futures, seed-specific optimization histories and Pareto archives, and combined solutions for seeds 102–104. | Evaluates consistency of optimization results across random seeds. |
| `MORDM_Analysis/NSGA2_K700_P100_G100_S101_Reoptimization/` | K=700 sampled futures, optimization histories, Pareto solutions, and metadata. | Evaluates sensitivity of Pareto solutions to uncertainty sample size. |
| `MORDM_Analysis/NSGA2_K1000_P100_G100_S101_Reoptimization/` | K=1,000 sampled futures, optimization histories, Pareto solutions, and metadata. | Extends the uncertainty-sample-size stability assessment. |
| `MORDM_Analysis/Strategies_Performance/` | Five selected irrigation strategies (S1–S5), clustered Pareto archive, future-level strategy performance, summary statistics, K=10,000 and K=50,000 uncertainty ensembles, and irrigation diagnostics. | Main strategy evaluation output. Supports robustness analysis, performance comparisons, and PRIM scenario discovery. |
| `MORDM_Analysis/Reevaluation_Selected_Strategies_K1000_5000_10000/` | Independent uncertainty ensembles, future-level strategy results, summary statistics, and comparisons across K=1,000, 5,000, and 10,000. | Evaluates the stability of irrigation and yield outcomes across uncertainty sample sizes. |
| `MORDM_Analysis/Sobol_Results/` | Sobol sampling designs, simulation results, first-order and total-order sensitivity indices, second-order interactions, and convergence summaries. | Supports global sensitivity analysis and sensitivity convergence assessment. |
| `MORDM_Analysis/PRIM_Results/` | PRIM peeling trajectory, candidate-box comparisons, selected vulnerability-region limits, and future-level performance and box membership. | Supports scenario discovery and identification of conditions associated with Strategy S3 failures. |

