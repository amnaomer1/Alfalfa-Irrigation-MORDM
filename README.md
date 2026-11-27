
# Balancing Water Conservation and Crop Production for Robust Alfalfa Irrigation under Uncertainty

**Manuscript status:** Currently being prepared for submission to *Agricultural Water Management*.

## 1. Overview

This repository contains the Python code, supporting data, and analysis workflows developed for the study *Balancing Water Conservation and Crop Production for Robust Alfalfa Irrigation under Uncertainty*, which evaluates alfalfa irrigation management strategies under weather and biophysical uncertainty in California's Imperial Valley.

The study integrates a daily, field-scale alfalfa simulation model with a Many-Objective Robust Decision Making (MORDM) framework. The simulation model was adapted from the Water Balance Model (WBM; Grogan et al., 2022) and pyWBM (Lafferty et al., 2025) to represent soil-water dynamics, irrigation decisions, crop evapotranspiration, water stress, and yield responses over a five-year production period. The analysis explores 16 management decision variables and 13 uncertain parameters.

The analytical workflow includes:

- **Multi-objective optimization:** Using the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to identify irrigation–yield tradeoffs.
- **Representative strategy selection:** Using K-means clustering to select five representative irrigation strategies along the Pareto front.
- **Performance and robustness assessment:** Evaluating representative strategies across 10,000 independently sampled states of the world (SOWs), separate from the uncertainty ensemble used during optimization.
- **Global sensitivity analysis:** Using Sobol' variance decomposition to identify uncertainties that most strongly influence irrigation and yield performance.
- **Scenario discovery:** Using the Patient Rule Induction Method (PRIM) to characterize combinations of uncertain conditions associated with the vulnerability of the most robust representative irrigation strategy (S3).
- **Model evaluation, convergence, and stability analyses:** Assessing model performance and the numerical reliability of the analytical results.

The repository includes Python scripts, Jupyter notebooks, model inputs, and supporting analysis outputs used to reproduce the results, figures, and tables presented in the manuscript and supplementary material.

## 2. Repository Structure

The repository is organized into the following main components:

```text
Alfalfa-Irrigation-MORDM/
│
├── README.md
├── LICENSE
├── requirements.txt
├── requirements_full.txt
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

The repository also includes the following documentation and software configuration files:

- `README.md`: Repository overview, documentation, and reproducibility instructions.
- `LICENSE`: Software licensing terms.
- `CITATION.cff`: Citation metadata for the repository.
- `requirements.txt`: Main Python package dependencies required to run the model and analyses.
- `requirements_full.txt`: Complete snapshot of the Python environment used in this study.

The main analysis output directories are documented separately in [Section 7: Expected Outputs](#7-expected-outputs).

## 3. Data Sources and Inputs

The study uses climate, soil, crop, and model evaluation datasets to simulate alfalfa irrigation and production in California's Imperial Valley.

The main data sources and input categories include:

- **Climate data:** ERA5 meteorological data (2000–2023), originally produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) and distributed through the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/). The data were accessed through the Water Systems Analysis Group (WSAG) data server at the University of New Hampshire (UNH) and used to calculate reference evapotranspiration and represent historical weather variability.

- **Soil data:** The USDA [Soil Survey Geographic (SSURGO) Database](https://websoilsurvey.nrcs.usda.gov/) and published regional soil-property information, used to characterize soil-water storage, hydraulic properties, and root-zone water availability.

- **Crop parameters:** Published literature and agricultural management guidelines, used to parameterize alfalfa growth, crop coefficients, water-stress response, water productivity, and yield-response relationships.

- **Model evaluation data:**
  - [California Irrigation Management Information System (CIMIS)](https://cimis.water.ca.gov/): Reference evapotranspiration and meteorological data.
  - [OpenET](https://openetdata.org/): Satellite-based estimates of actual crop evapotranspiration.
  - California Simulation of Evapotranspiration of Applied Water (Cal-SIMETAW): Estimates of potential crop evapotranspiration.
  - [Imperial County Agricultural Crop and Livestock Reports](https://agcom.imperialcounty.org/crop-reports/): Historical alfalfa yield statistics.

- **Historical irrigation and yield data:** The California Department of Water Resources [Agricultural Land and Water Use Estimates](https://water.ca.gov/Programs/Water-Use-And-Efficiency/Land-And-Water-Use/Agricultural-Land-and-Water-Use-Estimates) and Imperial County Agricultural Crop and Livestock Reports, used to provide historical reference ranges for interpreting the modeled irrigation–yield tradeoffs.

- **Uncertainty ranges and management bounds:** Parameter ranges informed by regional datasets, published literature, agricultural management guidelines, and exploratory modeling assumptions. These ranges define the uncertain conditions and management alternatives evaluated within the MORDM framework.

The model uses daily climate inputs derived from ERA5, including the following file:

`Input/ImperialCounty_ERA5_DailyClimate_2000_2023.csv`

Detailed data sources, references, processing procedures, model parameterization, uncertainty ranges, and management decision bounds are documented in the accompanying manuscript and supplementary material.

## 4. Software Requirements

The analyses use Python 3.11.7. The main Python dependencies are listed in `requirements.txt`, while `requirements_full.txt` provides a complete snapshot of the Python environment, including additional dependencies.

To install the required packages, run:

```bash
python -m pip install -r requirements.txt
```

Users can adjust the number of parallel workers in the relevant scripts according to their available computational resources.

## 5. Analysis Workflow and Reproducibility

The following table summarizes the scripts and notebooks used in this study, their functions, required dependencies, computational resources, and reference runtimes. Stages represent the logical analysis workflow; some stages can run independently once their required inputs are available.

### Terminology and Notation

The following terminology and notation are used throughout the repository:

- **States of the world (SOWs):** Sampled combinations of uncertain weather, soil, and crop conditions.
- **K:** Number of sampled SOWs used for optimization, convergence testing, or strategy performance and robustness evaluation.
- **N:** Base sample size used in the Sobol' global sensitivity analysis.
- **S1–S5:** Five representative irrigation strategies selected from the final Pareto archive using K-means clustering, ordered by increasing mean irrigation application.
- **Pareto front, Pareto solution, and Pareto archive:** These terms refer to numerical approximations obtained through multi-objective optimization, without implying globally verified Pareto optimality.
- **ET₀:** Reference evapotranspiration.
- **PETc:** Potential crop evapotranspiration.
- **AETc:** Actual crop evapotranspiration.

| Stage | Script or notebook | Description | Required inputs and dependencies | Allocated CPU cores | Reference runtime |
|---|---|---|---|---|---|
| 1 | `Baseline_Scenario.ipynb` | Runs the baseline alfalfa irrigation simulation and generates diagnostic figures. | Core model and climate inputs | — | — |
| 2 | `Model_Evaluation.ipynb` | Evaluates simulated ET₀, PETc, AETc, and alfalfa yield against reference datasets. | Core model, climate inputs, and evaluation datasets | — | — |
| 3 | `Convergence_10Seeds_K5000.py` | Evaluates convergence across three fixed irrigation strategies, seven sample sizes (K=100–5,000), and 10 random seeds. | Core model and climate inputs | 48 | 1 h 47 min |
| 4 | `Convergence_10Seeds_K10000.py` | Extends the convergence analysis to K=10,000 across three strategies and 10 random seeds. | Core model and climate inputs | 48 | 2 h 12 min |
| 5 | `convergence plots.ipynb` | Generates convergence plots for irrigation and yield means and distribution percentiles. | Stages 3–4 | — | — |
| 6 | `NSGA2_parallelization_OneSeed_100Gen.py` | Runs the main NSGA-II optimization (100 individuals, 100 generations, K=300) to identify approximately Pareto-efficient irrigation strategies. | Core model and climate inputs | 48 | 21 h 27 min |
| 7 | `NSGA2_parallelization_OneSeed_200Gen.py` | Extends NSGA-II optimization to 200 generations to assess Pareto-front convergence. | Core model and climate inputs | 48 | 43 h 19 min |
| 8 | `NSGA2_parallelization-MultiSeeds.py` | Runs NSGA-II with three additional random seeds (102–104) to assess optimization consistency. | Core model and climate inputs | 48 | 64 h 5 min |
| 9 | `Reoptimization_NSGA2_K700.py` | Repeats optimization with K=700 to assess sensitivity to uncertainty sample size. | Core model and climate inputs | 48 | 49 h 55 min 36 s |
| 10 | `Reoptimization_NSGA2_K1000.py` | Repeats optimization with K=1,000 to assess sensitivity to uncertainty sample size. | Core model and climate inputs | 48 | 71 h 50 min 18 s |
| 11 | `Select_Representative_Strategies.py` | Selects five representative strategies (S1–S5) using K-means clustering of the final Pareto archive. | Stage 6 final Pareto archive | — | — |
| 12 | `Strategy_performance_5strategies.py` | Evaluates five strategies across K=10,000 and optionally K=50,000 SOWs to assess irrigation, yield, robustness, and outcome stability. | Stage 11 CSV; Stage 6 optimization functions | 48 | 15 min 33 s (K=10,000); 1 h 17 min 18 s (K=50,000) |
| 13 | `Reevaluate_Selected_Strategies_K_1000_5000_10000.py` | Reevaluates five strategies across K=1,000, 5,000, and 10,000 to assess the stability of irrigation and yield outcomes. | Stage 11 CSV; Stage 6 optimization functions | 48 | 25 min 58 s |
| 14 | `Strategy_Irrigation_Diagnostics.py` | Quantifies routine, forced, and survival irrigation and examines seasonal irrigation patterns across 300 SOWs. | Stage 11 CSV; Stage 6 design SOWs; core model and climate inputs | 48 | 35 s |
| 15 | `Sobol_Analysis.py` | Calculates first-order, total-order, and second-order Sobol sensitivity indices for irrigation and yield across 13 uncertain parameters. | Stage 11 CSV; Stage 6 optimization functions | 48 | N=512: 22 min 25 s; N=1,024: 44 min 41 s; N=2,048: 1 h 28 min 39 s |
| 16 | `PRIM_Scenario_Discovery.py` | Identifies vulnerability regions for 	Strategy 3 (S3) using PRIM across 10,000 sampled SOWs. | Stage 12 K=10,000 outputs | — | — |
| 17 | `Manuscript_Figures_and_Tables.ipynb` | Reproduces manuscript Figures 2–5 and Table 3 using previously generated analysis outputs. | Optimization, performance, Sobol, and PRIM outputs | — | — |
| 18 | `Supplementary_Analysis_and_Figures.ipynb` | Reproduces supplementary Figures S13–S21 and Tables S11–S13 using previously generated analysis outputs. | Convergence, optimization, robustness, Sobol, and PRIM outputs | — | — |

**Notes:** Reference runtimes represent wall-clock durations recorded using SLURM on Penn State's Roar high-performance computing (HPC) system, with 48 CPU cores allocated for most computationally intensive analyses. The number of active parallel workers may differ from the number of allocated CPU cores and can be adjusted within the Python scripts. For example, the Sobol' sensitivity analysis uses 14 parallel workers by default. Actual runtimes will vary depending on hardware specifications, the number of parallel workers, and parallelization efficiency.

In the workflow table, a dash (—) in the reference runtime column indicates a runtime of less than one minute, while a dash in the Allocated CPU cores column indicates that no specific CPU allocation was recorded.

## 6. How to Reproduce the Results

The repository supports two approaches to reproducing the study results:

- **Reproducing figures and tables using existing analysis outputs**, without rerunning the computationally intensive simulations.
- **Reproducing the complete analysis from the original model inputs**, including model evaluation, optimization, robustness assessment, sensitivity analysis, and scenario discovery.

### 6.1. Environment Setup

Clone the repository:

```bash
git clone https://github.com/amnaomer1/Alfalfa-Irrigation-MORDM.git
cd Alfalfa-Irrigation-MORDM
```

Set up the Python environment and install the required packages as described in [Section 4](#4-software-requirements).

Before running the scripts or notebooks, ensure that the required input datasets and analysis outputs are available and that file paths are consistent with the repository structure. Adjust machine-specific paths and parallel worker settings where necessary.

### 6.2. Reproducing Figures and Tables Using Existing Outputs

When the required analysis outputs are available, the manuscript and supplementary figures and tables can be regenerated without rerunning the computationally intensive simulations.

Open and execute the relevant Jupyter notebooks:

| Notebook | Purpose |
|---|---|
| `Baseline_and_Model_Evaluation/Baseline_Scenario.ipynb` | Reproduces the baseline simulation and associated diagnostic figures. |
| `Baseline_and_Model_Evaluation/Model_Evaluation.ipynb` | Reproduces model evaluation analyses and figures. |
| `Pre_Optimization_Convergence_Tests/convergence plots.ipynb` | Generates convergence plots using previously generated convergence outputs. |
| `MORDM_Analysis/Manuscript_Figures_and_Tables.ipynb` | Reproduces manuscript Figures 2–5 and Table 3. |
| `MORDM_Analysis/Supplementary_Analysis_and_Figures.ipynb` | Reproduces supplementary Figures S13–S21 and Tables S11–S13. |

The required inputs and analysis outputs for each notebook are described in the workflow table in Section 5 and the output directory descriptions in Section 7.

If a required output is unavailable, regenerate it using the corresponding Python script before executing the notebook.

### 6.3. Reproducing the Complete Computational Analysis

To reproduce the complete analysis from the original model inputs, follow the workflow and dependencies described in Section 5.

The main analytical steps are:

1. Run the baseline simulation and model evaluation.
2. Conduct pre-optimization convergence tests to assess the stability of irrigation and yield estimates across uncertainty sample sizes.
3. Run the main NSGA-II optimization using 300 SOWs, a population size of 100, and 100 generations.
4. Conduct additional optimization convergence and stability tests using larger uncertainty ensembles, additional generations, and different random seeds.
5. Select five representative irrigation strategies (S1–S5) from the final Pareto archive using K-means clustering.
6. Evaluate representative strategy performance and robustness across 10,000 SOWs and conduct additional uncertainty-sampling stability tests.
7. Analyze seasonal irrigation patterns and irrigation components.
8. Conduct Sobol' global sensitivity analysis and sensitivity convergence tests.
9. Perform PRIM scenario discovery to identify combinations of uncertain conditions associated with Strategy 3 vulnerability.
10. Execute the relevant Jupyter notebooks to generate manuscript and supplementary figures and tables.

Some analytical stages can be executed independently once their required inputs and upstream analysis outputs are available.

### 6.4. Computational Requirements

The complete computational analysis involves computationally intensive simulations, particularly during multi-objective optimization and its associated convergence and stability tests.

The original analyses were conducted using Penn State's Roar high-performance computing (HPC) system. Users can adjust parallel worker settings according to their available computational resources.

Reference runtimes and allocated CPU cores are provided in Section 5. Reproducing figures and tables from existing analysis outputs is substantially less computationally demanding than rerunning the complete analysis.

### 6.5. Large Output Files

Two large SOW-level convergence CSV files associated with the K=100–5,000 and K=10,000 convergence analyses are not included in this GitHub repository because of GitHub's file-size limitations. The corresponding smaller convergence summaries are included.

The complete SOW-level outputs can be regenerated using the following scripts:

- `Pre_Optimization_Convergence_Tests/Convergence_10Seeds_K5000.py`
- `Pre_Optimization_Convergence_Tests/Convergence_10Seeds_K10000.py`

The required inputs, computational resources, and reference runtimes are provided in the analysis workflow table in Section 5.

## 7. Expected Outputs

### Main Output Files and Directories

The following table summarizes the main output directories generated by the analysis workflow. These directories contain simulation results, optimization archives, strategy evaluations, sensitivity indices, and scenario discovery outputs used to reproduce the manuscript and supplementary analyses.

| Output directory | Main contents | Purpose / downstream use |
|---|---|---|
| `Baseline_and_Model_Evaluation/Evaluation_Output/` | Combined daily ET₀, PETc, and AETc results from model evaluation simulations. | Supports model evaluation and comparison with reference datasets. |
| `Pre_Optimization_Convergence_Tests/Output_convergence_Normal_Extreme_VeryExtreme_K5000_10Seeds/` | 	SOW-level simulation results, strategy-level summaries, and convergence statistics for K=100–5,000 across 10 random seeds. | Evaluates uncertainty-sampling convergence before optimization. |
| `Pre_Optimization_Convergence_Tests/Output_convergence_Normal_Extreme_VeryExtreme_K10000_10Seeds/` | SOW-level results and convergence summaries for K=10,000 across 10 random seeds. | Extends the pre-optimization convergence assessment and supports convergence plots. |
| `MORDM_Analysis/NSGA2_K300_P100_G100_S101/` | K=300 sampled SOWs, generation-level populations, Pareto solutions, final Pareto archive, and optimization metadata. | Main optimization output. Provides the Pareto archive for representative strategy selection and sampled SOWs used in subsequent analyses. |
| `MORDM_Analysis/NSGA2_K300_P100_G200_S101/` | Generation-level optimization results, final Pareto solutions, and metadata from the 200-generation run. | Evaluates Pareto-front convergence with additional generations. |
| `MORDM_Analysis/NSGA2_K300_P100_G100_MultiSeeds/` | shared sampled SOWs, seed-specific optimization histories and Pareto archives, and combined solutions for seeds 102–104. | Evaluates consistency of optimization results across random seeds. |
| `MORDM_Analysis/NSGA2_K700_P100_G100_S101_Reoptimization/` | K=700 sampled SOWs, optimization histories, Pareto solutions, and metadata. | Evaluates sensitivity of Pareto solutions to uncertainty sample size. |
| `MORDM_Analysis/NSGA2_K1000_P100_G100_S101_Reoptimization/` | K=1,000 sampled SOWs, optimization histories, Pareto solutions, and metadata. | Extends the uncertainty-sample-size stability assessment. |
| `MORDM_Analysis/Strategies_Performance/` | Five selected irrigation strategies (S1–S5), clustered Pareto archive, SOW-level strategy performance, summary statistics, K=10,000 and K=50,000 uncertainty ensembles, and irrigation diagnostics. | Main strategy evaluation output. Supports robustness analysis, performance comparisons, and PRIM scenario discovery. |
| `MORDM_Analysis/Reevaluation_Selected_Strategies_K1000_5000_10000/` | Independent uncertainty ensembles, SOW-level strategy results, summary statistics, and comparisons across K=1,000, 5,000, and 10,000. | Evaluates the stability of irrigation and yield outcomes across uncertainty sample sizes. |
| `MORDM_Analysis/Sobol_Results/` | Sobol sampling designs, simulation results, first-order and total-order sensitivity indices, second-order interactions, and convergence summaries. | Supports global sensitivity analysis and sensitivity convergence assessment. |
| `MORDM_Analysis/PRIM_Results/` | PRIM peeling trajectory, candidate-box comparisons, selected vulnerability-region limits, and SOW-level performance and box membership. | Supports scenario discovery and identification of conditions associated with Strategy 3 (S3) failures. |

## 8. Citation

If you use the model, code, data, or analysis workflows provided in this repository, please cite the associated manuscript:

Omer, A., Hegde, P., Zuidema, S., Keller, K., Lammers, R., and Grogan, D. (2026). *Balancing Water Conservation and Crop Production for Robust Alfalfa Irrigation under Uncertainty*.

The full citation, including journal publication details and DOI, will be provided once available.

## 9. License

The source code in this repository is distributed under the GNU General Public License v3.0 (GPL-3.0).

See the [LICENSE](LICENSE) file for the full license terms.

The alfalfa simulation model was adapted from the following open-source models, both distributed under GPL-3.0:

- [Water Balance Model (WBM)](https://github.com/wsag/WBM) (Grogan et al., 2022).
- [pyWBM](https://github.com/wsag/pyWBM) (Lafferty et al., 2025).

Users are free to use, modify, and redistribute the software in accordance with the applicable license terms.

Third-party datasets remain subject to their original licensing and data-use conditions.

## 10. Acknowledgements

This work was co-supported by the U.S. Department of Energy, Office of Science, Biological and Environmental Research Program, Earth and Environmental Systems Modeling, MultiSector Dynamics, under Cooperative Agreement DE-SC0022141, and the Thayer School of Engineering at Dartmouth College.

Computationally intensive analyses were conducted using the Roar high-performance computing (HPC) system at Pennsylvania State University.

The development of the alfalfa simulation model builds upon the Water Balance Model (WBM) and pyWBM developed by the Water Systems Analysis Group (WSAG) at the University of New Hampshire.

## 11. Contact

For questions regarding the alfalfa simulation model, analysis workflows, or reproduction of the study results, please contact:

**Amna Omer**  
Water Systems Analysis Group (WSAG)  
University of New Hampshire, Durham, NH, USA  
Email: [amna.omer@unh.edu](mailto:amna.omer@unh.edu)

Technical questions, bug reports, and suggestions for improvements are also welcome through the [GitHub Issues](https://github.com/amnaomer1/Alfalfa-Irrigation-MORDM/issues) section of this repository.


