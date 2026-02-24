# Topological Comparison Tools for H1 Persistence  
**Version:** 0.1.0

## Overview
This repository contains Python utilities for statistically comparing **topological complexity** derived from gait-related time series. The scripts analyse **H1 total persistence** (a summary metric from persistent homology) to compare locomotor dynamics between **post-stroke** and **healthy** cohorts.

These tools **do not compute persistent homology** directly. They operate on **summary CSV exports** where each row corresponds to a *subject × side × variable* entry with a precomputed `h1_total_persistence` value.

## Scientific Context (What H1 Total Persistence Means)
**H1 total persistence** summarizes the strength and longevity of 1-dimensional topological features (loops) in a persistence diagram, typically obtained from a phase-space / state-space representation of a time series.

In this study context:
- Higher H1 total persistence generally reflects **greater dynamical irregularity/complexity** in the signal.
- Preserved or unchanged H1 total persistence across groups may suggest **stabilized/compensated output dynamics** despite internal variability.

## Repository Contents
### 1) `compare_topology.py`
**Group-level comparison** (post-stroke vs healthy), typically averaging across sides per subject and variable.

Outputs:
- CSV summary tables (descriptives, p-values, corrected p-values, effect sizes)
- Per-variable visualizations (boxplots and violin plots)
- Multiple-comparison correction (Benjamini–Hochberg FDR)

### 2) `compare_topology_by_side.py`
**Side-aware comparisons**, including:
- Post-stroke Paretic (P) vs Non-Paretic (N)
- Paretic (P) vs Healthy (Bilateral/B)
- Non-Paretic (N) vs Healthy (Bilateral/B)

Notes:
- The script harmonizes variable naming (e.g., removing prefixes such as `pside_` / `nside_`) so that homologous variables can be matched across datasets.

Outputs:
- CSV summary tables with statistics and effect sizes
- Per-variable visualizations (boxplots and violin plots)
- Benjamini–Hochberg FDR correction on p-values

## Input Data Format
Both scripts expect one or more CSV files (often named `all_subjects_summary.csv`) containing precomputed topological metrics.

### Required Columns
- `subject`  
  Subject identifier (string). If numeric IDs are embedded, the scripts may normalize/extract numeric parts internally.
- `side`  
  Side code (e.g., `P`, `N`, `B` or similar). Used by the side-aware script.
- `variable`  
  Biomechanical / neuromuscular variable name (e.g., `HipAngles`, `KneeAngles`, `VL_EMG`, `vGRF`).
- `h1_total_persistence`  
  Numeric H1 total persistence value (primary analysis metric).

### Optional Columns
- `h1_points`  
  Number of H1 points/features in the persistence diagram (if exported).

Additional columns are ignored.

## Expected Inputs
### Group comparison (post-stroke vs healthy)
Provide two CSV files:
- `--post`    Post-stroke summary CSV
- `--healthy` Healthy controls summary CSV

### Side-aware analysis
- Post-stroke CSV should contain side-separated measurements (e.g., paretic vs non-paretic)
- Healthy CSV typically contains bilateral measurements (often labeled `B` or similar)
- The script normalizes variable names to enable matching across cohorts

## Statistical Outputs
Both scripts compute, per variable and comparison:
- Descriptive statistics (e.g., mean, SD)
- Hypothesis testing (e.g., t-tests via `scipy.stats`)
- Effect sizes (Cohen’s *d*)
- Benjamini–Hochberg FDR correction (via `statsmodels`)
- Distribution plots (boxplots and violin plots)

## Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
- statsmodels

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy statsmodels