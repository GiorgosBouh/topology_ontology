# Topology Comparison Tools for H1 Persistence  
**Version:** 0.1.0  

This repository contains Python utilities for comparing **topological complexity** (H1 total persistence) between different groups (e.g., **post-stroke vs healthy**) using summary CSV files.

The scripts were designed for datasets where each row corresponds to a subject × side × variable combination, with precomputed H1 metrics (e.g., from persistence diagrams / TDA analysis).

---

## Contents

- `compare_topology.py`  
  Group-level comparison of **post-stroke vs healthy**, averaging across sides per subject and variable.

- `compare_topology_by_side.py`  
  **Side-aware** comparison:
  - post-stroke paretic (P) vs post-stroke non-paretic (N)
  - each of those vs healthy (Bilateral, B)  
  for each variable’s H1 total persistence.

Both scripts:
- Generate **CSV summaries** with statistics and effect sizes.
- Save **boxplots** and **violin plots** per variable / comparison.
- Apply **multiple-comparison correction** (Benjamini–Hochberg FDR) on p-values.

---

## Input Data Format

Both scripts expect as input one or more CSV files (typically named `all_subjects_summary.csv`) with at least the following columns:

- `subject` – subject identifier (any string; numeric part is normalized internally).
- `side` – side code (e.g. `P`, `N`, `B`, or similar), mainly used in the side-aware script.
- `variable` – name of the biomechanical / signal variable (e.g. `KneeFlexion`, `AnkleMoment`).
- `h1_points` – (optional) number of H1 points in the persistence diagram.
- `h1_total_persistence` – **numeric** value of total H1 persistence (this is what is analyzed).

Additional columns are ignored.

For **post-stroke vs healthy**:

- `--post`    = CSV from the post-stroke export (`all_subjects_summary.csv`)
- `--healthy` = CSV from the healthy export (`all_subjects_summary.csv`)

For the **side-aware** script, the post-stroke file should contain side-specific variables, and the healthy file typically contains bilateral variables. The script internally normalizes variable names (e.g., strips prefixes like `pside_` / `nside_`) so that matching variables can be compared.

---

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://scipy.org/) (for `scipy.stats`)
- [statsmodels](https://www.statsmodels.org/) (for multiple-comparison correction)

You can install them with:

```bash
pip install numpy pandas matplotlib scipy statsmodels
