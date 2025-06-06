# Political Polarization and Economic Factors Analysis

This project analyzes the relationship between income inequality (GINI), inflation (CPI), and political polarization in OECD countries. **Please refer to the Juypter Notebook for the complete data processing steps, detailed methodology, statistical analysis, interpretation of results, key findings, and discussion of limitations :)**

## Repository Structure
```text
├── data/                     # Raw input data files (e.g., MPDataset_MPDS2024a.csv, OECD_gini_cpi_data.csv)
├── notebooks/                # Jupyter notebooks for analysis (e.g., Polarization_Analysis.ipynb)
├── output/                   # Generated files (e.g., merged_political_oecd_data.csv, plots, analysis_ready_data.csv)
├── scripts/                  # Python scripts for data processing (e.g., merge_data.py, analyze_polarization.py)
├── .gitignore                # Specifies intentionally untracked files that Git should ignore
└── README.md                 # This file

```
## Setup and How to Run

**Prerequisites:**
* Python 3.x
* The following Python libraries:
    * `pandas`
    * `numpy`
    * `statsmodels`
    * `matplotlib`
    * `seaborn`
    * `jupyterlab` (recommended for running the `.ipynb` notebook, includes `notebook`)

**How to Run**
* Run Data Merging Script `scripts/merge_data.py`
* Run Data Analysis Script `scripts/analyse_polarisation.py`

## Data Sources and Citations
This project utilizes the following primary data sources:

**Manifesto Project Dataset:**

Citation: Lehmann, Pola / Franzmann, Simon / Al-Gaddooa, Denise / Burst, Tobias / Ivanusch, Christoph / Regel, Sven / Riethmüller, Felicia / Volkens, Andrea / Weßels, Bernhard / Zehnter, Lisa (2024): The Manifesto Data Collection. Manifesto Project (MRG/CMP/MARPOR). Version 2024a. Berlin: Wissenschaftszentrum Berlin für Sozialforschung (WZB) / Göttingen: Institut für Demokratieforschung (IfDem). https://doi.org/10.25522/manifesto.mpds.2024a

The specific file used was MPDataset_MPDS2024a.csv.

**OECD Economic Indicators:**

Data on GINI coefficients and Consumer Price Index (CPI) were sourced from the OECD data portal.

The specific file used in the initial merging process was OECD_gini_cpi_data.csv.
