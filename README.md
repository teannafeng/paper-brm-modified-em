This repository contains the supplemental files and code for the manuscript submitted to *Behavior Research Methods* on 2025-09-03.

## Manuscript information

- **Manuscript title:** A Modified Expectation-Maximization Algorithm for Accelerated Item Response Theory Model Estimation with Large Datasets
- **Manuscript ID:** BR-Org-25-740

## Quick start

### Set up a virtual environment and install dependencies

```bash
# Create virtual environment
python3 -m venv .venv       # if on Mac/Linux
python -m venv .venv        # if on Windows

# Activate it
source .venv/bin/activate   # if on Mac/Linux
.venv\Scripts\activate      # if on Windows

# Install dependencies
pip install -r requirements.txt
```

### Run a simulation

From the project directory, run one of the simulation scripts:

```bash
python3 -m v7_RUN_SIMULATION_1 # if on Mac/Linux
python -m v7_RUN_SIMULATION_1  # if on Windows
```

## Notes

- Each file includes a header comment describing its purpose and usage.
- Running the data simulator scripts directly will generate an example `Data` folder with sample outputs.
- The simulation runner scripts used the same random seeds (e.g., `prm_seed=2025`) as those in the manuscript.

## Folder structure

Details are provided in the header comments of individual files.

```text
paper-brm-modified-em/
│
├── v7_MODIFIED_EM.py        # python implementation of the modified EM algorithm
├── SMA.py                   # support module with helper functions
│
├── SIM_2PL.py               # data simulator for Simulation 1 and 2
├── SIM_2PL_FORM.py          # data simulator for Simulation 3
│
├── v7_RUN_SIMULATION_1.py   # runner script for Simulation 1
├── v7_RUN_SIMULATION_2.py   # runner script for Simulation 2
├── v7_RUN_SIMULATION_3.py   # runner script for Simulation 3
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt         # package dependencies
```
