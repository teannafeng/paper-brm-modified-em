"""
This script defines the item response data simulator for Simulation 1 
and Simulation 2 described in the manuscript:

"A Modified Expectation-Maximization Algorithm for Accelerated 
Item Response Theory Model Estimation with Large Datasets"

submitted to *Behavior Research Methods* (Manuscript ID: BR-Org-25-740).

The simulated data are used by simulation scripts:
- v7_RUN_SIMULATION_1.py
- v7_RUN_SIMULATION_2.py

Code written by: 
- Tianying (Teanna) Feng, University of California, Los Angeles

Manuscript authored by:
- Tianying (Teanna) Feng, University of California, Los Angeles
- Li Cai, University of California, Los Angeles

Usage:
Import from the simulation runner scripts, or run directly from the project directory.

On Windows:
    python -m SIM_2PL

On Mac/Linux:
    python3 -m SIM_2PL

Running this script directly will create an example Data folder with the following files:

1. Full item response data: 
   ./Data/[DATE]/NORM/[CONDITION STRING]/IRD_[CONDITION STRING].parquet

2. Item parameter data: 
   ./Data/[DATE]/NORM/PRM_[CONDITION STRING].txt

Notes:
- This script is provided as part of the open code materials for the above manuscript.
"""

import numpy as np
import pandas as pd
import os
import re

class SIM_2PL:
    def __init__(self, num_items, num_persons):
        self.num_items   = num_items
        self.num_persons = num_persons
        self.prm_seed    = None
        self.theta_seed  = None
        self.item_seed   = None
        self.theta       = None
        self.a           = None
        self.b           = None
        self.c           = None
        self.a_levels    = None
        self.b_levels    = None
        self.pattern     = None
        self.freq        = None
        self.data        = None
        self.check       = None

    def get_item_params(self):
        return self.a, self.c
    
    def get_num_unique_items(self):
        return self.num_items
    
    def get_num_items(self):
        return self.num_items

    def simulate(self, 
                 prm_seed: int, 
                 theta_seed: int,
                 item_seed: int,
                 a_levels = {'low': [ 0.5, 1.5],  'medium': [1.5,  2.5], 'high': [2.5, 3.5]}, 
                 b_levels = {'low': [-3.0, -1.5], 'medium': [-1.5, 1.5], 'high': [1.5, 3.0]}):
        grid = [
            (a_key, b_key, a_levels[a_key], b_levels[b_key])
            for a_key in a_levels
            for b_key in b_levels
        ]

        # Save seeds
        self.prm_seed   = prm_seed
        self.theta_seed = theta_seed
        self.item_seed  = item_seed

        # Randomly sample item parameters from a specific grid of low, med, high value levels
        self.a, self.b, self.c, self.a_levels, self.b_levels = self.sample_item_params(grid, self.prm_seed)

        # Randomly sample theta values from N(0,1) if not provided
        self.theta = self.sample_theta_standard_normal(self.theta_seed)

        # Store simulated data
        self.data, self.pattern, self.freq = self.simulate_item_data(
            self.num_persons, self.num_items, self.a, self.c, self.theta, item_seed
        )

        # Store checks 
        self.check = self._count_pairs_in_grid_cell(self.a, self.c, a_levels, b_levels)

    def simulate_with_prm(self, prm: dict, theta_seed: int, item_seed: int):
        # Randomly sample theta values from N(0,1) if not provided
        self.theta = self.sample_theta_standard_normal(theta_seed)
        self.theta_seed = theta_seed
        self.prm_seed   = 'FIXED'

        # Store input args
        self.a = np.array(prm['a'])
        self.c = np.array(prm['c'])
        self.b = -self.c / self.a

        # Store simulated data
        self.data, self.pattern, self.freq = self.simulate_item_data(
            self.num_persons, self.num_items, self.a, self.c, self.theta, item_seed
        )

    def simulate_item_data(self, 
                           num_persons: int, 
                           num_items: int, 
                           a: np.ndarray, 
                           c: np.ndarray, 
                           theta: np.ndarray,
                           rnd_seed: int):
        
        np.random.seed(rnd_seed)

        # Get linear predictor values
        person_idx = np.repeat(list(range(num_persons)), num_items) 
        item_idx   = np.tile(list(range(num_items)), num_persons)
        z          = a[item_idx] * theta[person_idx] + c[item_idx]

        # Get item response data
        response_flat = np.random.binomial(1, p = 1 / (1 + np.exp(-z)))
        response_matrix = pd.DataFrame(response_flat.reshape(num_persons, num_items))

        # Get response patterns
        pattern = response_matrix.value_counts().reset_index(name="r")

        # Get response frequencies for all response patterns
        freq = pattern.iloc[:, -1].to_numpy()
        pattern = pattern.drop(columns=['r']).to_numpy()

        return response_matrix, pattern, freq

    def sample_theta_standard_normal(self, rnd_seed: int):
        np.random.seed(rnd_seed)
        return np.random.randn(self.num_persons)

    def sample_item_params(self, grid: list, rnd_seed: int):
        np.random.seed(rnd_seed)
        samples_per_cell = self.num_items // len(grid)  # Evenly divide
        extra_samples = self.num_items % len(grid)      # Handle any remainder
        a = []
        b = []
        a_levels = []
        b_levels = []
        for i, (a_key, b_key, a_range, b_range) in enumerate(grid):
            # Determine how many samples to draw for this grid cell
            n_samples = samples_per_cell + (1 if i < extra_samples else 0)
            for _ in range(n_samples):
                # Sample A and B values uniformly from the ranges
                a_sample = np.random.uniform(*a_range)
                b_sample = np.random.uniform(*b_range)
                a.append(a_sample)
                b.append(b_sample)
                a_levels.append(a_key)
                b_levels.append(b_key)
        a = np.array(a)
        b = np.array(b)
        c = -a * b
        a_levels = np.array(a_levels)
        b_levels = np.array(b_levels)
        
        return a, b, c, a_levels, b_levels

    def _find_grid_cell_location(self, value: float, grid: list):
        for key, (low, high) in grid.items(): # type: ignore
            if low <= value < high:
                return key
        return None

    def _count_pairs_in_grid_cell(
            self, 
            a: np.ndarray, 
            b: np.ndarray, 
            a_levels: np.ndarray, 
            b_levels: np.ndarray
        ):
        counts_per_cell = {(a, b): 0 for a in a_levels for b in b_levels}
        for a_val, b_val in zip(a, b):
            a_cell = self._find_grid_cell_location(a_val, b_levels)
            b_cell = self._find_grid_cell_location(b_val, b_levels)
            if a_cell and b_cell:
                counts_per_cell[(a_cell, b_cell)] += 1
        return counts_per_cell
    
    def _create_folder(self, path: str):
        if not os.path.exists(path):
            print(f'Create folder at {path}')
            os.makedirs(path)
            return True
        else:
            print(f'Already exists: {path}')
            return False

    def save_prm_data(self, folder_path: str, fname_base: str, fixed_prm: bool = True):
        if fixed_prm:
            par_fname = '/PRM_' + re.sub(r'_TS\d+', '', fname_base) + '.txt'
        else:
            par_fname = '/PRM_' + fname_base + '.txt'

        fname = f"{folder_path}/{par_fname}"
        
        if not os.path.exists(fname):
            par = {
                'item_id' : range(1, self.num_items + 1),
                'a_true'  : self.a,
                'b_true'  : self.b,
                'c_true'  : self.c,
                'a_lvl'   : self.a_levels,
                'b_lvl'   : self.b_levels,
                'rnd_seed': self.prm_seed,
            }
            par = pd.DataFrame(par)
            par.to_csv(fname, index=False, header=False, sep='\t', float_format='%+0.6f')
            print(f'PRM saved to {fname}')
        else:
            print(f'Skipped: PRM already exists at {fname}')

    def save_ird_data(self, folder_path: str, fname_base: str):
        # Organize data
        ird = self.data.copy() # type: ignore
        ird.columns = ['v' + str(i+1) for i in range(ird.shape[1])]
        ird['theta'] = self.theta

        # Save data
        fname = f"{folder_path}/{fname_base}/IRD_{fname_base}.parquet"
        ird.to_parquet(fname, index=False, engine="pyarrow", compression="snappy")
        print(f"IRD saved to {fname}")

    def save(self, folder_path: str, fixed_prm: bool = True):
        iname       = 'I'  + '{:03d}'.format(self.num_items)
        pname       = 'P'  + str(self.num_persons)
        sname_prm   = 'PS' + str(self.prm_seed)
        sname_theta = 'TS' + str(self.theta_seed)
        fname_base = f"{iname}_{pname}_{sname_prm}_{sname_theta}"

        # Check if folder exists; if not, create it
        _ = self._create_folder(f"{folder_path}/{fname_base}")

        # Save item response data and parameter data
        self.save_ird_data(folder_path, fname_base)
        self.save_prm_data(folder_path, fname_base, fixed_prm=fixed_prm)

        return fname_base

if __name__ == '__main__':
    from datetime import date

    # Set folder path
    today = date.today()
    folder_path = f"./Data/{str(today)}/NORM/"

    # Create simulator
    sim = SIM_2PL(num_items=10, num_persons=8000)

    # Simulate data
    sim.simulate(prm_seed=1010, theta_seed=1010, item_seed=1010)

    # Save simulated data
    sim.save(folder_path=folder_path, fixed_prm=False)
    


