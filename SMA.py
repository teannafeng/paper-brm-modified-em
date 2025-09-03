"""
This script contains the support module (SMA) for the Python 
implementation of the modified expectation-maximization (EM) 
algorithm proposed in the manuscript:

"A Modified Expectation-Maximization Algorithm for Accelerated 
Item Response Theory Model Estimation with Large Datasets"

submitted to *Behavior Research Methods* (Manuscript ID: BR-Org-25-740).

This script defines supporting functions that are imported by 
the main algorithm implementation (`v7_MODIFIED_EM.py`).

Code written by: 
- Tianying (Teanna) Feng, University of California, Los Angeles

Manuscript authored by
- Tianying (Teanna) Feng, University of California, Los Angeles
- Li Cai, University of California, Los Angeles

Usage:
This module is not intended to be run directly. 
Its functions are imported by `v7_MODIFIED_EM.py`, 
which in turn is called by the simulation runner scripts.

Notes:
- This script is provided as part of the open code materials for the above manuscript. 
- It should be used in conjunction with `v7_MODIFIED_EM.py` and the simulation scripts.
"""

import numpy as np

class SMA:
    def __init__(
            self, 
            num_items: int, 
            wait: int = 30, 
            window_check: int = 10, 
            tol: float = 1e-6
        ):
        self.num_items = num_items
        self.wait = np.full(num_items, wait, dtype=int)
        self.window_check = window_check
        self.tol = tol
        self.stabilized = np.zeros(num_items, dtype=bool)
        self.est_vals = [[] for _ in range(num_items)]
        self.center_vals = [[] for _ in range(num_items)]

    def check_stable(self, vals):
        if len(vals) < self.window_check:
            return False
        return np.var(vals[-self.window_check:]) <= self.tol

    def update_sma(self, new_values):
        smoothed = []
        for i, val in enumerate(new_values):
            if self.stabilized[i]:
                smoothed.append(self.center_vals[i][-1])
                continue

            self.wait[i] -= 1
            if self.wait[i] > 0:
                smoothed.append(val)
                continue

            self.est_vals[i].append(val)
            center = np.mean(self.est_vals[i])
            self.center_vals[i].append(center)

            if not self.stabilized[i] and self.check_stable(self.center_vals[i]):
                self.stabilized[i] = True

            smoothed.append(self.center_vals[i][-1])
        return np.array(smoothed)