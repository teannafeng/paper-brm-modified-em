"""
This script defines the item response data simulator for Simulation 4
described in the manuscript:

"A Modified Expectation-Maximization Algorithm for Accelerated 
Item Response Theory Model Estimation with Large Datasets"

submitted to *Behavior Research Methods* (Manuscript ID: BR-Org-25-740).

The simulated data are used by simulation scripts:
- v7_RUN_SIMULATION_4.py

Code written by: 
- Tianying (Teanna) Feng, University of California, Los Angeles

Manuscript authored by:
- Tianying (Teanna) Feng, University of California, Los Angeles
- Li Cai, University of California, Los Angeles

Usage:
Import from the simulation runner scripts, or run directly from the project directory.

On Windows:
    python -m SIM_2PL_TESTLET

On Mac/Linux:
    python3 -m SIM_2PL_TESTLET

Running this script directly will create an example Data folder with the following files:

1. Mapping of items to latent factors (with means and variances): 
   ./Data/[DATE]/TESTLET/IFM_[CONDITION STRING].csv

2. Full item response data:
   ./Data/[DATE]/TESTLET/IRD_[CONDITION STRING].csv

3. Item parameter data: 
   ./Data/[DATE]/TESTLET/PRM_[CONDITION STRING].txt

Notes:
- This script is provided as part of the open code materials for the above manuscript.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
from scipy.special import expit

class SIM_2PL_TESTLET:
    def __init__(
        self,
        num_persons: int = 60_000,
        num_items_per_testlet: int = 10,
        num_testlets: int = 3,
        C: float = 1.0,
    ):
        self.n_persons         = num_persons
        self.items_per_testlet = num_items_per_testlet
        self.n_testlets        = num_testlets
        self.n_items           = num_items_per_testlet * num_testlets
        self.C = C

        self.item_params: Optional[pd.DataFrame]     = None         # PRM_
        self.item_factor_map: Optional[pd.DataFrame] = None         # IFM_
        self.responses: Optional[pd.DataFrame]       = None         # IRD_
        self.theta_g: Optional[np.ndarray]           = None         # LAV_
        self.theta_d: Optional[np.ndarray]           = None         # LAV_

        self.mean_g: Optional[float]                 = None
        self.var_g: Optional[float]                  = None
        self.mean_ds: Optional[list[float]]          = None
        self.var_ds: Optional[list[float]]           = None


    def _make_person_ids(self):
        return [f"p{i+1}" for i in range(self.n_persons)]


    def _make_item_ids(self):
        return [f"i{j+1}" for j in range(self.n_items)]
    

    def _make_testlet_ids(self):
        idx = np.repeat(np.arange(self.n_testlets), self.items_per_testlet)
        labels = [f"d{t+1}" for t in idx]
        return idx, labels


    def _make_item_factor_map(self):
        if self.item_params is None:
            raise RuntimeError("self.item_params is None.")
        if self.mean_ds is None:
            raise RuntimeError("self.mean_ds is None.")
        if self.var_ds is None:
            raise RuntimeError("self.var_ds is None.")
        if self.mean_g is None:
            raise RuntimeError("self.mean_g is None.")
        if self.var_g is None:
            raise RuntimeError("self.var_g is None.")
        
        item_ids = self.item_params["item_id"].tolist()
        testlet_idx = self.item_params["testlet_index"].to_numpy()

        rows = []
        for j, item_id in enumerate(item_ids):
            row = {
                "item_id": item_id,
                "general_mean": self.mean_g,
                "general_var": self.var_g,
            }
            for t in range(self.n_testlets):
                if testlet_idx[j] == t:
                    row[f"testlet{t+1}_mean"] = self.mean_ds[t]
                    row[f"testlet{t+1}_var"] = self.var_ds[t]
                else:
                    row[f"testlet{t+1}_mean"] = np.nan
                    row[f"testlet{t+1}_var"] = np.nan
            rows.append(row)

        self.item_factor_map = pd.DataFrame(rows)


    def _make_fname(self):
        if self.var_ds is None:
            raise RuntimeError("self.var_ds is None.")
        
        if len(set(self.var_ds)) == 1:
            tv = str(self.var_ds[0])
        else:
            tv = ",".join(str(v) for v in self.var_ds)

        return (
            f"TV{tv}_"
            f"P{self.n_persons}_"
            f"T{self.n_testlets}_"
            f"IT{self.items_per_testlet}"
        )


    def _sample_item_parameters(
        self,
        prm_seed: int,
        a_low: float  = 0.5,
        a_high: float = 3.5,
        c_mean: float = 0.0,
        c_sd: float   = 1.0,
    ):
        prm_rng = np.random.default_rng(prm_seed)
        a = prm_rng.uniform(a_low, a_high, size=self.n_items)
        c = prm_rng.normal(c_mean, c_sd, size=self.n_items)
        c = np.clip(c, -10.0, 10.0) # c in [-10.0, 10.0]

        item_ids = self._make_item_ids()
        testlet_idx, testlet_ids = self._make_testlet_ids()

        self.item_params = pd.DataFrame(
            {
                "item_id": item_ids,
                "testlet_index": testlet_idx,
                "testlet_id": testlet_ids,
                "a": a,
                "c": c,
                "constraint": self.C, # constraint on general vs. testlet slopes
            }
        )


    def _sample_general_latent_var(
        self,
        theta_seed: int,
        mean_g: float = 0.0,
        var_g: float  = 1.0
    ):
        self.mean_g = mean_g
        self.var_g = var_g

        theta_rng = np.random.default_rng(theta_seed)
        self.theta_g = theta_rng.normal(mean_g, np.sqrt(var_g), size=self.n_persons)


    def _sample_testlet_latent_vars(
        self,
        mean_ds: list[float],
        var_ds: list[float],
        theta_seed: int
    ):
        if len(mean_ds) != self.n_testlets or len(var_ds) != self.n_testlets:
            raise ValueError("len(mean_ds) != n_testlets or len(var_ds) != n_testlets.")
        
        self.mean_ds = list(mean_ds)
        self.var_ds = list(var_ds)

        theta_rng = np.random.default_rng(theta_seed)
        self.theta_d = np.vstack(
            [
                theta_rng.normal(mean_ds[t], np.sqrt(var_ds[t]), self.n_persons)
                for t in range(self.n_testlets)
            ]
        ).T
    

    def _simulate_item_data(self, item_seed: int):
        if self.item_params is None:
            raise RuntimeError("self.item_params is None.")
        if self.theta_g is None:
            raise RuntimeError("self.theta_g is None.")
        if self.theta_d is None:
            raise RuntimeError("self.theta_d is None.")

        item_ids = self.item_params["item_id"].tolist()
        person_ids = self._make_person_ids()

        a = self.item_params["a"].to_numpy()
        c = self.item_params["c"].to_numpy()
        testlet_idx = self.item_params["testlet_index"].to_numpy()

        Z = np.zeros((self.n_persons, self.n_items))
        for i in range(self.n_items):
           Z[:, i] = (
                a[i] * (self.theta_g + self.C * self.theta_d[:, testlet_idx[i]])
                + c[i]
           )
        P = expit(Z)

        item_rng = np.random.default_rng(item_seed)
        Y = item_rng.binomial(1, P)

        self.responses = pd.DataFrame(Y, columns=item_ids)
        self.responses.insert(0, "person_id", person_ids)


    def simulate(
        self,
        prm_seed  : int,
        item_seed : int,
        theta_seed: int,
        mean_ds   : list[float],
        var_ds    : list[float],
        a_low     : float = 0.5,
        a_high    : float = 3.5,
        c_mean    : float = 0.0,
        c_sd      : float = 1.0,
        mean_g    : float = 0.0,
        var_g     : float = 1.0,
    ):
        self._sample_item_parameters(a_low=a_low, a_high=a_high, c_mean=c_mean, c_sd=c_sd, prm_seed=prm_seed)
        self._sample_general_latent_var(mean_g=mean_g, var_g=var_g, theta_seed=theta_seed)
        self._sample_testlet_latent_vars(mean_ds=mean_ds, var_ds=var_ds, theta_seed=theta_seed)
        self._make_item_factor_map()
        self._simulate_item_data(item_seed=item_seed)

        if self.responses is None:
            raise RuntimeError("self.responses is None.")
        if self.item_params is None:
            raise RuntimeError("self.item_params is None.")

        IRD = self.responses.drop(columns=["person_id"]).to_numpy()
        PRM_A = self.item_params["a"].to_numpy()
        PRM_C = self.item_params["c"].to_numpy()

        return IRD, PRM_A, PRM_C


    def save(
            self, folder_path: str, 
            file_suffix: str = "", 
            as_parquet: bool = False,
            save_ird: bool   = True,
            save_prm: bool   = True,
            save_lav: bool   = False,
            save_ifm: bool   = True,
        ):
        if self.responses is None:
            raise RuntimeError("self.responses is None.")
        if self.item_params is None:
            raise RuntimeError("self.item_params is None.")
        if self.theta_g is None:
            raise RuntimeError("self.theta_g is None.")
        if self.theta_d is None:
            raise RuntimeError("self.theta_d is None.")
        if self.item_factor_map is None:
            raise RuntimeError("self.item_factor_map is None.")

        os.makedirs(folder_path, exist_ok=True)

        fname_base = self._make_fname()

        lav = pd.DataFrame({"person_id": self.responses["person_id"], "theta_g": self.theta_g})
        for t in range(self.n_testlets):
            lav[f"theta_d{t+1}"] = self.theta_d[:, t]

        if as_parquet:
            if save_ird:
                ird_path = os.path.join(folder_path, f"IRD_{fname_base}{file_suffix}.parquet")
                self.responses.to_parquet(ird_path, index=False)

            if save_lav:
                lav_path = os.path.join(folder_path, f"LAV_{fname_base}{file_suffix}.parquet")
                lav.to_parquet(lav_path, index=False)
        else:
            if save_ird:
                ird_path = os.path.join(folder_path, f"IRD_{fname_base}{file_suffix}.csv")
                self.responses.to_csv(ird_path, index=False)

            if save_lav:
                lav_path = os.path.join(folder_path, f"LAV_{fname_base}{file_suffix}.csv")
                lav.to_csv(lav_path, index=False)
        
        if save_prm:
            prm_path = os.path.join(folder_path, f"PRM_{fname_base}.txt")
            if not os.path.exists(prm_path):
                self.item_params.to_csv(prm_path, index=False, sep='\t', float_format='%+0.6f')

        if save_ifm:
            ifm_path = os.path.join(folder_path, f"IFM_{fname_base}.csv")
            if not os.path.exists(ifm_path):
                self.item_factor_map.to_csv(ifm_path, index=False)

if __name__ == "__main__":
    from datetime import date

    # Set folder path
    today = date.today()
    folder_path = f"./Data/{str(today)}/TESTLET/"

    # Set configs
    n_persons         = 60_000
    items_per_testlet = 10
    n_testlets        = 3
    mean_g            = 0.0
    var_g             = 1.0
    mean_ds           = [0.0] * n_testlets

    # Set simulation cases
    cases = {
        "c1": [0.05] * n_testlets,
        "c2": [0.10] * n_testlets,
        "c3": [0.20] * n_testlets,
        "c4": [0.50] * n_testlets,
    }

    # Simulate data
    for case_name, var_ds in cases.items():
        sim = SIM_2PL_TESTLET(
            num_persons           = n_persons,
            num_items_per_testlet = items_per_testlet,
            num_testlets          = n_testlets,
            C                     = 1.0,
        )

        IRD, PRM_A, PRM_C = sim.simulate(
            prm_seed   = 2025,
            theta_seed = 2025,
            item_seed  = 2025,
            mean_ds    = mean_ds,
            var_ds     = var_ds,
            a_low      = 0.5,
            a_high     = 3.5,
            c_mean     = 0.0,
            c_sd       = 1.0,
            mean_g     = mean_g,
            var_g      = var_g,
        )

        sim.save(folder_path=folder_path, as_parquet=False)
