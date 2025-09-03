"""
This script defines the item response data simulator for Simulation 3
described in the manuscript:

"A Modified Expectation-Maximization Algorithm for Accelerated 
Item Response Theory Model Estimation with Large Datasets"

submitted to *Behavior Research Methods* (Manuscript ID: BR-Org-25-740).

The simulated data are used by simulation scripts:
- v7_RUN_SIMULATION_3.py

Code written by: 
- Tianying (Teanna) Feng, University of California, Los Angeles

Manuscript authored by:
- Tianying (Teanna) Feng, University of California, Los Angeles
- Li Cai, University of California, Los Angeles

Usage:
Import from the simulation runner scripts, or run directly from the project directory.

On Windows:
    python -m SIM_2PL_FORM

On Mac/Linux:
    python3 -m SIM_2PL_FORM

Running this script directly will create an example Data folder with the following files:

1. Subset item response data for quick inspection: 
   ./Data/[DATE]/FORM/[CONDITION STRING]/IRD_CHECK_[CONDITION STRING].csv

2. Full item response data: 
   ./Data/[DATE]/FORM/[CONDITION STRING]/IRD_[CONDITION STRING].parquet

3. Data for plotting number of responses by item block: 
   ./Data/[DATE]/FORM/[CONDITION STRING]/PLOT_DATA_[CONDITION STRING].csv

4. Plot showing number of responses by item block:
    ./Data/[DATE]/FORM/[CONDITION STRING]/PLOT_[CONDITION STRING].png

5. Item parameter data: 
   ./Data/[DATE]/FORM/[CONDITION STRING]/PRM_[CONDITION STRING].dat

Notes:
- This script is provided as part of the open code materials for the above manuscript.
"""

import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

class SIM_2PL_FORM:
    def __init__(
            self, 
            num_persons: int, 
            num_blocks: int,
            num_items_per_block: int,
            num_blocks_per_form: int = 2
        ) -> None:

        self.num_persons = num_persons
        self.num_blocks = num_blocks
        self.num_forms = None
        self.num_items_per_block = num_items_per_block
        self.num_items_per_form = num_blocks_per_form * num_items_per_block
        self.num_blocks_per_form = num_blocks_per_form


        self.a_unit = np.empty(num_items_per_block, dtype=float)
        self.b_unit = np.empty(num_items_per_block, dtype=float)
        self.c_unit = np.empty(num_items_per_block, dtype=float)

    def get_num_unique_items(self):
        return self.num_items_per_block
    
    def get_num_items(self):
        return self.num_blocks * self.num_items_per_block
    
    def get_item_params(self):
        a_full, c_full = self._tile_unit_items()
        return a_full, c_full
    
    def get_min_blocks(self, target_num_forms: int):
        min_blocks =  np.ceil((1 + np.sqrt(1 + 8 * target_num_forms)) / 2)
        return int(min_blocks)

    def _generate_forms(self):
        # Create blocks
        unit_ids = list(range(self.num_blocks))
        unit_map = {
            u: np.arange(u * self.num_items_per_block, (u + 1) * self.num_items_per_block)
            for u in unit_ids
        }

        # Generate unit pairs
        form_unit_pairs = list(itertools.combinations(unit_ids, self.num_blocks_per_form))[:self.num_forms]

        # Create forms from unit pairs
        forms = []
        for u1, u2 in form_unit_pairs:
            items = np.sort(np.concatenate([unit_map[u1], unit_map[u2]]))
            forms.append(items)

        return forms

    def _sample_unit_item_params(self, rnd_seed: int):
        np.random.seed(rnd_seed)
        a_unit = np.random.lognormal(mean=0.3, sigma=0.4, size=self.num_items_per_block)
        b_unit = np.random.normal(loc=0.0, scale=1.0, size=self.num_items_per_block)
        c_unit = - a_unit * b_unit
        return a_unit, b_unit, c_unit
    
    def _tile_unit_items(self):
        a_full = np.tile(self.a_unit, self.num_blocks)
        c_full = np.tile(self.c_unit, self.num_blocks)
        return a_full, c_full

    def _simulate_item_params(self, rnd_seed: int):
        self.a_unit, self.b_unit, self.c_unit = self._sample_unit_item_params(rnd_seed)
        a_full, c_full = self._tile_unit_items()
        return a_full, c_full

    def _sample_theta_standard_normal(self, rnd_seed: int):
        np.random.seed(rnd_seed)
        return np.random.randn(self.num_persons)
    
    def _simulate_item_data(
            self, forms: np.ndarray, 
            person_forms: np.ndarray, 
            theta: np.ndarray, 
            a_full: np.ndarray, 
            c_full: np.ndarray, 
            rnd_seed: int
        ):
        # Create IRD data container
        ird_matrix = np.full((self.num_persons, self.get_num_items()), np.nan, dtype=np.float32) # type: ignore
        
        # Dictionary to track items in each form
        form_item_dict = {}

        # Create IRD data
        np.random.seed(rnd_seed)
        for form_idx, item_ids in tqdm(enumerate(forms), total=len(forms)):
            # Save mapping of form index to item IDs
            form_item_dict[form_idx] = item_ids

            # Simualte item data
            person_idx = np.where(person_forms == form_idx)[0]
            theta_batch = theta[person_idx]
            a_batch = a_full[item_ids]
            c_batch = c_full[item_ids]
            prob = 1 / (1 + np.exp(-( np.outer(theta_batch, a_batch) + c_batch ))) # type: ignore
            y = np.random.binomial(1, prob)
            ird_matrix[np.ix_(person_idx, item_ids)] = y

        return ird_matrix

    def simulate(self, prm_seed: int, theta_seed: int, item_seed: int, missing: int = -9):
        
        np.random.seed(12345)
        
        # Generate forms
        forms = self._generate_forms()
        self.num_forms = len(forms)

        # Randomly assign each person to a form
        person_forms = np.random.choice(self.num_forms, size=self.num_persons)

        # Simulate theta
        theta = self._sample_theta_standard_normal(theta_seed)

        # Simulate full PRM data
        a_full, c_full = self._simulate_item_params(prm_seed)

        # Simulate item data
        ird_matrix = self._simulate_item_data(forms, person_forms, theta, a_full, c_full, item_seed) # type: ignore

        # Fill missing cells to missing code
        ird_matrix = np.nan_to_num(ird_matrix, nan=missing)

        return ird_matrix, person_forms
    
    def _create_folder(self, path: str):
        if not os.path.exists(path):
            print(f'Create folder at {path}')
            os.makedirs(path)
            return True
        else:
            print(f'Already exists: {path}')
            return False

    def _save_prm_data(self, folder_path: str, fname_base: str, a_unit: np.ndarray, b_unit: np.ndarray, c_unit: np.ndarray):
        fname = f"{folder_path}/PRM_{fname_base}.dat"

        if os.path.exists(fname):
            print(f"Skipped: PRM already exists at {fname}")
            return pd.read_csv(fname)

        # Create item parameter table
        prm_data = pd.DataFrame({
            'a': a_unit,
            'b': b_unit,
            'c': c_unit
        }).sort_values(by='a').reset_index(drop=True)

        # Add item_id column
        prm_data.insert(0, 'item_id', np.arange(1, self.num_items_per_block + 1))

        # Save data
        prm_data.to_csv(fname, index=False, sep=",")
        print(f"PRM saved to {fname}")
        return prm_data

    def _save_ird_data(self, folder_path: str, fname_base: str, ird_matrix: np.ndarray, person_forms: np.ndarray):
        fname = f"{folder_path}/IRD_{fname_base}.parquet"

        if os.path.exists(fname):
            print(f"Skipped: IRD already exists at {fname}")
            return pd.read_parquet(fname, engine="pyarrow")

        # Organize data
        ird_data = pd.DataFrame(ird_matrix, columns=[f"item_{j+1}" for j in range(ird_matrix.shape[1])])

        # Add unsorted person_id and form_id
        ird_data.insert(0, "person_id", np.arange(1, self.num_persons + 1)) # 1-based person_id
        ird_data.insert(0, "form_id", person_forms + 1)                     # 1-based form_id

        # Save data
        ird_out = ird_data.sort_values(by=["form_id", "person_id"]).reset_index(drop=True)
        
        ird_out.to_parquet(fname, index=False, compression='snappy', engine="pyarrow")
        print(f"IRD saved to {fname}")
        return ird_out

    def _save_check_data(
            self, folder_path: str, 
            fname_base: str, 
            ird_data: pd.DataFrame, 
            num_samples: int, 
            sample_seed: int = 1010
        ):
        fname = f"{folder_path}/IRD_CHECK_{fname_base}_P{num_samples}.csv"

        if os.path.exists(fname):
            print(f"Skipped: CHECK already exists at {fname}")
            return pd.read_csv(fname)
        
        all_samples = []
        form_ids = ird_data["form_id"].unique()

        for form_id in form_ids:
            form_rows = ird_data[ird_data["form_id"] == form_id]
            sample_rows = form_rows.sample(n=num_samples, random_state=sample_seed) if len(form_rows) >= num_samples else form_rows
            all_samples.append(sample_rows)

        check_out = pd.concat(all_samples, ignore_index=True)
        check_out.to_csv(fname, index=False)
        print(f"CHECK saved to {fname}")
        return check_out
    
    def _create_annot_data(self, form_matrix: pd.DataFrame):
        annot_data = pd.DataFrame("", index=form_matrix.index, columns=form_matrix.columns)
        num_cols = len(form_matrix.columns)

        for i, row in form_matrix.iterrows():
            for start in range(0, num_cols, self.num_items_per_block):
                end = min(start + self.num_items_per_block, num_cols)
                mid = start + (end - start) // 2
                col = form_matrix.columns[mid]
                annot_data.at[i, col] = str(row[col])

        return annot_data
        
    def _plot_form_structure(
            self, 
            folder_path: str, 
            fname_base: str, 
            ird_data: pd.DataFrame, 
            missing: int = -9, 
            save_plot: bool = True, 
            show_plot: bool = True
        ):
        item_cols_new = [col for col in ird_data.columns if col.startswith("item_")]

        # Count non-missing responses by form_id
        form_matrix = (
            ird_data.groupby("form_id")[item_cols_new]
            .apply(lambda x: (x != missing).sum())
            .astype(int)
        )
        form_matrix.columns = [int(col.replace("item_", "")) for col in form_matrix.columns]
        form_matrix = form_matrix.reindex(sorted(form_matrix.columns), axis=1)

        # Plot as heatmap
        base_cmap = plt.get_cmap("Greys")
        shaded_cmap = base_cmap(np.linspace(0.0, 0.8, 256))
        cmap = ListedColormap(shaded_cmap)

        annot_data = self._create_annot_data(form_matrix)
        annot_data[annot_data == "0"] = ""

        plt.figure(figsize=(4, 2))
        ax = sns.heatmap(
            form_matrix, 
            cmap=cmap,
            cbar_kws={'label': 'No. Responses'},
            annot=annot_data,
            annot_kws={"size": 5},
            fmt="",
            cbar=False,
            linewidths=0.0
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
        ax.set_xlabel("X-axis Label", fontsize=6)
        ax.set_ylabel("Y-axis Label", fontsize=6)

        num_items = len(item_cols_new)
        x_tick_step = 10
        # Add 0.5 to enter each tick on cell (need to -1 to start on every 10)
        x_tick_positions = [i + 0.5 for i in range(x_tick_step-1, num_items, x_tick_step)]
        x_tick_labels = [str(i + 1) for i in range(x_tick_step-1, num_items, x_tick_step)]
        plt.xticks(ticks=x_tick_positions, labels=x_tick_labels, rotation=0)

        plt.xlabel("Item ID")
        plt.ylabel("Form ID")
        plt.tight_layout()
        plt.grid(True, linewidth=0.1)

        if save_plot:
            fname = f"{folder_path}/PLOT_{fname_base}.png"
            plt.savefig(fname, bbox_inches='tight', dpi=300)
            print(f"PLOT saved to: {fname}")

            fname = f"{folder_path}/PLOT_DATA_{fname_base}.csv"
            form_matrix.to_csv(fname, index=True)
            print(f"PLOT DATA saved to: {fname}")

        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save(
            self, 
            folder_path: str, 
            ird_matrix: np.ndarray, 
            person_forms: np.ndarray, 
            num_samples: int = 5, 
            sample_seed: int = 1010
        ):
        fname_base = f"P{self.num_persons}_I{self.get_num_items()}_F{self.num_forms}_BF{self.num_blocks_per_form}_IB{self.num_items_per_block}"
        path = f"{folder_path}/{fname_base}"

        # Check if folder exists; if not, create it
        _ = self._create_folder(path)

        # Save
        _ = self._save_prm_data(path, fname_base, self.a_unit, self.b_unit, self.c_unit)
        ird_data = self._save_ird_data(path, fname_base, ird_matrix, person_forms)
        _ = self._save_check_data(path, fname_base, ird_data, num_samples, sample_seed)
        self._plot_form_structure(path, fname_base, ird_data, save_plot=True, show_plot=False)

        print(
            f"IRD shape is {ird_data.shape} (including ID cols).\n"
            f"Each form has {self.num_items_per_form} items."
        )
    
if __name__ == "__main__":
    from datetime import date

    # Set folder path
    today = date.today()
    folder_path = f"./Data/{str(today)}/FORM/"

    # Set hyperparameters
    num_persons = 1_000_000
    num_blocks = 5
    num_items_per_unit = 20
    num_blocks_per_form = 2
    prm_seed = 1010
    theta_seed = 1010
    item_seed = 1010

    # Create simulator
    sim = SIM_2PL_FORM(
        num_persons=num_persons,
        num_blocks=num_blocks,
        num_items_per_block=num_items_per_unit,
        num_blocks_per_form=num_blocks_per_form,
    )

    # Simulate data
    ird_matrix, person_forms = sim.simulate(prm_seed=prm_seed, theta_seed=theta_seed, item_seed=item_seed)

    # Save simulated data
    sim.save(folder_path=folder_path, ird_matrix=ird_matrix, person_forms=person_forms)