"""
This script runs Simulation 4 reported in the manuscript:

"A Modified Expectation-Maximization Algorithm for Accelerated 
Item Response Theory Model Estimation with Large Datasets" 

submitted to *Behavior Research Methods* (Manuscript ID: BR-Org-25-740).

Code written by: 
- Tianying (Teanna) Feng, University of California, Los Angeles

Manuscript authored by:
- Tianying (Teanna) Feng, University of California, Los Angeles
- Li Cai, University of California, Los Angeles

Usage:
On Windows, run from the command line in the project directory using:
    python -m v7_RUN_SIMULATION_4

On Mac/Linux, run from the terminal in the project directory:
    python3 -m v7_RUN_SIMULATION_4

Notes:
- This script is provided as part of the open code materials for the above manuscript.
"""

# %%
from v7_MODIFIED_EM import MODIFIED_EM
from SIM_2PL_TESTLET import SIM_2PL_TESTLET
from datetime import date
import numpy as np
import pandas as pd
import os
import time
import itertools

# %%
M          = [0.10, 0.20, 0.30, 0.50, 1.00]      # Subset proportion
D          = [0.9]                               # Delta threshold
STP        = [0.5]                               # Fixed step size
WM         = [30]                                # SMA warm-up period
CW         = [10]                                # SMA check window  
DT         = [False]                             # Whether use sample size in delta transition
TOL        = 1e-6                                # Convergence tolerance
DEBUG      = False
PRM_SEED   = 2025
ITEM_SEED  = 2025

P  = 60_000                                      # Number of persons
T  = 3                                           # Number of testlets
IT = [10]                                        # Number of items per testlet
TV = [0.05, 0.10, 0.20, 0.50, 0.80, 1.00]        # Testlet variance

# %%
conditions = [
    (P, T, it, tv)
    for it, tv in itertools.product(IT, TV)
]

# Sort conditions
conditions.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

# All subset configurations
subset_conditions = list(itertools.product(M, D, STP, WM, CW, DT))

# Sort subset conditions by subset proportion (M)
subset_conditions.sort(key=lambda x: x[0])

# %%
def run_simulation(start_iter, num_persons, num_testlets, num_items_per_testlet, testlet_variance, num_reps, save_folder):
    print(
        f"\nRun condition: P{str(num_persons)}",
        f" | T{str(num_testlets)}",
        f" | IT{str(num_items_per_testlet)}"
        f" | TV{str(testlet_variance)}",
    )

    # Create date stamped folder to store output
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    simulator = SIM_2PL_TESTLET(
        num_persons           = num_persons,
        num_testlets          = num_testlets,
        num_items_per_testlet = num_items_per_testlet,
        C                     = 1.0, # C = 1.0 --> equal-slope constraint
    )
    num_items = num_testlets * num_items_per_testlet

    # Specify condition folder name
    tvname  = 'TV'  + str(testlet_variance)
    pname   = 'P'   + str(num_persons)
    tname   = 'T'   + str(num_testlets)
    itname  = 'IT'  + str(num_items_per_testlet)
    sim_save_folder  = f"{save_folder}/Simulation_Data/{tvname}_{pname}_{tname}_{itname}"

    for subset_perc, delta_thr, step, wait, window_chk, delta_useN in subset_conditions:

        subset_size = int(np.floor(num_persons * subset_perc))

        # Specify sub-condition file name
        mname   = 'M'   + str(int(subset_perc * 100))
        dname   = 'D'   + str(delta_thr)
        stpname = 'STP' + str(step)
        wmname  = 'WM'  + str(wait)
        cwname  = 'CW'  + str(window_chk)
        dtname  = 'DT'  + str(int(delta_useN))
        out_fname = f"{tvname}_{pname}_{tname}_{itname}_{mname}_{dname}_{stpname}_{wmname}_{cwname}_{dtname}.xlsx"
        out_path  = os.path.join(save_folder, out_fname)

        # Track whether standard EM (100%) has been run; store first-run results
        ran_standard_em = False
        standard_em_df_path = None

        for r in range(start_iter, num_reps + 1):

            if subset_perc == 1.0 and ran_standard_em:
                continue
            
            time.sleep(0.5)
            print(f"\nReplication: {r}")

            # Simulate data
            ird, a_true, c_true = simulator.simulate(
                prm_seed   = PRM_SEED,
                item_seed  = ITEM_SEED,
                theta_seed = r,
                mean_ds    = [0.0] * num_testlets,
                var_ds     = [testlet_variance] * num_testlets,
                a_low      = 0.5,
                a_high     = 2.5,
                c_mean     = 0.0,
                c_sd       = 1.0,
                mean_g     = 0.0,
                var_g      = 1.0,
            )
            freq = np.ones(ird.shape[0])
            simulator.save(folder_path=sim_save_folder, file_suffix=f"_TS{r}", as_parquet=False)

            # Run EM
            em_run = MODIFIED_EM()
            if subset_perc == 1.0: # Run full-data standard EM
                if not ran_standard_em:
                    print("\nStart: 100% subset size (Standard EM)")
                    result = em_run.run(
                        ird, 
                        freq, 
                        s1_iter     = 500, 
                        s2_iter     = 0, 
                        subset_size = None,
                        tol         = TOL,
                        get_se      = True,
                        verbose     = True,
                        debug_mode  = DEBUG,
                    )
            else: # Run subset-based modified EM
                print(
                    f"\nStart: {int(subset_perc * 100)} % subset | {delta_thr} delta threshold",
                    f" | {int(delta_useN)} delta transition",
                    f" | {wait} warm-up | {window_chk} check window"
                )
                
                result = em_run.run(
                    ird, 
                    freq, 
                    s1_iter     = 500, 
                    s2_iter     = 500, 
                    subset_size = subset_size, 
                    delta_thr   = delta_thr, 
                    delta_useN  = delta_useN,
                    step        = step,
                    wait        = wait,
                    window_chk  = window_chk,
                    tol         = TOL,
                    get_se      = True,
                    verbose     = True,
                    debug_mode  = DEBUG,
                ) 

            out = {
                    'converged'            : False,
                    'subset_perc'          : subset_perc,
                    'delta_thr'            : delta_thr,
                    'delta_useN'           : int(delta_useN),
                    'wait'                 : wait,
                    'window_chk'           : window_chk,
                    'step'                 : step,
                    'num_items'            : num_items,
                    'num_persons'          : num_persons,
                    'num_testlets'         : num_testlets,
                    'num_items_per_testlet': num_items_per_testlet,
                    'testlet_variance'     : testlet_variance,
                    'replication'          : str(r),
                    'est_time'             : None,
                    's1_time'              : None,
                    's2_time'              : None,
                    's1_updates'           : None,
                    's2_updates'           : None,
                }
    
            # Add results about estimation time
            if result['s2_time'] > 0.0:
                out['s2_time']      = result['s2_time']
                out['est_time']     = result['s1_time'] + result['s2_time']
                out['s2_updates']   = result['s2_end'] + 1
            else:
                out['est_time']     = result['s1_time']

            out['s1_time']          = result['s1_time']
            out['s1_updates']       = result['s1_end'] + 1
            out['converged']        = int(result['converged'])

            # Format output data 
            out = pd.DataFrame([out])

            # Get true parameter values
            a_tru = pd.DataFrame(a_true.reshape(-1, num_items), columns = ['a' + str(i+1) + '_true' for i in range(num_items)])
            c_tru = pd.DataFrame(c_true.reshape(-1, num_items), columns = ['c' + str(i+1) + '_true' for i in range(num_items)])

            # 1. If running subset-based modified EM: the last averaged iterates are final estimates (x_smoothed != x_history)
            # 2. If running full-data standard EM: the last raw iterates are final estimates (x_smoothed = x_history)
            a_est  = pd.DataFrame( result['a'].reshape(-1, num_items),           columns = ['a' + str(i+1) + '_est'  for i in range(num_items)])
            c_est  = pd.DataFrame( result['c'].reshape(-1, num_items),           columns = ['c' + str(i+1) + '_est'  for i in range(num_items)])
            a_bias = pd.DataFrame((result['a'] - a_true).reshape(-1, num_items), columns = ['a' + str(i+1) + '_bias' for i in range(num_items)])
            c_bias = pd.DataFrame((result['c'] - c_true).reshape(-1, num_items), columns = ['c' + str(i+1) + '_bias' for i in range(num_items)])

            # Get estimated SEs (_se) and true SEs (_tse) --> not evaluated in Study 4
            a_se   = pd.DataFrame(result['a_se'].reshape(-1, num_items),         columns = ['a' + str(i+1) + '_se'   for i in range(num_items)])
            c_se   = pd.DataFrame(result['c_se'].reshape(-1, num_items),         columns = ['c' + str(i+1) + '_se'   for i in range(num_items)])
            
            out = pd.concat([out, a_bias, c_bias, a_est, c_est, a_tru, c_tru, a_se, c_se], axis=1)
        
            # Append data to an Excel file
            if os.path.exists(out_path):
                with pd.ExcelWriter(out_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    out.to_excel(writer, sheet_name="Results", index=False, header=False, startrow=writer.sheets["Results"].max_row)
            else:
                with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                    out.to_excel(writer, sheet_name="Results", index=False, header=True)
            print(f"Saved: {out_path}")

        ### END OF REPLICATION LOOP ###
    
        # If any sub-condition runs standard EM, copy the results file
        # and rename it using the current sub-condition's file name.
        # The content stays the same, because:
        #   1. Sub-conditions do not affect standard EM results.
        #   2. However, for compatibility with later evaluation and plotting,
        #      it's helpful to have a version of the standard EM file
        #      saved under each sub-condition's name.
        if subset_perc == 1.0:
            if standard_em_df_path is None:
                standard_em_df_path = out_path
                ran_standard_em = True
            else:
                # For full-EM runs, update the delta_thr for later plotting;
                # the actual delta_thr has no impact on results;
                # fluctuation in est_time (wall-clock time) is expected.
                df_s1 = pd.read_excel(standard_em_df_path, sheet_name='Results')
                df_s1['delta_thr'] = delta_thr
                df_s2 = pd.read_excel(standard_em_df_path, sheet_name='Summary_Stats')
                delta_thr_col_idx = df_s2.columns.tolist().index('delta_thr')
                df_s2.iat[0, delta_thr_col_idx] = delta_thr

                with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                    df_s1.to_excel(writer, sheet_name='Results', index=False)
                    df_s2.to_excel(writer, sheet_name='Summary_Stats', index=False)
                continue # Skip the rest

        # When all replications are done, save the summary statistics
        # Load recplication results sheet
        df = pd.read_excel(out_path, sheet_name = "Results")  

        # Get columns that contain true values
        true_columns = [col for col in df.columns if '_true' in col]

        # Compute bias and RMSE for item parameters and for SEs
        est_bias_values = {}
        est_rmse_values = {}
        param_names = []

        for true_col in true_columns:
            est_col = true_col.replace('_true', '_est')
            
            param_name = true_col.replace('_true', '')
            param_names.append(param_name)

            if est_col in df.columns:
                est_bias = np.mean(df[est_col] - df[true_col])
                est_rmse = np.sqrt(np.mean((df[est_col] - df[true_col]) ** 2)) 
                est_bias_values[param_name] = est_bias
                est_rmse_values[param_name] = est_rmse
            else:
                est_bias_values[param_name] = np.nan
                est_rmse_values[param_name] = np.nan
                
        est_summ = pd.DataFrame({
            'parameter': param_names,
            'true_val' : [np.mean(df[f"{param}_true"]) for param in param_names],
            'est_mean' : [np.mean(df[f"{param}_est"]) for param in param_names],
            'est_bias' : [est_bias_values[param] for param in param_names],
            'est_rmse' : [est_rmse_values[param] for param in param_names],
        })

        numeric_cols = ['est_mean', 'true_val', 'est_bias', 'est_rmse']
        est_summ[numeric_cols] = est_summ[numeric_cols].astype(float).round(6)

        est_a = est_summ[est_summ['parameter'].str.startswith('a')].reset_index(drop=True)
        est_c = est_summ[est_summ['parameter'].str.startswith('c')].reset_index(drop=True)

        # Format a's and c's est_summ side by side
        est_summary = pd.concat([est_a, est_c], axis=1)

        # Compute additional summary stats
        summary_stats = {
            'prop_converged'       : df['converged'].mean(),
            'subset_perc'          : df['subset_perc'].unique()[0],
            'delta_thr'            : df['delta_thr'].unique()[0],
            'delta_useN'           : df['delta_useN'].unique()[0],
            'wait'                 : df['wait'].unique()[0],
            'window_chk'           : df['window_chk'].unique()[0],
            'step'                 : df['step'].unique()[0],
            'num_items'            : df['num_items'].unique()[0],
            'num_persons'          : df['num_persons'].unique()[0],
            'num_testlets'         : df['num_testlets'].unique()[0],
            'num_items_per_testlet': df['num_items_per_testlet'].unique()[0],
            'testlet_variance'     : df['testlet_variance'].unique()[0],
            'num_reps'             : num_reps,
            'mean_est_time'        : df['est_time'].mean(),
            'mean_s1_time'         : df['s1_time'].mean(),
            'mean_s2_time'         : df['s2_time'].mean(),
            'mean_s1_updates'      : df['s1_updates'].mean(),
            'mean_s2_updates'      : df['s2_updates'].mean()
        }

        summary_df = pd.DataFrame([summary_stats])
        mean_cols = [col for col in summary_df.columns if col.startswith('mean_')]
        summary_df[mean_cols] = summary_df[mean_cols].apply(lambda x: x.round(6))

        # Save summary results
        with pd.ExcelWriter(out_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'overlay') as writer:
            startrow = 0
            summary_df.to_excel(writer, sheet_name = "Summary_Stats", index = False, startrow = startrow, na_rep = np.nan) # type: ignore

            startrow = summary_df.shape[0] + 3
            est_summary.to_excel(writer, sheet_name = "Summary_Stats", index = False, startrow = startrow, na_rep = np.nan) # type: ignore

        print("Added summary sheet.")

        ### END OF SUB-CONDITION LOOP ###

# %%
if __name__ == "__main__":
    today = date.today()
    save_folder = f"./Simulation_4/{str(today)}/"

    num_reps = 100

    for num_persons, num_testlets, num_items_per_testlet, testlet_variance in conditions:
        run_simulation(
            start_iter            = 1, 
            num_persons           = num_persons, 
            num_testlets          = num_testlets, 
            num_items_per_testlet = num_items_per_testlet, 
            testlet_variance      = testlet_variance, 
            num_reps              = num_reps, 
            save_folder           = save_folder
        )
    
    print("All finished.")

# %%
