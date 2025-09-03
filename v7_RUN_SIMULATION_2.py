"""
This script runs Simulation 2 reported in the manuscript:

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
    python -m v7_RUN_SIMULATION_2

On Mac/Linux, run from the terminal in the project directory:
    python3 -m v7_RUN_SIMULATION_2

Notes:
- This script is provided as part of the open code materials for the above manuscript.
"""

# %%
from v7_MODIFIED_EM import MODIFIED_EM
from SIM_2PL import SIM_2PL
from datetime import date
import numpy as np
import pandas as pd
import os
import time
import itertools
import matplotlib.pyplot as plt

# %%
M     = [0.10, 0.20, 0.30, 0.50, 1.00]   # Subset proportion
D     = [0.90]                           # Delta threshold
STP   = [0.5]                            # Fixed step size
WM    = [30]                             # SMA warm-up period
CW    = [10]                             # SMA check window  
DT    = [False]                          # Whether use sample size in delta transition
P     = [60000]                          # Number of persons
I     = [12]                             # Number of items
TOL   = 1e-6                             # Convergence tolerance
DEBUG = True                             # Has to be true for this script to work

# Main conditions
conditions = list(itertools.product(P, I))

# Sort conditions by number of persons then items
conditions.sort(key=lambda x: (x[0], x[1]))

# All subset configurations
subset_conditions = list(itertools.product(M, D, STP, WM, CW, DT))

# Sort subset conditions by subset proportion (M)
subset_conditions.sort(key=lambda x: x[0])

# %%
# List of true item parameters
PRM = {
    'a': [0.68, 1.27, 1.56, 1.22, 1.38, 1.80, 1.33, 2.02, 1.16, 1.11, 0.94, 0.80],
    'c': [1.60, 2.24, 2.39, 1.57, 1.14, 0.21,-0.16,-1.19,-1.23,-1.70,-1.66,-1.83],
}

# List of gold-standard SEs based on expected FIM
TSE = {
    'a': [0.01525444, 0.02181192, 0.02558531, 0.01900873, 0.01972090, 0.02447509, 0.01818466, 0.02939037, 0.01769757, 0.01851394, 0.01705931, 0.01682350],
    'c': [0.01347961, 0.02061150, 0.02400600, 0.01581468, 0.01456956, 0.01397306, 0.01202998, 0.01837651, 0.01391744, 0.01599362, 0.01492119, 0.01519546],
}

# %%
# Helper functions
def collect_result(a_history, c_history, s1_end):
    # note that s1_end is zero-indexed
    num_items    = c_history.shape[1]
    a_df         = pd.DataFrame(a_history, columns=['a' + str(i+1) for i in range(num_items)])
    c_df         = pd.DataFrame(c_history, columns=['c' + str(i+1) for i in range(num_items)])
    df           = pd.concat([a_df.reset_index(drop=True), c_df], axis=1)
    df['update'] = df.index + 1
    df['mode']   = np.where(df.index <= (s1_end + 1), 'full', 'subset')
    # Mark the last iterate under 'mode'
    df.at[df.index[-1], 'mode'] = 'last'
    df           = df[['update', 'mode'] + list(a_df.columns) + list(c_df.columns)]
    return df

def plot_lines(ax, data, mode, color_normal, color_full, linestyle_full, label):
    """
    Plot segments with differing line colors and styles.
    """
    plotted = False

    for i in range(len(data) - 1):
        label = label if not plotted else None
        if mode[i] == "full":
            ax.plot([i, i + 1], [data[i], data[i + 1]], color=color_full, linestyle=linestyle_full, lw=0.8, label=None)
        else:
            ax.plot([i, i + 1], [data[i], data[i + 1]], color=color_normal, linestyle='solid', lw=1.0, label=label)
            plotted = True

def plot_traces(a_history, c_history, a_smoothed = None, c_smoothed = None, mode = None, title = ""):
    """
    Plot the estimates against the Kalman-filtered (smoothed) values.
    Returns the plot object to allow saving.
    """
    fig, ax1 = plt.subplots(figsize=(6, 3))

    # Convert mode to a NumPy array for easier processing
    mode = np.array(mode) if mode is not None else np.array(["normal"] * len(a_history))
    
    # Plot 'A' estimates and smoothed line
    ax1.scatter(range(len(a_history)), a_history, label="A estimates", marker='x', s=15.0, lw=1.0, color='blue', alpha=0.3)
    ax1.set_ylabel("A")

    # Plot 'C' estimates and smoothed line
    ax2 = ax1.twinx()
    ax2.scatter(range(len(c_history)), c_history, label="C estimates", marker='.', s=18.0, lw=1.0, color='orange', alpha=0.3)
    ax2.set_ylabel("C")

    if a_smoothed is not None:
        # Plot smoothed 'A' with mode-dependent coloring
        plot_lines(ax1, a_smoothed, mode, "darkblue", "blue", "dashed", "A smoothed")

        # Plot smoothed 'C' with mode-dependent coloring
        plot_lines(ax2, c_smoothed, mode, "darkorange", "orange", "dashed", "C smoothed")

    fig.suptitle(title)
    fig.tight_layout()
    
    # Combine legends from both axes and place it at the bottom
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False)
    
    return fig

def save_plots(a_history, a_smoothed, c_history, c_smoothed, c_true, df_hist, save_dir="./Plots/"):
    """
    Generate and save plots for each item.
    """

    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(c_true)):
        save_path = os.path.join(save_dir, f"ITEM_{i+1}.png")
        
        # Create the plot
        fig = plot_traces(
            a_history   = a_history[:, i], 
            c_history   = c_history[:, i], 
            a_smoothed  = a_smoothed[:, i] if len(a_smoothed) > 0 else None, 
            c_smoothed  = c_smoothed[:, i] if len(c_smoothed) > 0 else None, 
            mode        = df_hist['mode'],
            title       = f"Item {i+1}"
        )
        
        # Save the plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Plots saved in '{save_dir}' directory.")

# %%
def run_simulation(start_iter, num_items, num_persons, num_reps, save_folder):
    print(f"\nRun condition: I{str(num_items)} | P{str(num_persons)}")

    # Create date stamped folder to store output
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    sim_save_folder  = save_folder + '/Simulation_Data/' + 'I'  + str(num_items) + '/'
    hist_save_folder = save_folder + '/History_Data/'    + 'I'  + str(num_items) + '/'

    # Create date stamped folder to store est. history data
    if not os.path.exists(hist_save_folder):
        os.makedirs(hist_save_folder)

    # Track whether standard EM (100%) has been run; store first-run results
    ran_standard_em = False
    standard_em_df_path = None

    for subset_perc, delta_thr, step, wait, window_chk, delta_useN in subset_conditions:

        subset_size = int(np.floor(num_persons * subset_perc))

        # Specify by-condition file name
        iname   = 'I'   + '{:03d}'.format(num_items)
        pname   = 'P'   + str(num_persons)
        mname   = 'M'   + str(int(subset_perc * 100))
        dname   = 'D'   + str(delta_thr)
        stpname = 'STP' + str(step)
        wmname  = 'WM'  + str(wait)
        cwname  = 'CW'  + str(window_chk)
        dtname  = 'DT'  + str(int(delta_useN))

        out_fname = f"{mname}_{iname}_{pname}_{dname}_{stpname}_{wmname}_{cwname}_{dtname}.xlsx"
        out_path  = os.path.join(save_folder, out_fname)

        for r in range(start_iter, num_reps + 1):

            if subset_perc == 1.0 and ran_standard_em:
                continue
            
            time.sleep(0.5)
            print(f"\nReplication: {r}")

            # The same true item parameters are used across replications
            # The theta values vary across replications
            sim_2pl = SIM_2PL(num_items, num_persons)
            sim_2pl.simulate_with_prm(prm=PRM, theta_seed=r, item_seed=r)
            pattern = sim_2pl.pattern
            freq    = sim_2pl.freq
            a_true  = sim_2pl.a
            c_true  = sim_2pl.c
            
            # Save simulated data and prm
            sname_prm    = 'PS' + str(sim_2pl.prm_seed)
            sname_theta  = 'TS' + str(sim_2pl.theta_seed)
            sim_2pl.save(sim_save_folder, fixed_prm=True) # type: ignore
            
            em_run = MODIFIED_EM()

            out = {
                'converged'    : False,
                'subset_perc'  : subset_perc,
                'delta_thr'    : delta_thr,
                'delta_useN'  : int(delta_useN),
                'wait'         : wait,
                'window_chk'   : window_chk,
                'num_items'    : num_items,
                'num_persons'  : num_persons,
                'step'         : step,
                'replication'  : str(r),
                'est_time'     : None,
                's1_time'      : None,
                's2_time'      : None,
                's1_updates'   : None,
                's2_updates'   : None,
            }

            if subset_perc == 1.0: # Run full-data standard EM
                if not ran_standard_em:
                    print("\nStart: 100% subset size (Standard EM)")
                    result = em_run.run(
                        pattern, 
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
                    f"\nStart: {int(subset_perc * 100)} % subset | {delta_thr} delta threshold | {int(delta_useN)} delta transition",
                    f"| {wait} warm-up | {window_chk} check window"
                )
                
                result = em_run.run(
                    pattern, 
                    freq, 
                    s1_iter     = 100, 
                    s2_iter     = 100, 
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
            a_tru = pd.DataFrame(a_true.reshape(-1, num_items), columns = ['a' + str(i+1) + '_true' for i in range(num_items)]) # type: ignore
            c_tru = pd.DataFrame(c_true.reshape(-1, num_items), columns = ['c' + str(i+1) + '_true' for i in range(num_items)]) # type: ignore

            # 1. If running subset-based modified EM: the last averaged iterates are final estimates (x_smoothed != x_history)
            # 2. If running full-data standard EM: the last raw iterates are final estimates (x_smoothed = x_history)
            a_est  = pd.DataFrame(result['a_smoothed'][-1].reshape(-1, num_items),            columns = ['a' + str(i+1) + '_est'  for i in range(num_items)])
            c_est  = pd.DataFrame(result['c_smoothed'][-1].reshape(-1, num_items),            columns = ['c' + str(i+1) + '_est'  for i in range(num_items)])
            a_bias = pd.DataFrame((result['a_smoothed'][-1] - a_true).reshape(-1, num_items), columns = ['a' + str(i+1) + '_bias' for i in range(num_items)])
            c_bias = pd.DataFrame((result['c_smoothed'][-1] - c_true).reshape(-1, num_items), columns = ['c' + str(i+1) + '_bias' for i in range(num_items)])
            a_raw  = pd.DataFrame(result['a_history'][-1].reshape(-1, num_items),             columns = ['a' + str(i+1) + '_raw'  for i in range(num_items)])
            c_raw  = pd.DataFrame(result['c_history'][-1].reshape(-1, num_items),             columns = ['c' + str(i+1) + '_raw'  for i in range(num_items)])

            # Get estimated SEs (_se) and true SEs (_tse)
            a_se   = pd.DataFrame(result['a_se'].reshape(-1, num_items),                     columns = ['a' + str(i+1) + '_se'   for i in range(num_items)])
            c_se   = pd.DataFrame(result['c_se'].reshape(-1, num_items),                     columns = ['c' + str(i+1) + '_se'   for i in range(num_items)])
            a_tse  = pd.DataFrame(np.array(TSE['a']).reshape(-1, num_items),                 columns = ['a' + str(i+1) + '_tse'  for i in range(num_items)])
            c_tse  = pd.DataFrame(np.array(TSE['c']).reshape(-1, num_items),                 columns = ['c' + str(i+1) + '_tse'  for i in range(num_items)])

            out    = pd.concat([out, a_bias, c_bias, a_est, c_est, a_se, c_se, a_tru, c_tru, a_tse, c_tse, a_raw, c_raw], axis=1)
        
            # Append data to an Excel file
            if os.path.exists(out_path):
                with pd.ExcelWriter(out_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    out.to_excel(writer, sheet_name="Results", index=False, header=False, startrow=writer.sheets["Results"].max_row)
            else:
                with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                    out.to_excel(writer, sheet_name="Results", index=False, header=True)
            print(f"Saved: {out_path}")

            # Save history data for the 1st replication and every 50 replications
            if r == 1 or r % 50 == 0:
                est_hist_fname = f"{mname}_{iname}_{pname}_{dname}_{stpname}_{sname_prm}_{sname_theta}"
                est_hist_path  = hist_save_folder + est_hist_fname
                est_hist       = collect_result(result['a_history'], result['c_history'], result['s1_end'])
                est_hist.insert(0, 'converged', int(result['converged']))
                est_hist.to_csv(est_hist_path + ".csv", index = False)
                print(f"Saved: {est_hist_fname}.csv")
                save_plots(result['a_history'], result['a_smoothed'], result['c_history'], result['c_smoothed'], c_true, est_hist, save_dir=est_hist_path)

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

        # Get columns that contain SE values
        se_columns = [col for col in df.columns if '_se' in col]

        # Compute ESD, mean SE, bias, and RMSE
        ref_values = {}
        mean_se_values = {}
        bias_values = {}
        rmse_values = {}
        param_names = []

        for se_col in se_columns:
            est_col = se_col.replace('_se', '_est')
            param_name = se_col.replace('_se', '')
            tse_col = se_col.replace('_se', '_tse')
            param_names.append(param_name)

            if est_col in df.columns:
                ref = df[tse_col].unique()
                mean_se = df[se_col].mean()
                bias = mean_se - ref
                rmse = np.sqrt(np.mean((df[se_col] - ref) ** 2)) 
                
                ref_values[param_name] = ref
                mean_se_values[param_name] = mean_se
                bias_values[param_name] = bias
                rmse_values[param_name] = rmse
            else:
                ref_values[param_name] = np.nan
                mean_se_values[param_name] = np.nan
                bias_values[param_name] = np.nan
                rmse_values[param_name] = np.nan
                
        se_summary = pd.DataFrame({
            'parameter': param_names,
            'ref'      : [ref_values[param] for param in param_names],
            'mean_se'  : [mean_se_values[param] for param in param_names],
            'bias'     : [bias_values[param] for param in param_names],
            'rmse'     : [rmse_values[param] for param in param_names]
        })

        numeric_cols = ['ref', 'mean_se', 'bias', 'rmse']
        se_summary[numeric_cols] = se_summary[numeric_cols].astype(float).round(6)

        # Compute additional summary stats
        summary_stats = {
            'prop_converged' : df['converged'].mean(),
            'subset_perc'    : df['subset_perc'].unique()[0],
            'delta_thr'      : df['delta_thr'].unique()[0],
            'delta_useN'     : df['delta_useN'].unique()[0],
            'wait'           : df['wait'].unique()[0],
            'window_chk'     : df['window_chk'].unique()[0],
            'num_items'      : df['num_items'].unique()[0],
            'num_persons'    : df['num_persons'].unique()[0],
            'step'           : df['step'].unique()[0],
            'num_reps'       : num_reps,
            'mean_est_time'  : df['est_time'].mean(),
            'mean_s1_time'   : df['s1_time'].mean(),
            'mean_s2_time'   : df['s2_time'].mean(),
            'mean_s1_updates': df['s1_updates'].mean(),
            'mean_s2_updates': df['s2_updates'].mean()
        }

        summary_df = pd.DataFrame([summary_stats])
        mean_cols = [col for col in summary_df.columns if col.startswith('mean_')]
        summary_df[mean_cols] = summary_df[mean_cols].apply(lambda x: x.round(6))

        # Save summary results
        with pd.ExcelWriter(out_path, engine = 'openpyxl', mode = 'a', if_sheet_exists = 'overlay') as writer:
            startrow = 0
            summary_df.to_excel(writer, sheet_name = "Summary_Stats", index = False, startrow = startrow, na_rep = np.nan) # type: ignore

            startrow = summary_df.shape[0] + 3
            se_summary.to_excel(writer, sheet_name = "Summary_Stats", index = False, startrow = startrow, na_rep = np.nan) # type: ignore

        print("Added summary sheet.")

        ### END OF SUB-CONDITION LOOP ###


# %%
if __name__ == "__main__":
    today = date.today()
    save_folder = f"./Simulation_2/{str(today)}/"

    num_reps = 1000

    for num_persons, num_items in conditions:
        run_simulation(
            start_iter=1,
            num_items=num_items,
            num_persons=num_persons,
            num_reps=num_reps,
            save_folder=save_folder
        )
    
    print("All finished.")

# %%