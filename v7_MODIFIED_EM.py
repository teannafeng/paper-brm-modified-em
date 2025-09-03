"""
This script contains the Python implementation for the modified 
expectation-maximization (EM) algorithm proposed in the manuscript:

"A Modified Expectation-Maximization Algorithm for Accelerated 
Item Response Theory Model Estimation with Large Datasets"

submitted to *Behavior Research Methods* (Manuscript ID: XXXX).

The script defines the core functions that are imported and executed by the 
simulation scripts:
- v7_RUN_SIMULATION_1.py
- v7_RUN_SIMULATION_2.py
- v7_RUN_SIMULATION_3.py

Code written by: 
- Tianying (Teanna) Feng, University of California, Los Angeles

Manuscript authored by:
- Tianying (Teanna) Feng, University of California, Los Angeles
- Li Cai, University of California, Los Angeles

Usage:
This script is not intended to be run directly, as its main block contains only 'pass'. 
To reproduce results, use the simulation runner scripts, for example:

On Windows:
    python -m v7_RUN_SIMULATION_1

On Mac/Linux:
    python3 -m v7_RUN_SIMULATION_1

Notes:
- This script is provided as part of the open code materials for the above manuscript. 
"""

# %%
import time
from numba import njit
import numpy as np
from scipy.optimize import minimize
from scipy.special  import expit
from SMA import SMA

# %%
########################################################################################################
### BELOW ARE NUMBA-MODIFIED MAJOR FUNCTIONS THAT SPEED UP E-STEP AND XPD COMPUTATION OF MODIFIED EM ###
########################################################################################################

@njit
def compute_r01_numba(pattern, posteriors, r, missing=-9):
    """
    Numba-modified function to compute r1 and r0 (E-step).
    """
    S, I = pattern.shape
    K = posteriors.shape[1]

    r0 = np.zeros((I, K))
    r1 = np.zeros((I, K))

    for i in range(I):
        for s in range(S):
            u = pattern[s, i]
            if u == missing:
                continue
            for k in range(K):
                weight = r[s] * posteriors[s, k]
                r1[i, k] += u * weight
                r0[i, k] += (1 - u) * weight
    
    return r1, r0

@njit
def compute_posteriors_numba(P, pattern, weights, missing=-9, eps=1e-10):
    """
    Numba-modified function to compute posteriors (E-step).
    """
    S, I = pattern.shape
    K = P.shape[1]

    log_P = np.log(P + eps)
    log_1mP = np.log(1 - P + eps)
    log_weights = np.log(weights + eps)

    log_likelihoods = np.zeros((S, K))

    for i in range(I):
        for s in range(S):
            u = pattern[s, i]
            if u == missing:
                continue
            for k in range(K):
                if u == 1:
                    log_likelihoods[s, k] += log_P[i, k]
                else:
                    log_likelihoods[s, k] += log_1mP[i, k]

    posteriors = np.empty((S, K))
    marginals = np.empty(S)

    for s in range(S):
        for k in range(K):
            posteriors[s, k] = np.exp(log_likelihoods[s, k] + log_weights[k])
        marginals[s] = np.sum(posteriors[s])
        if marginals[s] == 0.0:
            raise ValueError("Marginal likelihood is zero.")
        for k in range(K):
            posteriors[s, k] /= marginals[s]

    return posteriors, marginals

@njit
def compute_P_numba(a, c, q_nodes, eps=1e-10):
    """
    Numba-modified function to compute response probability 
    matrix for all items.
    """
    I = a.shape[0]
    K = q_nodes.shape[0]
    P = np.empty((I, K))

    for i in range(I):
        for k in range(K):
            z = a[i] * q_nodes[k] + c[i]
            _P = 1 / (1 + np.exp(-z))
            P[i, k] = min(max(_P, eps), 1 - eps)
    
    return P

@njit
def compute_v_row_numba(pattern_row, posterior_row, P, q_nodes):
    """
    Numba-modified function to compute gradients of marginal LL.
    """
    I = pattern_row.shape[0]
    K = posterior_row.shape[0]

    v_row = np.zeros(2 * I)

    for i in range(I):
        v_a = 0.0
        v_c = 0.0
        for k in range(K):
            w_ik = (pattern_row[i] - P[i, k]) * posterior_row[k]
            v_a += w_ik * q_nodes[k]
            v_c += w_ik
        v_row[2 * i] = v_a
        v_row[2 * i + 1] = v_c

    return v_row

@njit
def compute_XPD_numba(pattern, r, posteriors, P, q_nodes):
    """
    Numba-modified function to compute XPD information matrix.
    """
    S, I = pattern.shape
    fim = np.zeros((2 * I, 2 * I))

    for s in range(S):
        v_row = compute_v_row_numba(pattern_row=pattern[s], posterior_row=posteriors[s], P=P, q_nodes=q_nodes)
        for i in range(2 * I):
            for j in range(2 * I):
                fim[i, j] += r[s] * v_row[i] * v_row[j]
    
    return fim

# %%
class MODIFIED_EM:
    def __init__(self):
        self.P = None 
        self.posteriors = None
        self.fim = None
        self.cov = None
        self.w = None
    
    ########################################################
    ### BELOW IS THE WRAPPER FUNCTION TO RUN MODIFIED EM ###
    ########################################################

    def run(self, 
            patterns, 
            frequencies, 
            q_nodes        = np.arange(-6, 6.1, 0.1), 
            s1_iter        = 100, 
            s2_iter        = 100,
            subset_size    = None,
            delta_thr      = 0.90,
            delta_useN     = False,
            step           = 0.5,
            wait           = 30,
            window_chk     = 10,
            missing        = -9,
            tol            = 1e-6,
            get_se         = True,
            verbose        = True,
            debug_mode     = False):
        """
        Wrapper function to run algorithm(s).
        """
        
        self._setup(
            patterns=patterns,
            frequencies=frequencies,
            q_nodes=q_nodes,
            s1_iter=s1_iter,
            s2_iter=s2_iter,
            subset_size=subset_size,
            delta_thr=delta_thr,
            delta_useN=delta_useN,
            step=step,
            wait=wait,
            window_chk=window_chk,
            missing=missing,
            tol=tol,
            get_se=get_se,
            verbose=verbose,
            debug_mode=debug_mode,
        )
        
        if self.s1_iter > 0:
            self._run_stage1()

        if self.s2_iter > 0 and not self.converged:
            self._run_stage2()

        return self._finalize()

    #############################################################
    ### BELOW ARE MAJOR FUNCTIONS THAT CONSTITUTE MODIFIED EM ###
    #############################################################

    def compute_COV(self, fim): 
        """
        Function to compute covariance matrix from a information matrix.
        """
        return np.linalg.pinv(fim)
    
    def compute_SEs(self, cov):
        """
        Function to compute SEs from a given covariance matrix.
        """
        SEs = np.sqrt(np.diag(cov))
        SE_a = SEs[::2]
        SE_c = SEs[1::2]
        return SE_a, SE_c

    def run_estep(self, a, c, pattern, r, q_nodes, missing=-9):
        """
        Function to perform E-step of EM algorithm.
        """
        S = len(r)
        K = len(q_nodes)

        if self.posteriors is None:
            self.posteriors = np.zeros((S, K))

        if self.w is None:
            self.w = self._compute_weights(q_nodes)

        self.P = compute_P_numba(a, c, q_nodes)
        posteriors, marginals = compute_posteriors_numba(self.P, pattern, self.w, missing)
        self.posteriors = posteriors
        LL = np.sum(r * np.log(marginals))

        r1, r0 = compute_r01_numba(pattern, posteriors, r, missing)

        return r0, r1, LL
    
    def run_mstep_full(self, a, c, r0, r1, q_nodes, method='BFGS'):
        """
        Function to perform M-step of standard EM.
        """
        num_items = len(a)
        new_a = np.copy(a)
        new_c = np.copy(c)

        for i in range(num_items):
            result = minimize(
                fun     = lambda p: self._compute_nll(p, i, r0, r1, q_nodes), 
                x0      = [a[i], c[i]],
                method  = method,
                options = {'disp': False}
            )
            new_a[i], new_c[i] = result.x

        return new_a, new_c
    
    def run_mstep_subset(self, a, c, r0, r1, q_nodes, step):
        """
        Function to perform M-step of modified EM.
        """
        num_items = len(a)
        new_a = np.copy(a)
        new_c = np.copy(c)

        for i in range(num_items):
            p = [a[i], c[i]]
            grad = self._compute_grad(p, i, r0, r1, q_nodes)
            hess = self._compute_hess(p, i, r0, r1, q_nodes)

            # Compute the update
            update = np.linalg.solve(hess, grad)
            
            # Update parameters for the current item
            x = p - step * update
            new_a[i], new_c[i] = x

            # Check bounds for a and c
            new_a[i] = np.clip(new_a[i], 0.3, 6.0)  
            c_min = -6 * new_a[i]  # From b in [-6, 6] -> c in [-6a, 6a]
            c_max =  6 * new_a[i]
            new_c[i] = np.clip(new_c[i], c_min, c_max)

        return new_a, new_c
    
    def create_subsets(self, ird, rnd_seed=1010, subset_size=2000, ird_mode=False):
        """
        Function to shuffle full item response data and create randomly sampled
        subsets (simple random sampling).
        """
        N = ird.shape[0]
        rng = np.random.default_rng(rnd_seed)
        sidx = np.arange(N)
        rng.shuffle(sidx) # shuffle the indices

        for i in range(0, N, subset_size):
            idx = sidx[i:i + subset_size]
            subset = ird[idx]

            if ird_mode:
                freq = np.ones(subset.shape[0], dtype=int)
                yield (subset, freq, None)
            else:
                pattern, freq = np.unique(subset, axis=0, return_counts=True)
                yield (pattern, freq, None)
    
    ##############################################################################
    ### BELOW ARE INTERNAL HELPER FUNCTIONS USED BY MAJOR OR WRAPPER FUNCTIONS ###
    ##############################################################################

    def _compute_Pi(self, a, c, q_nodes, eps=1e-10):
        """
        Internal function to compute response probability for an item.
        """
        z = a * q_nodes + c
        P = expit(z)
        return np.clip(P, eps, 1 - eps)
    
    def _compute_weights(self, q_nodes):
        """
        Internal function to set up quadrature weights.
        """
        w = np.exp(-0.5 * q_nodes ** 2) / np.sqrt(2 * np.pi)
        return w / np.sum(w)

    def _compute_nll(self, p, i, r0, r1, q_nodes):
        """
        Internal function to compute negative LL.
        """
        a, c = p[0], p[1]
        P = self._compute_Pi(a, c, q_nodes)
        return -np.sum(r1[i] * np.log(P) + r0[i] * np.log(1 - P))
    
    def _compute_grad(self, p, i, r0, r1, q_nodes):
        """
        Internal function to compute gradients for modified M-step update for an item.
        """
        a, c = p[0], p[1]
        P = self._compute_Pi(a, c, q_nodes)
        n = r0[i] + r1[i]
        dldz = n * P - r1[i]
        da = np.sum(dldz * q_nodes)
        dc = np.sum(dldz)
        return np.array([da, dc])
    
    def _compute_hess(self, p, i, r0, r1, q_nodes):
        """
        Internal function to compute Hessian for modified M-step update for an item.
        """
        a, c = p[0], p[1]
        P = self._compute_Pi(a, c, q_nodes)
        n = r0[i] + r1[i]
        ddl = n * P * (1 - P)
        hessian = np.zeros((2, 2))
        hessian[0, 0] = np.sum(ddl * q_nodes ** 2)
        hessian[0, 1] = hessian[1, 0] = np.sum(ddl * q_nodes)
        hessian[1, 1] = np.sum(ddl)
        return hessian
    
    def _compute_exp_diff(self, new, old=None, sample_size=None):
        """
        Internal function to compute exponentiated difference. If sample_size is None,
        this difference is the delta in Tian et al. (2012).
        """
        if old is None:
            return 0. 

        if sample_size is not None:
            delta = np.exp(-((new - old) / sample_size))
        else:
            delta = np.exp(-(new - old))
        return np.clip(delta, 0., 1.)

    def _setup(
            self, patterns, frequencies, q_nodes, s1_iter, s2_iter, subset_size,
            delta_thr, delta_useN, step, wait, window_chk, missing, tol, 
            get_se, verbose, debug_mode
        ):
        """
        Internal function to set up internal variables.
        """

        self.converged = False
        self.get_se = get_se
        self.verbose = verbose

        self.patterns = patterns
        self.r = frequencies
        self.q_nodes = q_nodes
        self.subset_size = subset_size
        self.delta_thr = delta_thr
        self.delta_useN = delta_useN
        self.step = step
        self.wait = wait
        self.window_chk = window_chk
        self.missing = missing
        self.tol = tol

        self.s1_iter = s1_iter
        self.s2_iter = s2_iter
        self.s1_time = 0.0
        self.s2_time = 0.0
        self.s1_end  = 0
        self.s2_end  = 0

        self.num_items = patterns.shape[1]
        self.a = np.ones(self.num_items)
        self.c = np.zeros(self.num_items)
        self.old_a = np.empty_like(self.a)
        self.old_c = np.empty_like(self.c)
        self.a_smt = None
        self.c_smt = None
        self.ll = 0.0

        self.r0 = np.zeros((self.num_items, len(q_nodes)))
        self.r1 = np.zeros_like(self.r0)

        self.num_updates = 0
        self.sample_size = sum(frequencies) if delta_useN else None

        if self.s2_iter > 0:
            self.sma_a = SMA(num_items=self.num_items, wait=wait, window_check=window_chk, tol=tol)
            self.sma_c = SMA(num_items=self.num_items, wait=wait, window_check=window_chk, tol=tol)
        else:
            self.sma_a = None
            self.sma_c = None

        self.response_matrix = (
            patterns if np.all(frequencies == 1)
            else np.repeat(patterns, frequencies.astype(int), axis=0)
        )
        self.ird_mode = bool(np.all(frequencies == 1)) # if not using pattern-freq format

        # Optional history tracking; only used if debug_mode is True
        self.debug_mode = debug_mode
        self.ll_history = []
        self.a_history = [self.a.copy()]
        self.c_history = [self.c.copy()]
        self.a_smoothed = []
        self.c_smoothed = []

    def _set_smoothed_arrays(self):
        """
        Internal function to get smoothed / averaged parameter values.
        """
        self.a_smoothed = self.a_history.copy()
        self.c_smoothed = self.c_history.copy()

    def _store_history(self, a=None, c=None, ll=None, a_smt=None, c_smt=None):
        """
        Internal function to store parameter estimation history.
        """
        if a is not None:   
            self.a_history.append(a)
        
        if c is not None:   
            self.c_history.append(c)
        
        if ll is not None:
            self.ll_history.append(ll)

        if a_smt is not None:
            self.a_smoothed.append(a_smt)

        if c_smt is not None:
            self.c_smoothed.append(c_smt)

    def _check_convergence(self, stage, iteration, max_iter, 
                           a_old, a_new, c_old, c_new, verbose=True,
                           delta=None, step=None, subset=None, num_updates=None):
        """
        Internal function to check convergence via largest absolute difference.
        """
        a_diff = np.abs(a_new - a_old)
        c_diff = np.abs(c_new - c_old)
        a_max_chg = np.max(a_diff)
        c_max_chg = np.max(c_diff)

        if verbose:
            stage_label = f"Stage {stage}: {iteration + 1:>3}/{max_iter}"
            extra = ""
            if delta is not None:
                extra += f", Delta: {delta:>6.6f}"
            if subset is not None:
                extra += f", Subset: {subset + 1:>3}"
            if step is not None:
                extra += f", Step: {step:>4.2f}"

            print(f"{stage_label}{extra}, "
                f"A: {np.argmax(a_diff)+1:>3} ({a_max_chg:>6.6f}), "
                f"C: {np.argmax(c_diff)+1:>3} ({c_max_chg:>6.6f})")

        if a_max_chg <= self.tol and c_max_chg <= self.tol:
            if verbose:
                msg = f"Early stopping at iter {iteration + 1}"
                if num_updates is not None:
                    msg += f" with {num_updates}."
                else:
                    msg += "."
                print(msg)
                print(f"Max. A chg: {a_max_chg:>7.7f}, Max. C chg: {c_max_chg:>7.7f}")
            return True  # if met convergence check

        return False # otherwise

    def _run_stage1(self):
        """
        Internal function to run Stage 1 of modified EM (or just standard EM).
        """
        if self.verbose:
            print("Stage 1 started.")

        old_ll = None
        start_time = time.time()

        for iter in range(self.s1_iter):
            np.copyto(self.old_a, self.a)
            np.copyto(self.old_c, self.c)

            # E-step on full data
            self.r0, self.r1, self.ll = self.run_estep(
                a=self.a, 
                c=self.c, 
                pattern=self.patterns, 
                r=self.r,
                q_nodes=self.q_nodes, 
                missing=self.missing,
            )

            # M-step
            self.a, self.c = self.run_mstep_full(
                a=self.a, c=self.c, r0=self.r0, r1=self.r1, q_nodes=self.q_nodes
            )

            # Transition check
            delta = self._compute_exp_diff(
                new=self.ll, old=old_ll, sample_size=self.sample_size
            )
            old_ll = self.ll

            if self.debug_mode:
                self._store_history(a=self.a, c=self.c, ll=self.ll)
            
            # Convergence check
            self.converged = self._check_convergence(
                stage=1, 
                iteration=iter, 
                max_iter=self.s1_iter, 
                a_old=self.old_a, 
                a_new=self.a, 
                c_old=self.old_c, 
                c_new=self.c, 
                verbose=self.verbose,
                delta=delta,
                step=None, 
                subset=None,
                num_updates=None,
            )

            if self.converged:
                break

            if self.s2_iter > 0 and delta >= self.delta_thr:
                if self.verbose:
                    print(f"Exiting Stage 1 and entering Stage 2 with delta = {delta:.4f}")
                break
        
        if self.get_se:
            self.fim = compute_XPD_numba(
                pattern=self.patterns, r=self.r, posteriors=self.posteriors, P=self.P, q_nodes=self.q_nodes
            )
            self.cov = self.compute_COV(fim=self.fim)
        
        self.s1_end = iter
        self.s1_time = time.time() - start_time
        self._set_smoothed_arrays()

    def _run_stage2(self):
        """
        Internal function to run Stage 2 of modified EM.
        """
        if self.verbose:
            print("Stage 2 started.")

        start_time = time.time()

        if self.get_se: # initalize fim matrix
            fim = np.zeros((2 * self.num_items, 2 * self.num_items))

        for iter in range(self.s2_iter):
            if self.get_se: # reset fim matrix to zero matrix
                fim.fill(0)

            ll = 0.0

            for b, (pattern, r, _) in enumerate(
                self.create_subsets(
                    ird=self.response_matrix, 
                    rnd_seed=iter, 
                    subset_size=self.subset_size,
                    ird_mode=self.ird_mode
                )):
                np.copyto(self.old_a, self.a)
                np.copyto(self.old_c, self.c)

                # E-step on subset
                r0, r1, ll_subset = self.run_estep(
                    a=self.a, c=self.c, pattern=pattern, r=r,
                    q_nodes=self.q_nodes, missing=self.missing
                )
                ll += ll_subset

                # Accumulate XPD
                if self.get_se:
                    fim += compute_XPD_numba(
                        pattern=pattern, r=r, posteriors=self.posteriors, P=self.P, q_nodes=self.q_nodes
                    )

                # M-step
                self.a, self.c = self.run_mstep_subset(
                    a=self.a, c=self.c, r0=r0, r1=r1, q_nodes=self.q_nodes, step=self.step
                )

                self.num_updates += 1

                # SMA smoothing
                self.a_smt = self.sma_a.update_sma(self.a)
                self.c_smt = self.sma_c.update_sma(self.c)

                if self.debug_mode:
                    self._store_history(a=self.a, c=self.c, ll=None, a_smt=self.a_smt, c_smt=self.c_smt)

                # Assign smoothed values back to item params
                self.a = self.a_smt
                self.c = self.c_smt

                # Convergence check
                self.converged = self._check_convergence(
                    stage=2, 
                    iteration=iter, 
                    max_iter=self.s1_iter, 
                    a_old=self.old_a, 
                    a_new=self.a, 
                    c_old=self.old_c, 
                    c_new=self.c, 
                    verbose=self.verbose,
                    delta=None,
                    step=self.step, 
                    subset=b,
                    num_updates=self.num_updates,
                )
                if self.converged:
                    break

            if self.debug_mode:
                self._store_history(ll=ll)

            if self.converged:
                break

        self.s2_end = self.s1_end + self.num_updates  # cumulative update count
        self.s2_time = time.time() - start_time

        if self.get_se:
            self.fim = fim
            self.cov = self.compute_COV(fim=self.fim)

    def _finalize(self):
        """
        Internal function to collect results and return.
        """
        returns = {
            'a': self.a,
            'c': self.c,
            
            'converged': self.converged,
            's1_end': getattr(self, 's1_end', None),
            's2_end': getattr(self, 's2_end', None),
            's1_time': getattr(self, 's1_time', 0.0),
            's2_time': getattr(self, 's2_time', 0.0),

            'debug_mode': int(self.debug_mode),
            'a_history': np.array(self.a_history),
            'c_history': np.array(self.c_history),
            'a_smoothed': np.array(self.a_smoothed),
            'c_smoothed': np.array(self.c_smoothed),
            'll_history': np.array(self.ll_history),
        }

        if self.get_se:
            if self.verbose:
                print("Computing SEs...")
            se_a, se_c = self.compute_SEs(self.cov)
            returns.update({'a_se': np.array(se_a), 'c_se': np.array(se_c)})

        return returns

# %%
if __name__ == '__main__':
    pass