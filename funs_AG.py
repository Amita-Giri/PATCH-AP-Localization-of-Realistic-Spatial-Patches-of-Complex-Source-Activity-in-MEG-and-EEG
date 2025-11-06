from joblib import Parallel, delayed
from multiprocessing import Pool
from time import time, sleep
import mne
import numpy as np
import os
from scipy.sparse import csr_matrix

# =========================================================================
# Section 1: Worker Function for Source Localization
# This function, `predict_sources_2`, acts as a worker that applies a list
# of different source localization algorithms (solvers) to a *single*
# sample of EEG/MEG data (`x_sample`). For each solver, it computes the
# inverse solution to estimate the brain source activity, measures the time
# taken for both setting up the solver and applying it, and returns the
# estimated source time courses along with the timing information.
# =========================================================================
def predict_sources_2(solver_dicts, fwd, info, x_sample, sim_info):
	stc_dict = dict()
	proc_time_make = dict()
	proc_time_apply = dict()
	start_make = 0
	end_make = 0

	for solver_dict in solver_dicts:
		solver_name = solver_dict["display_name"]
		make_args = solver_dict["make_args"]
		apply_args = solver_dict["apply_args"]
		recompute_make = solver_dict["recompute_make"]
		solver = solver_dict["solver_name"]

		evoked = mne.EvokedArray(x_sample, info, tmin=0)
		if not solver.made_inverse_operator or recompute_make:
			start_make = time()
			solver.make_inverse_operator(fwd, evoked, alpha="auto", **make_args)
			end_make = time()
		
		start_apply = time()
		stc = solver.apply_inverse_operator(evoked, **apply_args)
		end_apply = time()
		
		stc_dict[solver_name] = csr_matrix(stc.data)
		proc_time_make[solver_name] = end_make - start_make
		proc_time_apply[solver_name] =  end_apply - start_apply
    
	return stc_dict, proc_time_make, proc_time_apply

# =========================================================================
# Section 2: Parallel Processing Manager
# This function, `predict_sources_parallel3`, manages the parallel execution
# of the source localization process over an entire batch of data (`x_test`).
# It first performs a "warm-up" step by preparing the inverse operator for
# each solver. Then, it uses `joblib.Parallel` to distribute the `predict_sources_2`
# worker function across multiple CPU cores, applying it to each data sample
# in the batch concurrently. This significantly speeds up the analysis of
# large datasets.
# =========================================================================
def predict_sources_parallel3(solver_dicts, fwd, info, x_test, sim_info, n_jobs=-1):
    # prepare solvers
	evoked = mne.EvokedArray(x_test[0], info, tmin=0)
	for solver_dict in solver_dicts:
		solver_dict["solver_name"].make_inverse_operator(fwd, evoked, alpha="auto", **solver_dict["make_args"])

	res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(predict_sources_2)(solver_dicts, fwd, info, x_sample, sim_info_row) for x_sample, (_, sim_info_row) in zip(x_test, sim_info.iterrows()))
	return res

