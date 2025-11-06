import numpy as np
import colorednoise as cn

# =========================================================================
# Section 1: Generate a Set of Correlated Time Courses
# This function, `gen_correlated_sources`, creates a set of `Q` sinusoidal
# time series, each of length `T`, that have a specified correlation
# coefficient (`corr_coeff`) with each other. It works by first generating
# independent signals with random frequencies and phases, and then using the
# Cholesky decomposition of a target covariance matrix to introduce the
# desired correlation. A special case handles perfectly coherent sources.
# =========================================================================
def gen_correlated_sources(corr_coeff, T, Q, random_seed):
    rng = np.random.default_rng(random_seed)
    Cov = np.ones((Q, Q)) * corr_coeff + np.diag(np.ones(Q) * (1 - corr_coeff))  # required covariance matrix

    freq = np.random.randint(10, 31, size=Q)  # random frequencies between 10Hz to 30Hz
    phases = 2 * np.pi * np.random.rand(Q)  # random phases
    t = np.arange(10 * np.pi / T, 10 * np.pi + 0.01, 10 * np.pi / T)  # time vector

    Signals = np.sqrt(2) * np.cos(2 * np.pi * freq[:, np.newaxis] * t + phases[:, np.newaxis])  # the basic signals

    if corr_coeff < 1:
        A = np.linalg.cholesky(Cov)  # Cholesky Decomposition
        S = A @ Signals
    else:
        S = np.tile(Signals[0], (Q, 1))  # Coherent Sources

    return S

# =========================================================================
# Section 2: Create a Batch of Correlated Time Courses
# This function acts as a wrapper to generate a complete set of time courses
# for a simulation batch. It repeatedly calls `gen_correlated_sources` for
# each sample in the batch and concatenates the results into a single large
# numpy array, which can then be used to drive the source activity in a
# larger simulation framework.
# =========================================================================
def TimeCourse(corr_coeff, batch_size, n_timepoints, n_sources, random_seed):
    rng = np.random.seed(random_seed)
    TimeCourses = np.zeros((batch_size * n_sources, n_timepoints))

    temp = 0
    
    for _ in range(batch_size):
        time_courses = gen_correlated_sources(corr_coeff, n_timepoints, n_sources, random_seed)
        TimeCourses[temp:temp+n_sources,:] = time_courses
        temp = temp+n_sources

    return TimeCourses



# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan  8 10:04:17 2024

# @author: amita
# """
# import numpy as np
# import colorednoise as cn

# def gen_correlated_sources(corr_coeff, T, Q,random_seed):
#     Cov = np.ones((Q, Q)) * corr_coeff + np.diag(np.ones(Q) * (1 - corr_coeff))  # required covariance matrix

#     freq = np.random.randint(10, 31, size=Q)  # random frequencies between 10Hz to 30Hz

#     phases = 2 * np.pi * np.random.rand(Q)  # random phases

#     t = np.arange(10 * np.pi / T, 10 * np.pi+0.01, 10 * np.pi / T) #[:-1]  # time vector
    
#     Signals = np.sqrt(2) * np.cos(2 * np.pi * freq[:, np.newaxis] * t + phases[:, np.newaxis])  # the basic signals

#     if corr_coeff < 1:
#         A = np.linalg.cholesky(Cov)  # Cholesky Decomposition
#         # S = np.dot(A, Signals)
#         S = A @ Signals
#     else:
#         S = np.tile(Signals[0], (Q, 1))  # Coherent Sources

#     return S

# def TimeCourse(corr_coeff,use_cov=True, batch_size=1284, batch_repetitions=30, n_sources=10, 
#               n_orders=2, amplitude_range=(0.001,1), n_timepoints=20, 
#               snr_range=(1, 100), n_timecourses=5000, beta_range=(0, 3),
#               return_mask=True, scale_data=True, return_info=False,
#               add_forward_error=False, forward_error=0.1, remove_channel_dim=False, 
#               inter_source_correlation=0.5, diffusion_smoothing=True, 
#               diffusion_parameter=0.1, fixed_covariance=False, iid_noise=False, 
#               random_seed=None, Patchrank=2, verbose=0):
    
#     rng = np.random.default_rng(random_seed)
#     # betas = rng.uniform(*beta_range,n_timecourses)
#     # time_courses = np.stack([cn.powerlaw_psd_gaussian(beta, n_timepoints) for beta in betas], axis=0)
#     # # Normalize time course to max(abs()) == 1
#     # time_courses = (time_courses.T / abs(time_courses).max(axis=1)).T
    
#     time_courses = np.zeros((batch_size * n_sources, n_timepoints))
#     temp = 0
#     # time_courses = []
#     for i in range(0,batch_size):
#         ant = gen_correlated_sources(corr_coeff, n_timepoints, n_sources)
#         time_courses[temp:temp+n_sources,:] = ant
#         print(np.corrcoef(ant))
#         temp = temp+n_sources

#     # time_courses = np.load('time_courses.npy')
#     amplitude_values = rng.uniform(*amplitude_range, size=n_timecourses)
    
#     TimeCourses = []
#     for i in range(0,n_timecourses):
#         res =  np.dot(amplitude_values[i], time_courses[i])
#         TimeCourses.append(res)
        
#     yield (TimeCourses)
       
    
