# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:06:06 2023
@author: amita
"""

# Simulate activity for rank2 patch

from copy import deepcopy
import numpy as np
import colorednoise as cn
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix, vstack
import mne
import pandas as pd
        
# =========================================================================
# Section 1: Main Data Simulation Generator
# This function is a Python generator that creates batches of simulated
# EEG/MEG data. It simulates brain activity originating from spatially
# extended "patches" of cortex, projects this activity to the sensors using
# a forward model, and adds noise. It's designed to be highly flexible,
# allowing control over the number of sources, their size (order), complexity
# (rank), temporal dynamics, and the signal-to-noise ratio (SNR).
# =========================================================================
def SimulationGenerator(fwd, TimeCourses, Patchranks, Smoothness_order, random_sequence, use_cov=True, batch_size=1284, batch_repetitions=30, n_sources=10,
              n_orders=2, amplitude_range=(0.001,1), n_timepoints=20,
              snr_range=(1, 100), n_timecourses=5000, beta_range=(0, 3),
              return_mask=True, scale_data=True, return_info=False,
              add_forward_error=False, forward_error=0.1, remove_channel_dim=False,
              inter_source_correlation=0.5, diffusion_smoothing=True,
              diffusion_parameter=0.1, fixed_covariance=False, iid_noise=False,
              random_seed=None, verbose=0):

    # --- Subsection 1.1: Initialization and Leadfield Loading ---
    # Initialize the random number generator for reproducibility and load the leadfield matrix.
    rng = np.random.default_rng(random_seed) # random seed
    leadfield = deepcopy(fwd["sol"]["data"]) # Leadfield Matrix (A)
    n_chans, n_dipoles = leadfield.shape

    # --- Subsection 1.2: Set Up Simulation Parameters ---
    # Determine the range for the number of sources and patch orders per sample.
    if isinstance(n_sources, (int, float)):
        n_sources = [n_sources, n_sources+1]
    minn_sources, maxn_sources = n_sources

    if isinstance(n_orders, (tuple, list)):
        min_order, max_order = n_orders
        if min_order == max_order:
            max_order += 1

    # --- Subsection 1.3: Create Spatial Smoothing Operator (K) ---
    # Calculate the graph Laplacian from the source space adjacency to define spatial relationships.
    # The gradient matrix (K) is derived from this, acting as a smoothing operator.
    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=verbose)
    adjacency = csr_matrix(adjacency)
    gradient = np.identity(n_dipoles) - diffusion_parameter*laplacian(adjacency)
    gradient = csr_matrix(gradient) # smoothing operator (K)
    del adjacency

    # --- Subsection 1.4: Pre-compute Higher-Order Smoothing Matrices (K^q) ---
    # Iteratively apply the gradient to itself to create smoothing matrices for increasing
    # spatial extents (orders). `sources` will store K^0, K^1, K^2, etc., stacked vertically.
    sources = csr_matrix(np.identity(n_dipoles))
    for i in range(1, max_order):
        new_sources = csr_matrix(sources.toarray()[-n_dipoles:, -n_dipoles:]) @ gradient
        row_maxes = new_sources.max(axis=0).toarray().flatten()
        new_sources = new_sources / row_maxes[np.newaxis]
        sources = vstack([sources, new_sources])

    # --- Subsection 1.5: Select Source Locations for the Batch ---
    # Randomly determine the number of sources for each sample in the batch and then
    # randomly select the center dipole locations for these sources.
    n_candidates = n_dipoles
    n_sources_batch = rng.integers(minn_sources, maxn_sources, batch_size)
    selection = [rng.integers(0, n_candidates, n) for n in n_sources_batch]

    # --- Subsection 1.6: Main Loop to Generate Batch Data ---
    # Iterate through each sample in the batch to generate the full source and sensor data.
    temp = 0
    Y = np.zeros((batch_size,n_chans, n_timepoints))
    KQFull, EFull, WFull, SFull, SdotFull, SSFull = [], [], [], [], [], []

    for i in range(0,batch_size):
        Patch_order = random_sequence[i] # source extent (l)
        if Patch_order >= 0:
            start = int(Patch_order*n_dipoles)
            kq = sources.toarray()[start:start+n_dipoles, :] #k^q where q=l
        kq = csr_matrix(kq)
        
        x = np.zeros((n_chans, n_timepoints))      
        
        KQ, E, W, S, Sdot, SS = [], [], [], [], [], []
        for k in range(0,minn_sources):
            # Patch basis matrix (E) corresponding to ith patch
            
            Dipoles_inpatch = np.where(kq[selection[i][k]].toarray()!=0)[1]            
            # Initialize the patch matrix E_i for this patch
            E_i = np.zeros((n_dipoles, len(Dipoles_inpatch)))
            
            # Loop through each dipole in the current patch and set the corresponding basis vector
            for j, idx in enumerate(Dipoles_inpatch):
                # Create the standard basis vector for the current dipole
                e_ij = np.zeros(n_dipoles)
                e_ij[idx] = 1
                
                # Assign this vector to the patch matrix E_i
                E_i[:, j] = e_ij
            
            if Patch_order == 0:
                min_Prank = 1
            else:
                min_Prank = Patchranks[k]
                
            # W weight matrix
            Wamp = np.random.normal(0, 10, size=(len(Dipoles_inpatch), min_Prank))
            Wamp = Wamp.reshape(-1, 1) if min_Prank == 1 else Wamp
            s = np.array(TimeCourses[temp:temp+min_Prank])
            sdot = Wamp @ s
            ss = kq @ E_i @ sdot
            
            x = x + leadfield @ ss
            
            # Concatenate
            KQ.append(kq)
            E.append(E_i)
            W.append(Wamp)
            S.append(s)
            Sdot.append(sdot)
            SS.append(ss)
            

        # Concatenate
        KQFull.append(KQ)
        EFull.append(E)
        WFull.append(W)
        SFull.append(S)
        SdotFull.append(Sdot)
        SSFull.append(SS)
        
        snr_levels = rng.uniform(low=snr_range[0], high=snr_range[1], size=batch_size)
        x1 = add_white_noise(x, snr_range[0], rng, iid=iid_noise)
        Y[i] = x1

        
    # --- Subsection 1.7: Yield Batch Output ---
    # Return the generated data. If requested, also include a pandas DataFrame with metadata about the simulation.
    if return_info:
        info = pd.DataFrame(dict(
            n_sources=n_sources_batch, 
            # amplitudes=amplitude_values, 
            snr=snr_levels, 
            # inter_source_correlations=inter_source_correlations, 
            n_orders=[[min_order, max_order],]*batch_size,
            diffusion_parameter=[diffusion_parameter,] * batch_size,
            n_timepoints=[n_timepoints,] * batch_size,
            n_timecourses=[n_timecourses,] * batch_size,
            iid_noise=[iid_noise,] * batch_size
            ))
        yield (Y, KQFull, EFull, WFull, SFull, SdotFull, SSFull, selection, info)
    else:
        yield (Y, KQFull, EFull, WFull, SFull, SdotFull, SSFull, selection)
        
        
# =========================================================================
# Section 2: Noise Addition Function
# This helper function adds white noise to a clean signal matrix to achieve
# a specified Signal-to-Noise Ratio (SNR).
# =========================================================================
def add_white_noise(X_clean, snr, rng, iid=False):
    ''' '''
    X_noise = rng.standard_normal(X_clean.shape)
    
    # # According to Adler et al.:
    X_clean_energy = np.trace(X_clean@X_clean.T)/(X_clean.shape[0]*X_clean.shape[1])
    noise_var = X_clean_energy/snr
    scaler = np.sqrt(noise_var)

    X_full = X_clean + X_noise*scaler
    return X_full

# =========================================================================
# Section 3: Weight Matrix Calculation Function
# This function computes and returns a spatial weight/smoothing matrix (K^q)
# for a specified smoothness order (q). It encapsulates the logic from
# subsections 1.3 and 1.4 of the main generator.
# =========================================================================
def weightmatrix(fwd,n_dipoles,diffusion_parameter,n_orders,diffusion_smoothing,verbose):
    if isinstance(n_orders, (tuple, list)):
        min_order, max_order = n_orders
        if min_order == max_order:
            max_order += 1
    else:
        min_order = 0
        max_order = n_orders

    adjacency = mne.spatial_src_adjacency(fwd["src"], verbose=verbose)
    # Convert to sparse matrix for speedup
    adjacency = csr_matrix(adjacency)
    if diffusion_smoothing:
        # print("Using Diffusion Smoothing on Graph")
        gradient = np.identity(n_dipoles) - diffusion_parameter*laplacian(adjacency)
    else:
        gradient = abs(laplacian(adjacency))
    
    # Convert to sparse matrix for speedup
    gradient = csr_matrix(gradient)
    del adjacency    
    
    
    sources = csr_matrix(np.identity(n_dipoles))
    for i in range(1, max_order): # store sources of smoothnessorder from 0 to max_order      
        # New Smoothness order
        new_sources = csr_matrix(sources.toarray()[-n_dipoles:, -n_dipoles:]) @ gradient
        
        # Scaling
        row_maxes = new_sources.max(axis=0).toarray().flatten()
        new_sources = new_sources / row_maxes[np.newaxis]

        sources = vstack([sources, new_sources])

    # if min_order > 0:
    start = int(min_order*n_dipoles)
    sources_All = sources.toarray()
    sources = sources.toarray()[start:start+n_dipoles, :]    
    
    # Weights_Conc = csr_matrix(sources_All)
    Weights = sources # final sources of smoothness order
    return Weights
        
        
    
import numpy as np
from scipy.spatial.distance import cdist

# =========================================================================
# Section 4: Gaussian Smoothing Functions
# These helper functions are used for applying spatial smoothing based on
# a Gaussian kernel, which is an alternative method to graph-based smoothing.
# =========================================================================
def gaussian_kernel(x, sigma):
    """
    Compute the Gaussian kernel value at point x.

    Parameters:
    - x: Distance value.
    - sigma: Standard deviation of the Gaussian kernel.

    Returns:
    - Gaussian kernel value.
    """
    return np.exp(-0.5 * (x / sigma)**2)

def spatial_smooth(D, Pos, sigma):
    """
    Perform spatial smoothing on data D using Gaussian smoothing.

    Parameters:
    - D: Input data of shape [20x1].
    - Pos: Position data of shape [20x3].
    - sigma: Standard deviation of the Gaussian kernel.

    Returns:
    - Smoothed data.
    """
    smoothed_D = np.zeros_like(D)
    for i in range(len(D)):
        # Calculate distances between the current point and all other points
        distances = cdist(Pos, Pos[i:i+1])
        # Calculate weights using Gaussian kernel
        weights = gaussian_kernel(distances, sigma)
        # Normalize weights
        weights /= np.sum(weights)
        # Apply weights to smooth the data
        smoothed_D[i] = np.sum(weights * D)
    return smoothed_D


