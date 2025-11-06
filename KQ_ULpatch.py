# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 05:10:27 2024

@author: amita
"""
import sys; sys.path.insert(0, r'/home/amita/Patch_AP/1stversion/invert')
import mne
import os
import numpy as np
from invert.util import pos_from_forward
from SimFunction import weightmatrix
import pickle as pkl

# =========================================================================
# Section 1: Initial Parameter Setup
# This block defines the global parameters for the pre-computation script.
# It sets the diffusion for spatial smoothing, the maximum neighborhood
# order (patch size), and the maximum rank (complexity) for the patch models.
# =========================================================================
diffusion_parameter = 0.1
maxorder = 10
maxPatchRank = 5

# =========================================================================
# Section 2: Load and Prepare Forward Models
# This section loads multiple forward models with different spatial
# resolutions (e.g., 'coarse', 'fine'). This is a crucial step to avoid the
# "inverse crime," where the same model is used for both data simulation
# and source localization. Using different models makes the analysis more
# robust and realistic.
# =========================================================================
# Specify the path to the directory containing the forward model files
file_path = "forward_models"
model_paths = {
    "coarse-80": r"128_ch_coarse_80_ratio-fwd.fif", # 5124 sasmpling points
    "fine-80": r"128_ch_fine_80_ratio-fwd.fif", # 8196 sasmpling points
    "fine-50": r"128_ch_fine_50_ratio-fwd.fif",
    # # "fine-20": r"128_ch_fine_20_ratio-fwd.fif"
    }
fwds = dict()
for name, model_path in model_paths.items():
    fwd_inv = mne.read_forward_solution(os.path.join(file_path,model_path), verbose=0)
    fwd_inv = mne.convert_forward_solution(fwd_inv, force_fixed=True)
    fwds[name] = fwd_inv

# =========================================================================
# Section 3: Pre-computation Loop for Each Forward Model
# This main loop iterates through each of the loaded forward models. For
# each model, it pre-computes two key components required for patch-based
# source localization algorithms:
# 1. Weight Matrices (KQ): Defines the spatial extent and smoothness of patches.
# 2. Leadfield Patches (ULpatch): The principal components of the leadfield
#    for each possible patch.
# These pre-computations significantly speed up the actual inverse solution process later.
# =========================================================================
    
for fwd_name, fwd_inv in fwds.items():
    print("Forward Model: ", fwd_name)
    leadfields = fwd_inv['sol']['data']  
    n_chans, n_dipoles = leadfields.shape
    pos_inv = pos_from_forward(fwd_inv)

    # ==============================================
    # Weights with different smoothness order
    # ==============================================  
    
    # =====================================================
    # Subsection 3.1: Compute and Save Weight Matrices (KQ)
    # This block calculates spatial weight matrices for different
    # neighborhood orders (extents). Each matrix defines how much
    # neighboring dipoles contribute to a central patch dipole,
    # effectively defining the patch's size and smoothness. The
    # results are saved to a file for later use.
    # =====================================================
    Weights = []
    for ik in range(0,maxorder+1):
        order = (ik,ik+1)
        Weig = weightmatrix(fwd_inv,n_dipoles,diffusion_parameter,order,diffusion_smoothing=True,verbose=0)
        Weights.append(Weig)
        
    # Save the dictionary to a file using pickle
    with open('KQ_MaxExtent_{}_{}.pkl'.format(maxorder,fwd_name), 'wb') as file:
        pkl.dump(Weights, file)
        
    # ==============================================
    # Leadfield Patch
    # ==============================================      
    
    # =====================================================
    # Subsection 3.2: Compute and Save Leadfield Patches (ULpatch)
    # This block pre-computes the principal components of the
    # leadfield for every possible patch, defined by its center,
    # extent (n_orders), and rank (complexity). It uses Singular
    # Value Decomposition (SVD) to find an orthonormal basis (U)
    # for the signal subspace of each patch. This avoids costly
    # re-computation during the inverse solution.
    # =====================================================
    # # Lpatch with different smoothness order   
    Lpatch_ALLFulls, Sig_ALLFulls = [], []                 
    for Patrank in range(0,maxPatchRank):    
        print(Patrank)
        Lpatch_Fulls, Sig_Fulls = [], []
        for n_orders in range(0,maxorder+1,1):
            # print(n_orders)             
            if n_orders==0 or Patrank==0:
                EstPatchrank = 1
            else:
                EstPatchrank = Patrank
            Lpatch_Full, Sig_Full = [], []
            for nn in range(n_dipoles):
                Dipoles_inpatch = np.where(Weights[n_orders][nn]!=0)[0]
                L = leadfields[:,Dipoles_inpatch] 
                                              
                U, Sig, vt1 = np.linalg.svd(L, full_matrices=False)
                Sig_Full.append(Sig[:EstPatchrank])
                Lpatch_Full.append(U[:, :EstPatchrank])  # 128 x EstPatchrank
            Lpatch_Fulls.append(Lpatch_Full)
            Sig_Fulls.append(Sig_Full)    
                
        Lpatch_ALLFulls.append(Lpatch_Fulls)
        Sig_ALLFulls.append(Sig_Fulls)
        
            
                
    # Save the dictionary to a file using pickle
    with open('ULpatch_MaxExtent_{}_MaxRank_{}_{}.pkl'.format(maxorder,maxPatchRank,fwd_name), 'wb') as file:
        pkl.dump(Lpatch_ALLFulls, file)
    
    
        
        