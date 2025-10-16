# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 06:23:29 2023

@author: amita
"""
from scipy.linalg import eig
import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix, vstack
import time

# =========================================================================
# Section 1: Weighted Adaptive Pursuit (weighted_ap) Algorithm
# This function implements a source localization algorithm based on adaptive
# pursuit. It iteratively identifies the most likely brain source patches
# by finding the patch that best explains the sensor data at each step.
# The algorithm can test various combinations of patch complexities (ranks)
# and includes an optional refinement step to improve the solution.
# =========================================================================

def weighted_ap(y, n_sources, Weights, max_iter, Lpatch_Fulls, n_orders, diffusion_parameter, n_dipoles, refine_solution=True, covariance_type="AP"):
    ''' Create the AP-MUSIC inverse solution to the EEG data.
    Parameters
    ----------
    y : numpy.ndarray
        EEG data matrix (channels, time)
    n : int/ str
        Number of eigenvectors to use or "auto" for l-curve method.
    k : int
        Number of recursions.
    stop_crit : float
        Criterion to stop recursions. The lower, the more dipoles will be
        incorporated.
    refine_solution : bool
        If True: Re-visit each selected candidate and check if there is a
        better alternative.

    Return
    ------
    x_hat : numpy.ndarray
        Source data matrix (sources, time)
    '''  
    n_chans = y.shape[0]  
    
    mu = 0  # 1e-3
    C = y@y.T + mu * np.trace(np.matmul(y,y.T)) * np.eye(y.shape[0]) # Array Covariance matrix   
    
    # Generate combinations of source ranks to test different source models

    Num_sources, Source_Rank = [], []
    for comb in combinations(n_sources):
        ranks = np.where(np.array(comb)!=0)[0]
        num_sources, sour_ranks = [], []
        for rr in range(len(ranks)):
            num_sources.append(comb[ranks[rr]])
            sour_ranks.append(ranks[rr])                        
        Source_Rank.append(sour_ranks)
        # print(Source_Rank)
        Num_sources.append(num_sources)
    
    from itertools import permutations
    SAP_Grand = []
    for index in range(len(Source_Rank)):
        # print(index)
        n_sources_pat = np.sum(Num_sources[index])
        Rankorder = [source for rank, num_sources, ii in zip(Source_Rank[index], Num_sources[index], range(0, 13)) for source in [rank] * num_sources]
        Ranklist = list(set(permutations(Rankorder))) # order also matters
        for rankruns in range(0,len(Ranklist)):
            # print(index,rankruns)
            Rankorder = Ranklist[rankruns]
            
            # --- Step 1: Find the first source patch ---
            # Calculate the cost function (trace of projected covariance) for all possible patches

            ap_val1 = np.zeros((n_orders,n_dipoles))
            BRank, BOrder, BDipole, S_AP, maxval  = [], [], [], [], []
            akx = Rankorder[0]
    
            #################################            
            for ik in range(0,n_orders):
                for nn in range(n_dipoles):
                    Lpatch = Lpatch_Fulls[ik][nn]
                    upper = np.dot(Lpatch.T, np.dot(C, Lpatch))
                    lower = np.linalg.inv(np.dot(Lpatch.T, Lpatch))
                    result = np.dot(upper, lower)
                    ap_val1[ik, nn] = np.trace(result)    

            # Find the dipole/ patch with highest correlation with the residual
            best_order, best_dipole = np.unravel_index(np.argmax(ap_val1), ap_val1.shape)
            best_rank = akx
            S_AP.append([best_rank, best_order, best_dipole])
            maxval.append([np.max(ap_val1)])
            best_dipole = np.where(Weights[best_order][best_dipole]!=0)[0]   
            BOrder.append(best_order)
            BDipole.append(best_dipole) 
            BRank.append(best_rank)
            
            # (b) Now, add one source at a time
            # --- Step 2: Sequentially find remaining sources ---

            for ii in range(1, n_sources_pat):
                akx = Rankorder[ii]
                # Create a projection matrix Q to remove the influence of already found sources
                A = np.hstack([Lpatch_Fulls[order][dipole] for rank,order,dipole in S_AP])        
                P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
                Q = np.identity(P_A.shape[0]) - P_A

                # Recalculate the cost function in the projected (residual) space
                ap_val2 = np.zeros((n_orders,n_dipoles))
                QCQ = np.dot(Q, C).dot(Q)
                for ik in range(0,n_orders):
                    for nn in range(n_dipoles):
                        Lpatch = Lpatch_Fulls[ik][nn]
                        upper = np.dot(Lpatch.T, np.dot(QCQ, Lpatch))
                        lower = np.linalg.inv(np.dot(Lpatch.T, np.dot(Q, Lpatch)))
                        result = np.dot(upper, lower)
                        ap_val2[ik,nn] = np.trace(result)   
                
                #mask
                # Mask out already selected dipole locations to avoid re-selection
                for temp in range(0,len(BOrder)):      
                    ap_val2[:, BDipole[temp]] = 0 # from previous iteration
                # Find the dipole/ patch with highest correlation with the residual
                # Find the next best patch and store its parameters
                best_order, best_dipole = np.unravel_index(np.argmax(ap_val2), ap_val2.shape)
                best_rank = akx
                S_AP.append([best_rank, best_order, best_dipole])
                maxval.append([np.max(ap_val2)])
                
                best_dipole = np.where(Weights[best_order][best_dipole]!=0)[0]   
                BOrder.append(best_order)
                BDipole.append(best_dipole)
                BRank.append(best_rank)
                
            # --- Step 3: (Optional) Refine the solution ---
            S_AP_2 = deepcopy(S_AP)
            if len(S_AP) > 1 and refine_solution:
                for iter in range(max_iter):
                    S_AP_2_Prev = deepcopy(S_AP_2)
                    for q in range(len(S_AP_2)):
                        akx = Rankorder[q]
                        S_AP_TMP = S_AP_2.copy()
                        S_AP_TMP.pop(q) 
                        
                        A = np.hstack([Lpatch_Fulls[order][dipole] for rank,order,dipole in S_AP_TMP])                                        
                        P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
                        Q = np.identity(P_A.shape[0]) - P_A              
                        ap_val2 = np.zeros((n_orders,n_dipoles))
                        QCQ = np.dot(Q, C).dot(Q)
                        for ik in range(0,n_orders):
                            for nn in range(n_dipoles):
                                Lpatch = Lpatch_Fulls[ik][nn]
                                upper = np.dot(Lpatch.T, np.dot(QCQ, Lpatch))
                                lower = np.linalg.inv(np.dot(Lpatch.T, np.dot(Q, Lpatch)))
                                result = np.dot(upper, lower)
                                ap_val2[ik,nn] = np.trace(result)  
                        
                        #mask
                        ax1 = BOrder.copy()
                        ax1.pop(q)
                        ax = BDipole.copy()
                        ax.pop(q) 
                        #mask
                        for temp in range(0,len(ax1)):      
                            ap_val2[:, ax[temp]] = 0 # from previous iteration
                        
                        best_order, best_dipole = np.unravel_index(np.argmax(ap_val2), ap_val2.shape)
                        best_rank = akx
                        S_AP_2[q] = [best_rank, best_order, best_dipole]
                        maxval[q] = [np.max(ap_val2)]
                        
                        best_dipole = np.where(Weights[best_order][best_dipole]!=0)[0]   
                        BOrder[q] = best_order
                        BDipole[q] = best_dipole
                        BRank[q] = best_rank
                        
                    if S_AP_2 == S_AP_2_Prev:  # and iter>0:
                        break
             
            # return S_AP_2, BOrder, BDipole 
            SAP_Grand.append([S_AP_2, BRank, BOrder, BDipole ])

    indexfinal =0
    return SAP_Grand[indexfinal][0], SAP_Grand[indexfinal][1], SAP_Grand[indexfinal][2], SAP_Grand[indexfinal][3], SAP_Grand

# =========================================================================
# Section 2: Reconstruct Source Time Courses (J_Estimated)
# This function takes the estimated patch parameters (location, order, rank)
# from a localization algorithm and reconstructs the full brain source activity
# over time. It first estimates the time course for each patch and then
# projects this activity back onto the individual dipoles within the patch.
# =========================================================================
    
def J_Estimated(batch_size,n_dipoles,n_timepoints,Y,n_sourcespatch,Bestorder_WAp,BRank_WAp,EstLoc_WAp,Weights,leadfields,Lpatch_Fulls,mult):
    J_pred_Patchh = []
    for i in range(0,batch_size):
        J_pred_WAP = np.zeros((n_dipoles,n_timepoints))  
        # check whether the same source is localized again
        if len(np.unique(BRank_WAp[i])) != len(BRank_WAp[i]) and len(np.unique(Bestorder_WAp[i])) != len(Bestorder_WAp[i]) and len(np.unique(EstLoc_WAp[i])) != len(EstLoc_WAp[i]):             
            unique_indices = np.unique(EstLoc_WAp[i], return_index=True)[1]
            # Select corresponding elements from Bestorder_WAp and BRank_WAp
            selected_EstLoc_WAp = [EstLoc_WAp[i][j] for j in unique_indices]
            selected_bestorder_wap = [Bestorder_WAp[i][j] for j in unique_indices]
            selected_brank_wap = [BRank_WAp[i][j] for j in unique_indices]
            
            BRank_WAp[i] = selected_brank_wap
            Bestorder_WAp[i] = selected_bestorder_wap
            EstLoc_WAp[i] = selected_EstLoc_WAp
            
            n_sourcespatch = mult*(len(EstLoc_WAp[i]))            
            
        # --- Step 1: Estimate patch time courses ---
        # Construct the combined leadfield matrix L1 for all identified patches
        S_AP = []  
        for j in range(0, len(Bestorder_WAp[i])):                
            S_AP.append([BRank_WAp[i][j], Bestorder_WAp[i][j], EstLoc_WAp[i][j]])
            
        L1 = np.hstack([Lpatch_Fulls[order][dipole] for rank, order, dipole in S_AP])
        Sddot = (np.linalg.inv(L1.T @ L1) @ L1.T) @ Y[i]

        # --- Step 2: Project patch activity to dipole activity ---
        Sdot = []  # Initialize Sdot as a list to collect results
        E = []
        for j in range(0, len(Bestorder_WAp[i])): 
            kq = Weights[Bestorder_WAp[i][j]]
            Dipoles_inpatch = np.where(kq[EstLoc_WAp[i][j]]!=0)[0]    
            
            # Initialize the patch matrix E_i for this patch
            E_i = np.zeros((n_dipoles, len(Dipoles_inpatch)))
            
            # Loop through each dipole in the current patch and set the corresponding basis vector
            for jj, idx in enumerate(Dipoles_inpatch):
                # Create the standard basis vector for the current dipole
                e_ij = np.zeros(n_dipoles)
                e_ij[idx] = 1
                
                # Assign this vector to the patch matrix E_i
                E_i[:, jj] = e_ij
                
            E.append(E_i)
            
            if Bestorder_WAp[i][j] != 0:
                min_Prank = BRank_WAp[i][j]
            else:
                min_Prank = 1
    
            # Extract the appropriate slice of Sddot
            sddot = Sddot[j * min_Prank : (j+1) * min_Prank , :]
            
            # Get the corresponding L1 matrix
            L1 = leadfields[:,Dipoles_inpatch]
            
            # Perform singular value decomposition (SVD)
            U, Sig, vt1 = np.linalg.svd(L1, full_matrices=False)
            
            # Convert singular values to a diagonal matrix
            Sig_diag = np.diag(Sig[:min_Prank])  # Only take the first 'min_Prank' singular values

            print("Dimension: ", len(Sig_diag), len(vt1[:min_Prank, :]), len(sddot))

            # Calculate the pseudo-inverse and sdot
            sdot = np.linalg.pinv(Sig_diag @ vt1[:min_Prank, :]) @ sddot
            
            # Append sdot to the list
            Sdot.append(sdot)

        # Stack the collected sdot vertically
        # --- Step 3: Combine and smooth final source estimate ---
        Sdot = np.vstack(Sdot)
        E = np.hstack(E)
        kr = Weights[mult]
        kr = csr_matrix(kr)
        # Apply final spatial smoothing (kr) to the combined dipole activities
        J_pred_Patch = kr @ (E @ Sdot) 
        J_pred_Patchh.append(J_pred_Patch)          
                
    return J_pred_Patchh

# =========================================================================
# Section 3: Patch RAP-MUSIC Algorithm
# This function implements a patch-based version of the Recursively Applied
# and Projected (RAP) MUSIC algorithm. It iteratively finds sources by
# searching for the patch whose signal subspace has the largest intersection
# with the data's signal subspace. After each source is found, its contribution
# is projected out of the data.
# =========================================================================
    
def PatchRAP(y, n, k, n_dipoles, Lpatch_Fulls, n_orders,akx):
    # Compute data covariance and its SVD to find the initial signal subspace (Us)
    C = y@y.T
    n_chans = y.shape[0] 
    

    I = np.identity(n_chans)
    Q = np.identity(n_chans)
    U, D, _= np.linalg.svd(C, full_matrices=False)
    
    n_comp = deepcopy(n)
    # akx = 2
    
    Us = U[:, :n_comp]    
    
    C_initial = Us @ Us.T
    dipole_idc = []
    source_covariance = np.zeros(n_dipoles)
    S_AP = []

    
    BRank, BOrder, BDipole, S_AP, maxval  = [], [], [], [], []
    for i in range(int(k/akx)):
        # print(f"Source Iteration ", i)
        Ps = Us @ Us.T
        # PsQ = Q.T @ Ps @ Q
        PsQ = Q.T @ Ps.T @ Ps @ Q
        Q1 = Q.T @ Q
        
        mu = np.zeros((n_orders, n_dipoles))
        for ik in range(0,n_orders):
            for nn in range(n_dipoles):
                Lpatch = Lpatch_Fulls[ik][nn]
                norm_1 = Lpatch.T @ PsQ @ Lpatch
                norm_2 = Lpatch.T @ Q1 @ Lpatch
                ax=norm_1 @ np.linalg.inv(norm_2)
                [eigenvals,ee] = np.linalg.eig(ax)
                # norm_1 = np.dot(Lpatch.T, np.dot(PsQ, Lpatch))
                # norm_2 = np.dot(Lpatch.T, np.dot(Q1, Lpatch))
                # mu[ik, nn] = np.trace(np.linalg.solve(norm_2, norm_1))
                # mu[ik, nn] = np.trace(np.linalg.solve(norm_2, norm_1))
                mu[ik, nn] = np.max(eigenvals)
                
     
        # #mask
        # for temp00 in range(0,len(BOrder)):      
        #     mu[:, BDipole[temp00]] = 0 # from previous iteration
    
        # Find the dipole/ patch with highest correlation with the residual
        best_order, best_dipole = np.unravel_index(np.argmax(mu), mu.shape)
        best_rank = akx
        S_AP.append([best_rank,best_order, best_dipole])
        
        BOrder.append(best_order)
        BDipole.append(best_dipole)
        BRank.append(best_rank)

        # source_covariance += np.squeeze(self.gradients[best_order][best_dipole] * (1/np.sqrt(i+1)))
        # source_covariance += np.squeeze(self.gradients[best_order][best_dipole])

        if i == 0:
            # B = leadfields[best_order][:, best_dipole][:, np.newaxis]
            B = Lpatch_Fulls[best_order][best_dipole]
        else:
            # B = np.hstack([B, leadfields[best_order][:, best_dipole][:, np.newaxis]])
            # B = np.hstack([B, Lpatch_Fulls[akx][best_order][best_dipole]])
            B = np.hstack([Lpatch_Fulls[order][dipole] for rank,order,dipole in S_AP])        
                                                         
        P_A = B @ np.linalg.pinv(B.T @ B) @ B.T
        Q = np.identity(P_A.shape[0]) - P_A  
        
        C = Q @ Us
        U, D, _= np.linalg.svd(C, full_matrices=False)        
        
        # Truncate eigenvectors
        Us = U[:, :n_comp]  
    return S_AP, BRank, BOrder, BDipole                    
    
# =========================================================================
# Section 4: Helper and Utility Functions
# This section contains various helper functions used by the main algorithms,
# including functions for generating source rank combinations, updating
# estimated locations, and wrapping standard MNE inverse solvers for
# parallel execution.
# =========================================================================

# --- combinations ---
# A hardcoded lookup table that provides different ways to combine patch
# ranks to sum to a total number of source components.

def combinations(n_sources_pat):
    comb = []
    if n_sources_pat == 1:
        comb.append([0, 1, 0, 0, 0, 0, 0])
    elif n_sources_pat == 2:
        # comb.append([0, 2, 0, 0, 0, 0, 0])
        comb.append([0, 0, 1, 0, 0, 0, 0])
    elif n_sources_pat == 3:
        comb.append([0, 3, 0, 0, 0, 0, 0])
        comb.append([0, 1, 1, 0, 0, 0, 0])
        comb.append([0, 0, 0, 1, 0, 0, 0])
    elif n_sources_pat == 4:
        # comb.append([0, 4, 0, 0, 0, 0, 0])
        # comb.append([0, 1, 0, 1, 0, 0, 0])
        # comb.append([0, 2, 1, 0, 0, 0, 0])
        comb.append([0, 0, 2, 0, 0, 0, 0])
        # comb.append([0, 0, 0, 0, 1, 0, 0])
    elif n_sources_pat == 5:
        comb.append([0, 5, 0, 0, 0, 0, 0])
        comb.append([0, 1, 0, 0, 1, 0, 0])
        comb.append([0, 2, 0, 1, 0, 0, 0])
        comb.append([0, 3, 1, 0, 0, 0, 0])
        comb.append([0, 0, 1, 1, 0, 0, 0])
        comb.append([0, 1, 2, 0, 0, 0, 0])
        comb.append([0, 0, 0, 0, 0, 1, 0])
    elif n_sources_pat == 6:
        # comb.append([0, 6, 0, 0, 0, 0, 0])
        # comb.append([0, 1, 0, 0, 0, 1, 0])        
        # comb.append([0, 2, 0, 0, 1, 0, 0])
        # comb.append([0, 3, 0, 1, 0, 0, 0])
        # comb.append([0, 4, 1, 0, 0, 0, 0])
        # comb.append([0, 0, 1, 0, 1, 0, 0])
        # comb.append([0, 1, 1, 1, 0, 0, 0])
        # comb.append([0, 2, 2, 0, 0, 0, 0])
        # comb.append([0, 0, 0, 2, 0, 0, 0])
        # comb.append([0, 0, 0, 0, 0, 0, 1])
        comb.append([0, 0, 3, 0, 0, 0, 0])        
    return comb

# --- UpdatedEst_Loc ---
# A post-processing function that refines an estimated patch location to
# a single, most representative dipole location within that patch.
def UpdatedEst_Loc(EstLoc_method,Weights,leadfields,JustDipoleLoc):
    EstLoc_meth = []
    EstLoc_methodnew = []
    if len(EstLoc_method[0][0]) == 2:
        for i in range(0,len(EstLoc_method)):
            Loc = EstLoc_method[i]
            modified_sublist = [[1] + sublist for sublist in Loc]
            EstLoc_meth.append(modified_sublist)
    else:
        EstLoc_meth = EstLoc_method 
    EstLoc_method = EstLoc_meth               
        
       
    for i in range(0,len(EstLoc_method)):
        Truevloc = [] 
        if JustDipoleLoc == False:
            for j in range(0,len(EstLoc_method[i])):
                dip = EstLoc_method[i][j][2]
                W = Weights[EstLoc_method[i][j][1]]
                Dipoles_inpatch = np.where(W[dip]!=0)[0]
                L = leadfields[:,Dipoles_inpatch]
                #####                
                if EstLoc_method[i][j][1] != 0:
                    Weights1=np.squeeze(W[Dipoles_inpatch,dip])
                    Weights1 = np.diag(Weights1)
                else:
                    Weights1=W[Dipoles_inpatch,dip]
                    Weights1=Weights1.reshape(-1,1)                
                L = leadfields[:,Dipoles_inpatch] @ Weights1
                ####
                U, Sig, vt1 = np.linalg.svd(L, full_matrices=False) 
                L1 = leadfields[:,Dipoles_inpatch]
                waMP = (np.linalg.inv(L1.T@L1) @ L1.T) @ U[:,range(0,EstLoc_method[i][j][0])] 
                v = vt1[range(0,EstLoc_method[i][j][0]),:] # first singu;ar vector
                for kk in range(0,EstLoc_method[i][j][0]):
                    Truevloc.append(Dipoles_inpatch[np.argmax(v[kk])])
                    # Truevloc.append(Dipoles_inpatch[np.argmax(waMP[:,kk])])
        else:
            for j in range(0,len(EstLoc_method[i])):
                dip = EstLoc_method[i][j][2]
                Truevloc.append(dip)                
        
        EstLoc_methodnew.append(Truevloc) 
        
    return EstLoc_methodnew

# --- sparsemethods ---
# A wrapper function to run standard MNE-Python inverse solvers (e.g.,
# MNE, dSPM, sLORETA) on a batch of data in parallel using joblib. This
# is useful for comparing custom algorithms against established methods.   
def sparsemethods(Y,info,fwd_inv,inv_method,prep_leadfield_invm,noise_cov,loose,depth):
    if prep_leadfield_invm == True:
        leadfield = fwd_inv['sol']['data']
        leadfield /= np.linalg.norm(leadfield, axis=0)
        fwd_inv['sol']['data'] = leadfield.copy()
        
    import mne
    from mne.minimum_norm import apply_inverse, make_inverse_operator
    from joblib import Parallel, delayed
    snr = 3.0  # use smaller SNR for raw data
    
    def process_sample(Y_sample, info, fwd_inv, noise_cov, inv_method, snr):
        evoked = mne.EvokedArray(Y_sample, info, verbose=0).set_eeg_reference("average", projection=True, verbose=0).apply_proj()
        evoked._data = Y_sample
        lambda2 = 1.0 / snr**2
        inverse_operator = make_inverse_operator(evoked.info, fwd_inv, noise_cov, depth=None, loose=0, verbose=True)
        stc = apply_inverse(evoked, inverse_operator, lambda2, inv_method, pick_ori=None)
        return stc.data
    
    def run_parallel(Y, info, fwd_inv, noise_cov, inv_method, snr):
        FULLstc = Parallel(n_jobs=-1)(delayed(process_sample)(Y[i], info, fwd_inv, noise_cov, inv_method, snr) for i in range(len(Y)))
        return FULLstc
    
    # Call the function to run in parallel
    FULLstc = run_parallel(Y, info, fwd_inv, noise_cov, inv_method, snr)
    
    return FULLstc   

    
        
    # batch_size = len(Y) 
    # import mne
    # from mne.minimum_norm import apply_inverse, make_inverse_operator
    # FULLstc = []
    # for i in range(0,batch_size):
    #     evoked = mne.EvokedArray(Y[i], info, verbose=0).set_eeg_reference("average", projection=True, verbose=0).apply_proj()
    #     evoked._data=Y[i]
    #     snr = 3.0  # use smaller SNR for raw data
    #     lambda2 = 1.0 / snr**2
    #     inverse_operator = make_inverse_operator(evoked.info, fwd_inv, noise_cov, depth=None, loose=0, verbose=True)
    #     stc = apply_inverse(evoked, inverse_operator, lambda2, inv_method, pick_ori=None)
    #     FULLstc.append(stc.data)
    # return stc,FULLstc             
     
