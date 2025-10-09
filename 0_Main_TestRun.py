# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 05:32:08 2024

@author: amita
"""

import sys; sys.path.insert(0, r'/home/amita/Patch_AP/1stversion/invert')
import mne
import pickle as pkl

from time import time
# from copy import deepcopy
import os
import ot
import numpy as np
# from Simulate_rank2patch_WEIGHTING import rank2generator
# from Simulate_rank2patch_WEIGHTING import Smoothgenerator_Smoothed
# from TimeCourses import TimeCourse
from invert import Solver
from invert.util import pos_from_forward
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import PATCH_APFunction
from PATCH_APFunction import J_Estimated
from PATCH_APFunction import sparsemethods
import time
# import LErrorEst
# from scipy.stats import wasserstein_distance
# import multiprocessing
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def eval_emd(distances, values_1, values_2):
    values_1 = abs(values_1).mean(axis=-1)
    values_2 = abs(values_2).mean(axis=-1)
    
    # values_1 = abs(values_1.mean(axis=-1))
    # values_2 = abs(values_2.mean(axis=-1))
    
    values_1 = values_1 / np.sum(values_1)
    values_2 = values_2 / np.sum(values_2)    
    
    emd_value = ot.emd2(values_1, values_2, distances)
    return emd_value

def eval_emdnew(distances, values_1, values_2):  
    emd_value1 = []
    for aki in range(len(values_1[0])):        
        values_11 = abs(values_1[:,aki]) / np.sum(abs(values_1[:,aki]))
        values_22 = abs(values_2[:,aki]) / np.sum(abs(values_2[:,aki]))            
        emd_value1.append(ot.emd2(values_11, values_22, distances))
    emd_value = np.median(emd_value1)
    return emd_value


#%% 
# ==============================
# Section 1: Load Simulated Data
# ==============================
# Specify the path to the directory containing the files
file_path = "forward_models"
mult = 2

# Read the forward solution from the specified file
fwd_for = mne.read_forward_solution(os.path.join(file_path, "128_ch_coarse_80_ratio-fwd.fif"), verbose=0)
fwd_for = mne.convert_forward_solution(fwd_for, force_fixed=True)
pos_for = pos_from_forward(fwd_for)

# Create the full file path for the "128_ch_info.pkl" file
fn = os.path.join(file_path, "128_ch_info.pkl")
with open(fn, 'rb') as f:
    info = pkl.load(f)

# Define a function to save the data
def save_data(data_dict, folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pkl.dump(data_dict, f)

# Define the folder path
folder_save = "Evaluated_Data"
folder_pathsave = os.path.join(os.getcwd(), folder_save)
os.makedirs(folder_pathsave, exist_ok=True)  # Create the folder if it doesn't exist

#%%    
# ======================================================================================
# Section 5: Finer forward model for inverse source localization. (Step to avoid Inverse Crime)
# ======================================================================================
model_paths = {
    # "coarse-80": r"128_ch_coarse_80_ratio-fwd.fif", # 5124 sasmpling points
    "fine-80": r"128_ch_fine_80_ratio-fwd.fif", # 8196 sasmpling points
    # "fine-50": r"128_ch_fine_50_ratio-fwd.fif",
    # "fine-20": r"128_ch_fine_20_ratio-fwd.fif"
    }
fwds = dict()
for inv_name, model_path in model_paths.items():
    fwd_inv = mne.read_forward_solution(os.path.join(file_path,model_path), verbose=0)
    fwd_inv = mne.convert_forward_solution(fwd_inv, surf_ori=True, force_fixed=True,use_cps=True, verbose=0) 
    fwds[inv_name] = fwd_inv
    pos_inv = pos_from_forward(fwd_inv)
distances = cdist(pos_for, pos_inv)

# %%    
# ==============================================
# Section : Load 
# ============================================== 
leadfields = fwd_inv['sol']['data']              
n_dipoles = np.shape(leadfields)[1]    

Weights = pkl.load(open(r'/home/amita/Patch_AP/2ndversion/KQ_MaxExtent_{}_{}.pkl'.format(10, inv_name), 'rb'))
Lpatch_Fulls = pkl.load(open(r'/home/amita/Patch_AP/2ndversion/ULpatch_MaxExtent_{}_MaxRank_{}_{}.pkl'.format(10,5, inv_name), 'rb'))  
Lpatch_Fulls = Lpatch_Fulls[mult]

# %%    
# ==============================================
# Section 4: Load the Simulated data
# ==============================================               
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

# Define the folder path
folder_load = "Simulated_Data"
folder_pathload = os.path.join(os.getcwd(), folder_load)

# %%    
# ==============================================
# Section : Main Code
# ============================================== 
batch_size = 1 # number of nonte-carlo repettions 
plotmax = 1
# Patchranks_Full  = [[1],[2],[3],[1,1],[1,2],[1,3],[2,2],[2,3],[1,2,3]]   
# Patchranks_Full  = [[1],[2],[1,1],[1,2],[2,2]]    
Patchranks_Full  = [[1,2]]  
# Patchranks_Full  = [[1,1]]#,[1,1]]     
for corr_coeff in [0.5]:
    for Smoothness_order in range(2,4,2):        
        for Patchranks in Patchranks_Full: 
            snr_values = []                     
            for snr_db in range(0,5,5):
                start_time = time.time()
                # n_sources = np.sum(Patchranks)
                n_sources = len(Patchranks)
                n_sourcespatch = len(Patchranks) * mult
                # n_sourcespatch = n_sources
                n_orders = 8
                max_iter = n_sources+6
                diffusion_parameter = 0.1
                n_reg_params = 10
                n_jobs = 15
                # Iterate through all files in the folder
                filename = f"Data_corr_{corr_coeff}_smooth_{Smoothness_order}_patchranks_{Patchranks}_snr_{snr_db}.pkl"
                file_path = os.path.join(folder_pathload, filename)
                loaded_data = load_data(file_path)
                Y = loaded_data["Y"]
                SSFull = loaded_data["SddotFull"]
                SS = [sum(sublist) for sublist in SSFull]
                Y = Y[:batch_size]
                sim_info = loaded_data["sim_info"]
                sim_info.loc[:, 'n_sources'] = n_sources
                n_timepoints = len(Y[0,0,:]) 
                # %%    
                # ==============================================
                # Section 4: Plot Simulated brain activity
                # ==============================================
                pp = dict(surface='inflated', hemi='both', background="white", verbose=0, colorbar=False, time_viewer=False)
                # pp["colorbar"] = True
                # pp['clim'] = dict(kind='value', lims=[-1, -0.001, 1])
                pos_for = pos_from_forward(fwd_for)
                source_model = fwd_for['src']
                vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
                distances = cdist(pos_for, pos_inv)
                argsorted_distance_matrix = np.argsort(distances, axis=1)
                subject = "fsaverage"
                # batch_size = 10
                if batch_size<plotmax:
                    for i in range(0,batch_size):   
                        subject = "fsaverage"        
                    
                    
                        tmin = 0
                        tstep = 1/1000  
                        stc = mne.SourceEstimate(SS[i], vertices, tmin=tmin, tstep=tstep, 
                                                    subject=subject, verbose=0)
                            
                        stc_ = stc.copy()
                        stc_.data =SS[i] #abs(stc_.data / np.max(stc_.data, axis=0))
                            
                        brain = stc_.plot(
                            hemi="both",
                            views=["ven"],
                            brain_kwargs=dict(title="Simulated Source Activity"),
                            colorbar=True,
                            cortex="low_contrast",
                            background="white",
                        )
            
            
                
                
                #%% 
                # ==============================
                # Section 6: source localization
                # ==============================       
                prep_leadfield = False
                prep_leadfield_CC = True
                prep_leadfield_invm = False
                stop_crit = 0
                solver_dicts = [
                    {
                        "solver_name": "RAP-MUSIC",
                        "display_name": "RAP-MUSIC",
                        "prep_leadfield": prep_leadfield,            
                        "make_args": {
                            "n": n_sources, 
                            "k": n_sources,
                            "n_orders": 0,
                            "refine_solution": False,
                            "stop_crit": 0.,
                        },
                        "apply_args": {
                            
                        },
                        "recompute_make": True
                    },
                    {
                        "solver_name": "FLEX-MUSIC",
                        "display_name": "FLEX-MUSIC",
                        "prep_leadfield": prep_leadfield,        
                        "make_args": {
                            "n": n_sources, 
                            "k": n_sources,
                            "n_orders": n_orders,
                            "refine_solution": False,
                            "stop_crit": 0.,
                            "diffusion_parameter": diffusion_parameter
                        },
                        "apply_args": {
                            
                        },
                        "recompute_make": True
                    },
                    {
                        "solver_name": "AP",
                        "display_name": "AP",
                        "prep_leadfield": prep_leadfield,
                        "make_args": {
                            "n": n_sources, 
                            "k": n_sources,
                            "n_orders": 0,
                            "refine_solution": True,
                            "stop_crit": 0.,
                            "max_iter": 6
                        },
                        "apply_args": {
                            
                        },
                        "recompute_make": True
                    },
                    {
                        "solver_name": "FLEX-AP",
                        "display_name": "FLEX-AP",
                        "prep_leadfield": prep_leadfield,
                        "make_args": {
                            "n": n_sources, 
                            "k": n_sources,
                            "n_orders": n_orders,
                            "refine_solution": True,
                            "stop_crit": 0.,
                            "diffusion_parameter": diffusion_parameter,
                            "max_iter": 6
                        },
                        "apply_args": {
                            
                        },
                        "recompute_make": True
                    },  
                    # {
                    #     "solver_name": "Convexity-Champagne",
                    #     "display_name": "Convexity-Champagne",
                    #     "prep_leadfield": prep_leadfield_CC,
                    #     "make_args": { 
                    #     },
                    #     "apply_args": {
                            
                    #     },
                    #     "recompute_make": True
                    # },                    
                ]
                
                # %%    
                # ==============================================
                # Section : Source Localization
                # ============================================== 
                leadfields = fwd_inv['sol']['data']
                n_dipoles = np.shape(leadfields)[1]                            
                
                Bestorder_PatchAP, Bestorder_PatchRAP = [], []
                EstLoc_PatchAP, EstLoc_PatchRAP = [], [] 
                BRank_PatchAP, BRank_PatchRAP = [], []
                
                from funs_AG import predict_sources_parallel3
                x_test = Y
                Fullstcs = []  
                stcs = dict()    
                for solver_dict in solver_dicts:
                    solver_dict["solver_name"] = Solver(solver_dict["display_name"], n_reg_params=n_reg_params)
                    
                res = predict_sources_parallel3(solver_dicts, fwd_inv, info, x_test[:], sim_info, n_jobs=n_jobs)
                # Organize/ Store Inverse solutions and simulations
                solver_names = [sd["display_name"] for sd in solver_dicts]
                stc_dict = {sd["display_name"]: [] for sd in solver_dicts}
                proc_time_make = {sd["display_name"]: [] for sd in solver_dicts}
                proc_time_apply = {sd["display_name"]: [] for sd in solver_dicts}
                
                for sample in res:
                    for solver, stc in sample[0].items():
                        stc_dict[solver].append(stc.toarray())
                    for solver, t in sample[1].items():
                        proc_time_make[solver].append(t)
                    for solver, t in sample[2].items():
                        proc_time_apply[solver].append(t)
                
                # %%    
                # ==============================================
                # Section : Source Localization "MNE"  # sLORETA, MNE, dSPM
                # ==============================================         
                from mne.minimum_norm import apply_inverse, make_inverse_operator
                inv_method = ["MNE", "sLORETA", "dSPM"]
                for solver in inv_method:
                    print(solver)
                    noise_cov = mne.make_ad_hoc_cov(info, std=None, verbose=None)
                    stc = sparsemethods(Y,info,fwd_inv,solver,prep_leadfield_invm,noise_cov,loose=0,depth=0)
                    stc_dict[solver]=stc                                
                
                ###################################################################################
                from joblib import Parallel, delayed

                def process_Patch(i, Y, n_sourcespatch, Weights, max_iter, Lpatch_Fulls, n_orders, diffusion_parameter, n_dipoles, mult, leadfields):
                    swapweight, BRank_PatchAP, Bestorder_PatchAP, BDipole, SAP_Grand = PATCH_APFunction.weighted_ap(Y[i], n_sourcespatch, Weights, max_iter, Lpatch_Fulls, n_orders, diffusion_parameter, n_dipoles, refine_solution=True, covariance_type="AP")                                                
                    EstLoc_PatchAP = [inner_list[2] for inner_list in swapweight]
                                   
                    swapweight, BRank_PatchRAP, Bestorder_PatchRAP, BDipole = PATCH_APFunction.PatchRAP(Y[i], n_sourcespatch, n_sourcespatch, n_dipoles, Lpatch_Fulls, n_orders, mult)                                     
                    EstLoc_PatchRAP = [inner_list[2] for inner_list in swapweight]  
                    
                    data_dict = {
                        'EstLoc_PatchRAP': EstLoc_PatchRAP,
                        'EstLoc_PatchAP': EstLoc_PatchAP,
                        'BRank_PatchRAP': BRank_PatchRAP,
                        'BRank_PatchAP': BRank_PatchAP,
                        'Bestorder_PatchRAP': Bestorder_PatchRAP,
                        'Bestorder_PatchAP': Bestorder_PatchAP
                    }
                    return data_dict
                
                if __name__ == '__main__':
                    inputs_Patch = [(i, Y, n_sourcespatch, Weights, max_iter, Lpatch_Fulls, n_orders, diffusion_parameter, n_dipoles, mult, leadfields) for i in range(batch_size)]
                    Data_Patch = Parallel(n_jobs=n_jobs, backend="loky")(delayed(process_Patch)(*input_patch) for input_patch in inputs_Patch)
                    
                    Bestorder_PatchAP = [data["Bestorder_PatchAP"] for data in Data_Patch]
                    BRank_PatchAP = [data["BRank_PatchAP"] for data in Data_Patch]
                    EstLoc_PatchAP = [data["EstLoc_PatchAP"] for data in Data_Patch]
                    
                    Bestorder_PatchRAP = [data["Bestorder_PatchRAP"] for data in Data_Patch]
                    BRank_PatchRAP = [data["BRank_PatchRAP"] for data in Data_Patch]
                    EstLoc_PatchRAP = [data["EstLoc_PatchRAP"] for data in Data_Patch]                                
                
                ##########################
                # Make copies of the arrays before passing them to J_Estimated
                Bestorder_PatchAP_copy = Bestorder_PatchAP.copy()
                BRank_PatchAP_copy = BRank_PatchAP.copy()
                EstLoc_PatchAP_copy = EstLoc_PatchAP.copy()
                # Make copies of the arrays before passing them to J_Estimated
                Bestorder_PatchRAP_copy = Bestorder_PatchRAP.copy()
                BRank_PatchRAP_copy = BRank_PatchRAP.copy()
                EstLoc_PatchRAP_copy = EstLoc_PatchRAP.copy()
                
                J_pred_Patch_AP = J_Estimated(batch_size,n_dipoles,n_timepoints,Y,n_sourcespatch,Bestorder_PatchAP_copy,BRank_PatchAP_copy,EstLoc_PatchAP_copy,Weights,leadfields,Lpatch_Fulls,mult)
                J_pred_Patch_RAP = J_Estimated(batch_size,n_dipoles,n_timepoints,Y,n_sourcespatch,Bestorder_PatchRAP_copy,BRank_PatchRAP_copy,EstLoc_PatchRAP_copy,Weights,leadfields,Lpatch_Fulls,mult)
                Patch_method = ["PATCH RAP","PATCH AP"]
                stc_dict["PATCH RAP"] = J_pred_Patch_RAP
                stc_dict["PATCH AP"] = J_pred_Patch_AP
                solver_names = solver_names + inv_method + Patch_method 
                                            
                #%% 
                #==============================
                #Section 8: Plot Brain Figure
                #==============================                 
                if batch_size<plotmax:
                    for i in range(0,batch_size): 
                        solver_names_new = []
                        axes_left = []
                        for i_row, solver_name in enumerate(solver_names):
                            stc.data = stc_dict[solver_name][0].copy()
                            brain = stc.plot(
                                hemi="both",
                                # subjects_dir=subjects_dir,
                                # initial_time= Time_of_interest, #Time_of_interest,
                                views=["ven"],
                                # clim=solver_dict["clim"],
                                brain_kwargs=dict(title=f"{solver_name}"),
                                colorbar=True,
                                # time_viewer=False,
                                cortex="low_contrast",
                                background="white",
                            )
                                
                #%% 
                #==============================
                #Section 8: calculate EMD
                #============================== 
                def calculate_emd(i, SS, solver_names, stc_dict, J_pred_Patch_RAP, J_pred_Patch_AP, distances):
                    EMD_results = {}
                    J_true = SS[i]
                    for solver_name in solver_names:
                        J_Pred = stc_dict[solver_name][i]
                        emd_result = eval_emd(distances, J_true, J_Pred)
                        evaluated_emd_name = f"{solver_name}"
                        EMD_results[evaluated_emd_name] = emd_result
                
                    J_Pred = J_pred_Patch_RAP[i]
                    emd_result = eval_emd(distances, J_true, J_Pred)
                    EMD_results["PATCH RAP"] = emd_result
                
                    J_Pred = J_pred_Patch_AP[i]
                    emd_result = eval_emd(distances, J_true, J_Pred)
                    EMD_results["PATCH AP"] = emd_result
                
                    return EMD_results
                
                if __name__ == '__main__':
                    EMD_results_list = Parallel(n_jobs=n_jobs, backend="loky")(delayed(calculate_emd)(i, SS, solver_names, stc_dict, J_pred_Patch_RAP, J_pred_Patch_AP, distances) for i in range(batch_size))
                    
                    EMD_results = {}
                    for result_dict in EMD_results_list:
                        for key, value in result_dict.items():
                            if key not in EMD_results:
                                EMD_results[key] = []
                            EMD_results[key].append(value)
                    
                    # Example of accessing the EMD results
                    for i, result in enumerate(EMD_results_list):
                        print(f"Sample {i}")
                    
                    EMD_medians = {key: np.median(values) for key, values in EMD_results.items()}                
                    EMD_stderr = {key: np.std(values) / np.sqrt(len(values)) for key, values in EMD_results.items()}
                    snr_values.append(snr_db)
          
            
                # Extract keys and values from medians and std_devs dictionaries
                method_names = list(EMD_medians.keys())
                medians_values = list(EMD_medians.values())
                std_devs_values = list( EMD_stderr.values())
                
                # Predefined order of method names
                desired_order = ['dSPM', 'MNE', 'Convexity-Champagne','sLORETA','RAP-MUSIC','FLEX-MUSIC','PATCH RAP','AP','FLEX-AP',
                                 'PATCH AP']
                # Reorder the method names if they are available, otherwise ignore
                reordered_method_names, reordered_medians_values,reordered_std_devs_values = [],[],[]
                for method in desired_order:
                    if method in method_names:
                        index = method_names.index(method)
                        reordered_method_names.append(method)
                        reordered_medians_values.append(medians_values[index])
                        reordered_std_devs_values.append(std_devs_values[index])
                method_names = reordered_method_names
                medians_values = reordered_medians_values
                std_devs_values = reordered_std_devs_values
                
            
                # Set the width of the bars
                bar_width = 0.35
                fwd_name = inv_name
                # Plotting
                fig, ax = plt.subplots(figsize=(18, 11))
            
                index = np.arange(len(snr_values))
                # Plot bars for medians
                for ii, method_name in enumerate(method_names):
                    ax.bar(index + ii * bar_width, medians_values[ii], bar_width, yerr=std_devs_values[ii], capsize=5, label=method_name)
            
                ax.set_xlabel('SNR dB', fontsize=24) 
                ax.set_ylabel('Earth Movers Distance', fontsize=24) 
                if fwd_name == "coarse-80":
                    ax.set_title(f'No Model Error (Iteration = {batch_size}), Smoothing = {Smoothness_order}, Patchranks = {Patchranks}', fontsize=24)
                elif fwd_name == "fine-80":
                    ax.set_title(f'Source Space Model Error (Iteration = {batch_size}), Smoothing = {Smoothness_order}, Patchranks = {Patchranks}', fontsize=24)
                elif  fwd_name == "fine-50":
                    ax.set_title(f'Source Space + Conductivity Model Error (Iteration = {batch_size}), Smoothing = {Smoothness_order}, Patchranks = {Patchranks}', fontsize=24) 
            
                ax.set_xticks(index + bar_width / 2)
                ax.set_xticklabels(snr_values)
                # Adjust legend font size
                ax.legend(fontsize=20)
                ax.yaxis.grid(True)
            
                # Adjust tick label font size
                ax.tick_params(axis='both', which='major', labelsize=20)
                # plt.ylim((-0.5,30))
            
                plt.show()
                
                end_time = time.time()  # Record end time
                elapsed_time = end_time - start_time
                print(f"Elapsed time for corr_coeff={corr_coeff}, Smoothness_order={Smoothness_order}, Patchranks={Patchranks}, snr_db={snr_db}: {elapsed_time} seconds")
    
            
            
            
                        
                        
            
            
            
            