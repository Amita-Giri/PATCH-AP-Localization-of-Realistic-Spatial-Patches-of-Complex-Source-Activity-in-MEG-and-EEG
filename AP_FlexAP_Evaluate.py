import os
current_directory = os.getcwd()
import sys; sys.path.insert(0, current_directory)

import pprint
import mne
import pickle as pkl
import time
import os
import numpy as np
from invert import Solver
from invert.util import pos_from_forward
from scipy.spatial.distance import cdist
import PATCH_APFunction
from PATCH_APFunction import J_Estimated
from PATCH_APFunction import sparsemethods
from scipy.sparse import csr_matrix


file_path = "forward_models"
# file_path = "forward_models"
mult = 2
folder_path = os.path.join(file_path, "128_ch_coarse_80_ratio-fwd.fif")

fwd_for = mne.read_forward_solution(folder_path, verbose=0)
fwd_for = mne.convert_forward_solution(fwd_for, force_fixed=True)
pos_for = pos_from_forward(fwd_for)

fn = os.path.join(file_path, "128_ch_info.fif")
info = mne.io.read_info(fn)

def save_data(data_dict, folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pkl.dump(data_dict, f)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

folder_save = "Evaluated_Data"
folder_pathsave = os.path.join(os.getcwd(), folder_save)
os.makedirs(folder_pathsave, exist_ok=True)

model_paths = {
#    "coarse-80": r"128_ch_coarse_80_ratio-fwd.fif", # 5124 sasmpling points
     "fine-80": r"128_ch_fine_80_ratio-fwd.fif", # 8196 sasmpling points
    # "fine-50": r"128_ch_fine_50_ratio-fwd.fif",
    # "fine-20": r"128_ch_fine_20_ratio-fwd.fif"
    }
fwds = dict()
for inv_name, model_path in model_paths.items():
    fwd_inv = mne.read_forward_solution(os.path.join(file_path,model_path), verbose=0)
    fwd_inv = mne.convert_forward_solution(fwd_inv, force_fixed=True)
    fwds[inv_name] = fwd_inv
    pos_inv = pos_from_forward(fwd_inv)
distances = cdist(pos_for, pos_inv)

inv_name = "fine-80"

leadfields = fwd_inv['sol']['data']
n_dipoles = np.shape(leadfields)[1]

file_path = ""

Weights = pkl.load(open(os.path.join(file_path, "KQ_MaxExtent_{}_{}.pkl".format(10, inv_name)), 'rb'))
Lpatch_Fulls = pkl.load(open(os.path.join(file_path, "ULpatch_MaxExtent_{}_MaxRank_{}_{}.pkl".format(10,5, inv_name)), 'rb'))  
Lpatch_Fulls = Lpatch_Fulls[mult]

folder_load = "Simulated_Data"
folder_path = "Simulated_Data"
folder_pathload = os.path.join(os.getcwd(), folder_load)

batch_size = 250 # number of monte-carlo repettions 
plotmax = 2
Patchranks_Full = [[1,2]]    

for corr_coeff in [0.5]:
   for Smoothness_order in range(2,4,2):        
       for Patchranks in Patchranks_Full: 
           snr_values = []                      
           for snr_db in range(-5,10,5):
               start_time = time.time()
               # n_sources = np.sum(Patchranks)
               n_sources = len(Patchranks)
               n_sourcespatch = len(Patchranks) * mult
               # n_sourcespatch = n_sources
               n_orders = 8
               max_iter = n_sources+6
               diffusion_parameter = 0.1
               n_reg_params = 10
               n_jobs = 5
               # Iterate through all files in the folder
               filename = f"Data_corr_{corr_coeff}_smooth_{Smoothness_order}_patchranks_{Patchranks}_snr_{snr_db}.pkl"
               file_path = os.path.join(folder_pathload, filename)
               loaded_data = load_data(file_path)
               Y = loaded_data["Y"]
               SddotFull = loaded_data["SddotFull"]
               Y = Y[:batch_size]
               sim_info = loaded_data["sim_info"]
               sim_info.loc[:, 'n_sources'] = n_sources
               n_timepoints = len(Y[0,0,:]) 
               # =====================================================
               # Subsection 5.1: Plot Ground Truth Source Activity
               # (Optional) For small batches, this part visualizes the
               # true simulated source activity (ground truth) to
               # provide a visual reference for the localization results.
               # =====================================================
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
                       stc = mne.SourceEstimate(sum(SddotFull[i]), vertices, tmin=tmin, tstep=tstep, 
                                                   subject=subject, verbose=0)
                           
                       stc_ = stc.copy()
                       stc_.data = sum(SddotFull[i]) #abs(stc_.data / np.max(stc_.data, axis=0))
                           
                       brain = stc_.plot(
                           hemi="both",
                           views=["ven"],
                           brain_kwargs=dict(title="Simulated Source Activity"),
                           colorbar=True,
                           cortex="low_contrast",
                           background="white",
                       )
           
               
               
               # =====================================================
               # Subsection 5.2: Configure Inverse Solvers
               # This block sets up the configuration for several
               # source localization algorithms (RAP-MUSIC, FLEX-MUSIC,
               # Champagne, etc.) by defining their parameters in a
               # list of dictionaries.
               # =====================================================     
               prep_leadfield = False
               prep_leadfield_CC = True
               prep_leadfield_invm = False
               stop_crit = 0
               solver_dicts = [
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
                   }                  
               ]
               
               # =====================================================
               # Subsection 5.3: Run Subspace-Based Inverse Solvers
               # This part executes the inverse solvers defined in the
               # `solver_dicts` list. It uses a parallel processing
               # function (`predict_sources_parallel3`) to efficiently
               # compute the source estimates for the entire batch.
               # =====================================================
               leadfields = fwd_inv['sol']['data']
               n_dipoles = np.shape(leadfields)[1]                            
               
               from funs_AG import predict_sources_parallel_AP_FlexAP
               x_test = Y
               Fullstcs = []  
               stcs = dict()    
               for solver_dict in solver_dicts:
                   solver_dict["solver_name"] = Solver(solver_dict["display_name"], n_reg_params=n_reg_params)
                   # solver_dict["solver_name"] = Solver(solver_dict["display_name"], n_reg_params=n_reg_params)
                   
               sap = predict_sources_parallel_AP_FlexAP(solver_dicts, fwd_inv, info, x_test[:], sim_info, n_jobs=n_jobs)
               
               data_dict = {}
               data_dict["sap"] = sap
               
               folder_save = "Evaluated_Data"
               folder_pathsave = os.path.join(os.getcwd(), folder_save)
               os.makedirs(folder_pathsave, exist_ok=True)
               
               save_data(data_dict, folder_pathsave, f"SAP_{inv_name}_Data_corr_{corr_coeff}_smooth_{Smoothness_order}_patchranks_{Patchranks}_snr_{snr_db}.pkl")