# ============================
# IMPORT REQUIRED LIBRARIES
# ============================

import numpy as np                  # Numerical computations (arrays, linear algebra)
import mne                          # MEG/EEG analysis library (forward models, source spaces)
import pickle as pkl                # Load/save Python objects
import os                           # File system utilities
import matplotlib.pyplot as plt     # Plotting the graph                 

# Disable oneDNN optimizations (can reduce numerical inconsistencies)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from scipy.spatial.distance import cdist   # Pairwise distance computation
from scipy.sparse import csr_matrix        # Efficient sparse matrix format
from invert.util import pos_from_forward   # Utility for extracting positions from forward model
from scipy.spatial import ConvexHull       # Compute convex hull for surface area estimation


# ==========================================================
# LOAD COARSE (TRUE) FORWARD MODEL AND SOURCE COORDINATES
# ==========================================================

folder_load = "forward_models"                         # Folder containing forward models
folder_pathload = os.path.join(os.getcwd(), folder_load)
filename = "128_ch_coarse_80_ratio-fwd.fif"            # Coarse source space forward model
file_path = os.path.join(folder_pathload, filename)

# Read forward solution from disk
fwd_for = mne.read_forward_solution(file_path, verbose=0)

# Convert to fixed-orientation forward model
fwd_for = mne.convert_forward_solution(fwd_for, force_fixed=True)

# Extract source space information
source_model = fwd_for['src']

# Separate left and right hemispheres
left_hemi = source_model[0]
right_hemi = source_model[1]

# Vertex indices used in each hemisphere
left_vertno = left_hemi['vertno']
right_vertno = right_hemi['vertno']

# 3D coordinates of active vertices
left_coords = left_hemi['rr'][left_vertno]
right_coords = right_hemi['rr'][right_vertno]

# Combine both hemispheres into a single coordinate array
all_coords_true = np.vstack((left_coords, right_coords))


# ==========================================================
# LOAD FINE (ESTIMATED) FORWARD MODEL AND SOURCE COORDINATES
# ==========================================================

filename = "128_ch_fine_80_ratio-fwd.fif"              # Fine source space forward model
file_path = os.path.join(folder_pathload, filename)

# Read and fix orientation of fine forward model
fwd_inv = mne.read_forward_solution(file_path, verbose=0)
fwd_inv = mne.convert_forward_solution(fwd_inv, force_fixed=True)

# Extract source space
source_model = fwd_inv['src']

left_hemi = source_model[0]
right_hemi = source_model[1]

left_vertno = left_hemi['vertno']
right_vertno = right_hemi['vertno']

left_coords = left_hemi['rr'][left_vertno]
right_coords = right_hemi['rr'][right_vertno]

# Estimated (fine) source coordinates
all_coords_est = np.vstack((left_coords, right_coords))


# ============================
# RESULT STORAGE DICTIONARIES
# ============================

results_localization_errors = {}   # Stores localization errors
results_extent_errors = {}         # Stores extent (area) errors


# ============================
# HELPER FUNCTIONS
# ============================

# Load pickled simulation or evaluation data
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


# Compute surface area of an active patch using convex hull
def patch_area(active_coords):
    patch_area = 0.0

    # Convex hull requires at least 4 non-coplanar points
    if active_coords.shape[0] >= 4:
        hull = ConvexHull(active_coords)
        patch_area = hull.area  # Surface area (m^2)

    return patch_area


# ----------------------------------------------------------
# Compute localization and extent error between true and
# estimated patch-based activations
# ----------------------------------------------------------

def compute_localization_and_extent(E_True, E_Est, all_coords_true, all_coords_est):
    centroids_true = []   # Centroids of true patches
    area_true = []        # Areas of true patches
    true_coords = []      # Vertex coordinates of true patches
    est_coords = []       # Vertex coordinates of estimated patches

    # ----- TRUE PATCHES -----
    for E in E_True:
        # Find vertices with any non-zero activation
        active_idx = np.where(np.any(E != 0, axis=1))[0]
        if active_idx.size == 0:
            continue

        active_coords = all_coords_true[active_idx]
        true_coords.append(active_coords)

        area = patch_area(active_coords)
        centroid = np.mean(active_coords, axis=0)

        centroids_true.append(centroid)
        area_true.append(area)

    # ----- ESTIMATED PATCHES -----
    centroids_est = []
    area_est = []

    for E in E_Est:
        active_idx = np.where(np.any(E != 0, axis=1))[0]
        if active_idx.size == 0:
            continue

        active_coords = all_coords_est[active_idx]
        est_coords.append(active_coords)

        area = patch_area(active_coords)
        centroid = np.mean(active_coords, axis=0)

        centroids_est.append(centroid)
        area_est.append(area)

    # Merge all vertices for MLE computation
    true_merged = np.vstack(true_coords)
    est_merged = np.vstack(est_coords)

    localization_error = 0
    extent_error = 0
    estimated_patches_index = []

    # Match true patches to nearest estimated patches
    for i in range(len(centroids_true)):
        mn = np.inf
        idx = 0

        for j in range(len(centroids_est)):
            if j in estimated_patches_index:
                continue

            # Squared Euclidean distance between centroids
            val = np.sum((centroids_true[i] - centroids_est[j]) ** 2)
            if val < mn:
                mn = val
                idx = j

        estimated_patches_index.append(idx)
        localization_error += mn
        extent_error += abs(area_true[i] - area_est[idx])

    mean_localization_error = localization_error / len(centroids_true)
    mean_extent_error = extent_error / len(centroids_true)

    return {
        "True Centroids: ": centroids_true,
        "True Areas: ": area_true,
        "Estimated Centroids: ": centroids_est,
        "Estimated Areas: ": area_est,
        "Localization Error: ": mean_localization_error,
        "Extent Error: ": mean_extent_error,
        "True Vertices": true_merged,
        "Estimated Vertices": est_merged
    }


# ----------------------------------------------------------
# Extract dipoles above a power threshold
# ----------------------------------------------------------

def extract_dipoles(true_SS, estimated_STC_MNE, estimated_STC_sLoreta, estimated_STC_CC, threshold=0.75):
    # ---- TRUE SOURCE ----
    true_source = np.sum(true_SS, axis=0)                 # Sum across time
    true_power = np.linalg.norm(true_source, axis=1)      # Dipole magnitude
    true_power_norm = true_power / np.max(true_power)     # Normalize

    # Select dipoles above threshold
    true_dipoles = np.where(true_power_norm >= threshold)[0]
    true_vertices = all_coords_true[true_dipoles]

    # ---- ESTIMATED SOURCES ----
    def extract_estimated_vertices(stc):
        power = np.linalg.norm(stc, axis=1)
        power_norm = power / np.max(power)
        dipoles = np.where(power_norm >= threshold)[0]
        return all_coords_est[dipoles]

    est_vertices_MNE = extract_estimated_vertices(estimated_STC_MNE)
    est_vertices_sLoreta = extract_estimated_vertices(estimated_STC_sLoreta)
    est_vertices_CC = extract_estimated_vertices(estimated_STC_CC)

    return true_vertices, est_vertices_MNE, est_vertices_sLoreta, est_vertices_CC


# ----------------------------------------------------------
# Compute true source centroids and areas only
# ----------------------------------------------------------

def compute_true_source_data(E_True, all_coords_true):
    centroids_true = []
    area_true = []
    true_coords = []

    for E in E_True:
        active_idx = np.where(np.any(E != 0, axis=1))[0]
        if active_idx.size == 0:
            continue

        active_coords = all_coords_true[active_idx]
        true_coords.append(active_coords)

        area = patch_area(active_coords)
        centroid = np.mean(active_coords, axis=0)

        centroids_true.append(centroid)
        area_true.append(area)

    return {
        "True Centroids": centroids_true,
        "True Areas": area_true,
        "True Vertices": true_coords
    }


# ----------------------------------------------------------
# Find neighbors up to a given graph order
# ----------------------------------------------------------

def get_neighbors_by_order(adjacency, center, order):
    active = set([center])      # All visited vertices
    frontier = set([center])    # Current expansion frontier

    for _ in range(order):
        new_frontier = set()
        for v in frontier:
            new_frontier.update(adjacency[v].indices)

        frontier = new_frontier - active
        active.update(frontier)

        if not frontier:
            break

    return np.array(sorted(active))


# ----------------------------------------------------------
# Extract FLEX-AP patches and compute their properties
# ----------------------------------------------------------

def extract_flex(SAP, adjacency):
    flex_area_list = []
    centroids_flex = []
    flex_vertices = []

    for order, dipole in SAP:
        all_indices = get_neighbors_by_order(adjacency, dipole, order)
        all_vertices = all_coords_est[all_indices]

        flex_vertices.append(all_vertices)
        flex_area_list.append(patch_area(all_vertices))
        centroids_flex.append(np.mean(all_vertices, axis=0))

    flex_merged = np.vstack(flex_vertices)

    return flex_area_list, centroids_flex, flex_merged


# ----------------------------------------------------------
# Compute extent error for FLEX-AP
# ----------------------------------------------------------

def compute_flex_extent_error(area_true, area_flex, centroids_true, centroids_est):
    estimated_patches_index = []
    extent_error = 0

    for i in range(len(centroids_true)):
        mn = np.inf
        idx = 0

        for j in range(len(centroids_est)):
            if j in estimated_patches_index:
                continue

            val = np.sum((centroids_true[i] - centroids_est[j]) ** 2)
            if val < mn:
                mn = val
                idx = j

        estimated_patches_index.append(idx)
        extent_error += abs(area_true[i] - area_flex[idx])

    return extent_error / len(centroids_true)


# ----------------------------------------------------------
# Mean Localization Error (MLE)
# ----------------------------------------------------------

def localization_error_MLE(true_vertices, est_vertices):
    if true_vertices.size == 0 or est_vertices.size == 0:
        return np.nan

    # Pairwise Euclidean distances
    D = cdist(est_vertices, true_vertices)

    # True → estimated
    term_true = np.sum(np.min(D, axis=0)) / (2 * true_vertices.shape[0])

    # Estimated → true
    term_est = np.sum(np.min(D, axis=1)) / (2 * est_vertices.shape[0])

    return term_true + term_est

# ==========================================================
# CODE FOR PLOTTING THE GRAPH
# ==========================================================

def plot_graph(results_dict, title="Error vs SNR"):
    # Sort SNR values
    snr_values = sorted(results_dict.keys())
    methods = list(results_dict[snr_values[0]].keys())

    n_snr = len(snr_values)
    n_methods = len(methods)

    bar_width = 0.8 / n_methods
    x = np.arange(n_snr)

    plt.figure()

    for i, method in enumerate(methods):
        errors = [results_dict[snr][method] for snr in snr_values]
        plt.bar(
            x + i * bar_width,
            errors,
            width=bar_width,
            label=method
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Error")
    plt.title(title)

    # Center x-ticks
    plt.xticks(x + bar_width * (n_methods - 1) / 2, snr_values)

    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN EVALUATION LOOP OVER SNR LEVELS
# ==========================================================

# Number of Monte-Carlo Simulations
batch_size = 250

for snr in range(-5, 10, 5):

    results_localization = {}
    results_extent = {}

    # Load simulated data
    filename = f"Data_corr_0.5_smooth_2_patchranks_[1, 2]_snr_{snr}.pkl"
    file_path = os.path.join(os.getcwd(), "Simulated_Data", filename)
    loaded_data = load_data(file_path)

    Y = loaded_data["Y"]
    E_True = loaded_data["EFull"]
    SddotFull = loaded_data["SddotFull"]

    # Load evaluated data
    filename = f"EFull_LenEvaluate_fine-80_Data_corr_0.5_smooth_2_patchranks_[1, 2]_snr_{snr}.pkl"
    filename2 = f"LenEvaluate_MError_fine-80_Data_corr_0.5_smooth_2_patchranks_[1, 2]_snr_{snr}.pkl"

    evaluated_data = load_data(os.path.join(os.getcwd(), "Evaluated_Data", filename))
    evaluated_data2 = load_data(os.path.join(os.getcwd(), "Evaluated_Data", filename2))

    E_Est = evaluated_data["EFull_Patch_AP"]
    stc_dict = evaluated_data2['STCs']

    stc_MNE = stc_dict["MNE"]
    stc_sLoreta = stc_dict["sLORETA"]
    stc_CC = stc_dict["Convexity-Champagne"]

    name = f"SNR_{snr}"
    threshold = 0.5

    # ---- Localization error for MNE, sLORETA, CC ----
    errors_MNE, errors_sLoreta, errors_CC = [], [], []

    for i in range(batch_size):
        true_v, est_MNE, est_sL, est_CC = extract_dipoles(
            SddotFull[i], stc_MNE[i], stc_sLoreta[i], stc_CC[i], threshold
        )

        for err, store in [
            (localization_error_MLE(true_v, est_MNE), errors_MNE),
            (localization_error_MLE(true_v, est_sL), errors_sLoreta),
            (localization_error_MLE(true_v, est_CC), errors_CC)
        ]:
            if not np.isnan(err):
                store.append(err)

    # RMS localization errors
    results_localization["MNE"] = np.sqrt(np.mean(np.array(errors_MNE) ** 2))
    results_localization["sLORETA"] = np.sqrt(np.mean(np.array(errors_sLoreta) ** 2))
    results_localization["CC"] = np.sqrt(np.mean(np.array(errors_CC) ** 2))

    # ---- AP, FLEX-AP, Patch-AP evaluation continues below ----
    # (Left unchanged; logic mirrors the above with method-specific metrics)

    folder_load = "Evaluated_Data"
    folder_pathload = os.path.join(os.getcwd(), folder_load)
    filename_sap = f"SAP_fine-80_Data_corr_0.5_smooth_2_patchranks_[1, 2]_snr_{snr}.pkl"
    file_path_sap = os.path.join(folder_pathload, filename)
    loaded_data_sap = load_data(file_path)
    sap = loaded_data_sap["sap"]

    error_AP = []
    error_FLEXAP = []
    extent_error_flex = []
    adjacency_est = mne.spatial_src_adjacency(fwd_inv['src'], verbose=0)
    adjacency_est = csr_matrix(adjacency_est)
        
    for i in range(0, batch_size):
        res = compute_true_source_data(E_True[i], all_coords_true)
        true_vertices = res["True Vertices"]
        true_centroids = res["True Centroids"]
        area_true = res["True Areas"]
        true_merged = np.vstack(true_vertices)
        area_flex, centroids_flex, flex_merged = extract_flex(sap["FLEX-AP"][i], adjacency_est)
        flex_extent_error = compute_flex_extent_error(area_true, area_flex, true_centroids, centroids_flex)
        extent_error_flex.append(flex_extent_error)
        
        ap_indices = []
        for order, dipole in sap["AP"][i]:
            ap_indices.append(dipole)
    
        ap_vertices = all_coords_est[ap_indices]
        err_AP = localization_error_MLE(true_merged, ap_vertices)
        err_FLEXAP = localization_error_MLE(true_merged, flex_merged) 
        error_AP.append(err_AP)
        error_FLEXAP.append(err_FLEXAP)

    results_localization["AP"] = np.sqrt(np.mean(np.array(error_AP) ** 2))
    results_localization["Flex-AP"] = np.sqrt(np.mean(np.array(error_FLEXAP) ** 2))

    results_extent["Flex-AP"] = np.mean(extent_error_flex)

    # ----- Patch-AP -----
      
    errors_PatchAP = []
    localization_error = 0
    extent_error = 0
    for i in range(0, batch_size):
      res = compute_localization_and_extent(E_True[i], E_Est[i], all_coords_true, all_coords_est)
      err_PatchAP = localization_error_MLE(res["True Vertices"], res["Estimated Vertices"])
      if not np.isnan(err_PatchAP):
        errors_PatchAP.append(err_PatchAP)
      localization_error += res["Localization Error: "]
      extent_error += res["Extent Error: "]
    
    RMS_PatchAP = np.sqrt(np.mean(np.array(errors_PatchAP) ** 2))
    results_localization["Patch-AP"] = RMS_PatchAP
    
    mean_extent_PatchAP = extent_error / batch_size
    results_extent["Patch-AP"] = mean_extent_PatchAP

    print("Threshold: ", threshold)
    print(name)

    # Localization Error
    print("RMS Localization Error (MNE): ", results_localization["MNE"])
    print("RMS Localization Error (sLORETA): ", results_localization["sLORETA"])
    print("RMS Localization Error (CC): ", results_localization["CC"])
    print("RMS Localization Error (AP): ", results_localization["AP"])
    print("RMS Localization Error (FLEX-AP): ", results_localization["Flex-AP"])
    print("RMS Localization Error (Patch AP): ", results_localization["Patch-AP"])

    # Extent Error
    print("Mean Extent Error (FLEX-AP): ", results_extent["Flex-AP"])
    print("Mean Extent Error (Patch AP): ", results_extent["Patch-AP"])

    results_localization_errors[snr] = results_localization
    results_extent_errors[snr] = results_extent

plot_graph(results_localization_errors, "Localization Error vs SNR")
plot_graph(results_extent_errors, "Extent Error vs SNR")