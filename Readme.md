# PATCH-AP-Solution — README

Project: PATCH-AP (Patch-based source localization) 

## Overview
This repository contains notebooks and modules to simulate MEG data, run patch-based and standard inverse solvers, and evaluate results (EMD metric). Main notebooks:
- 1_MEG_Data_Simulation.ipynb — simulate MEG data
- 2_Evaluate.ipynb — run solvers and save evaluated results
- 3_Predict.ipynb — compute EMD / analyze results
Other helper modules: PATCH_APFunction.py, SimFunction.py, TimeCourses.py, funs_AG.py, invert/ (package)

## Requirements
Minimum: Python 3.8+ (tested on 3.8–3.10)  
For MNE: 0.24.1 compatible with Python 3.8
Main packages (install with pip):

Simpler combined pip line:
```powershell
pip install mne numpy scipy joblib pandas matplotlib POT tensorflow
```

Notes:
- `pickle`, `os`, `time`, `sys`, `pprint`, `copy` are in Python stdlib.
- If `ot` import fails, install POT via `pip install POT` (it exposes `import ot`).
- Some packages (mne, tensorflow) have platform-specific binary requirements; use conda if pip fails.

## How to run notebooks
- Open notebook in Jupyter or VS Code and run cells top → bottom.
- Ensure working directory is the `Python Scripts` folder (the notebooks rely on relative paths).

## File / data locations
- forward_models/ — contains forward solutions and info (e.g. `128_ch_coarse_80_ratio-fwd.fif`, `128_ch_info.fif`)
- Simulated_Data/ — pickled simulated data files created by notebook 1_MEG_Data_Simulation.ipynb
- Evaluated_Data/ — results saved by notebook 2_Evaluate.ipynb
- Results/ / Manuscript_Figures/ — outputs and figures

## Common issues & fixes
- FileNotFoundError when reading forward files:
  - Confirm current working directory is `Python Scripts`.
  - Confirm `forward_models/128_ch_coarse_80_ratio-fwd.fif` exists.
  - Use absolute path if needed.
- ImportError for local modules (PATCH_APFunction, funs_AG, invert):
  - Ensure `sys.path` contains the `Python Scripts` folder before imports:
    ```python
    import sys, os
    sys.path.insert(0, os.getcwd())
    ```
  - Or run notebooks from the `Python Scripts` directory.
- `import ot` error:
  - Install POT: `pip install POT` (exposes `import ot`).

## Re-running / Batch parameters
- `batch_size` controls number of Monte Carlo repetitions processed at once.
- `mult` is used to scale patches/sources and index precomputed patch data (Lpatch_Fulls[mult]). Adjust consistently across notebooks.

## Executing evaluation or prediction scripts
- Run 1_MEG_Data_Simulation first to generate simulated data.
- Run 2_Evaluate to produce evaluated inverse solutions (saved to Evaluated_Data).
- Run 3_Predict to compute EMD and final metrics (reads from Simulated_Data and Evaluated_Data).
