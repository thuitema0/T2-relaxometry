# T2-relaxometry
Repository for T2-relaxometry project

## t2_explore.py
- Addresses core task 1
- TO DO: Include functionality for specific regions of interest, rather than whole brain, can use the segmentation files to do this:
  - case01-seg.nii.gz = a multi-class brain segmentation for the T2 image data.
  - case01-par.nii.gz = a multi-class brain parcellation for the T2 image data.
  - case01-par_lobe.nii.gz = a simplified multi-class brain parcellation for the T2 image data.

## t2_analytical.py
- Addresses first part of core task 2
- Analytical solution

## t2_multiecho.py
- Addresses second part of core task 2
- Fit for S0 and T2
- Need to fix fit for bounded T2 log least squares, currently produces HUGE values, on order of 1e18 (√ Fixed)


## Mono-exponential T2 Relaxometry Pipeline

Requirements: 
numpy
pandas
nibabel
scipy
matplotlib

* **`task2_preterm_mono_fit_all.py` (Parameter Fitting):** The core computational script. It reads multi-echo MRI NIfTI data and performs voxel-wise mono-exponential T2 fitting. It implements four distinct fitting methods for comparison: Least Square (ls), Weighted Least Squares (`weighted_ls`), Non-Negative Least Squares grid search (`nnls_grid`), and Non-Linear Least Squares (`nlls`). It outputs 3D parameter maps for T2, S0, and NRMSE. P.S. NRMSE quantifies the quality of fitting, lower NRMSE indicates higher fitting quality.

**To run: **

python task2_preterm_mono_fit_all.py --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT> --methods ls weighted_ls nnls_grid nlls

* **`task2_preterm_mono_evaluate.py` (Quantitative Evaluation):** This script handles tissue-level statistical analysis. It applies anatomical segmentation masks (CSF, GM, WM) to the generated parameter maps and calculates summary statistics (e.g., mean, median, standard deviation) for the T2 and NRMSE values. Results are aggregated and exported as CSV reports. Main metrics are valid-fit-fraction and runtime, as all four algoithms have similar median T2, NRMSE values.

**To run:**

python task2_preterm_mono_evaluate.py --data-root <DATA_ROOT> --fit-root <OUTPUT_ROOT>

* **`task2_preterm_mono_visualize_brain_slices.py` (QC Visualization):** An automated quality control plotting tool for voxel-wise fitting performance. It selects representative axial slices across the brain mask (e.g., 30%, 50%, 70% depth) and generates side-by-side comparisons containing the anatomical background, the grayscale parameter map, and a color-coded overlaying on the anatomical map. To observe T2 and NRMSE the map, smoother region indicates better fitting performance, while hollow or unstable-looking regions suggest poor fitting or unreliable parameter estimation in those voxels.

**To run:**

python task2_preterm_mono_visualize_brain_slices.py --data-root <DATA_ROOT> --fit-root <OUTPUT_ROOT>
