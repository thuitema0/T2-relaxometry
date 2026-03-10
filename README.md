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

* **`task2_preterm_mono_fit_all.py` (Parameter Fitting):** The core computational script. It reads multi-echo MRI NIfTI data and performs voxel-wise mono-exponential T2 fitting. It implements three distinct fitting methods for comparison: Weighted Least Squares (`weighted_ls`), Non-Negative Least Squares grid search (`nnls_grid`), and Non-Linear Least Squares (`nlls`). It outputs 3D parameter maps for T2, S0, and NRMSE.

* **`task2_preterm_mono_evaluate.py` (Quantitative Evaluation):** This script handles tissue-level statistical analysis. It applies anatomical segmentation masks (CSF, GM, WM) to the generated parameter maps and calculates summary statistics (e.g., mean, median, standard deviation) for the T2 and NRMSE values. Results are aggregated and exported as CSV reports.

* **`task2_preterm_mono_visualize_brain_slices.py` (QC Visualization):** An automated quality control plotting tool. It selects representative axial slices across the brain mask (e.g., 30%, 50%, 70% depth) and generates side-by-side comparisons containing the anatomical background, the grayscale parameter map, and a color-coded overlay. Outputs are saved as high-resolution PNG figures.
