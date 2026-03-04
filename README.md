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
- Need to fix fit for bounded T2 log least squares, currently produces HUGE values, on order of 1e18
