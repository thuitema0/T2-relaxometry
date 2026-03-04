import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

'''
This script is for exploring the multi-echo data and understanding the signal decay curves.
It includes functions to visualize the echo images, plot signal decay curves for specific 
voxels and ROIs, and analyze monotonicity of the signal across echoes. 

TO DO:
- Use the specific roi files:
    case01-seg.nii.gz = a multi-class brain segmentation for the T2 image data.
    case01-par.nii.gz = a multi-class brain parcellation for the T2 image data.
    case01-par_lobe.nii.gz = a simplified multi-class brain parcellation for the T2 image data.
  to plot ROI-specific decay curves and compare across regions.
'''

# Helper functions
def load_case(root, case="case01"):
    '''
    Loads the multi-echo data, mask, and echo times for a given case.

    Parameters:
    - root: directory containing the data files
    - case: identifier for the case (e.g., "case01")

    Returns:
    - data: 4D array (X,Y,Z,E) of echo images
    - mask: 3D boolean array indicating voxels to analyze
    - TEs: 1D array of echo times corresponding to the last dimension of data
    '''
    data = nib.load(Path(root)/f"{case}-qt2_reg.nii.gz").get_fdata()  # (X,Y,Z,E)
    mask = nib.load(Path(root)/f"{case}-mask.nii.gz").get_fdata().astype(bool)
    TEs  = np.loadtxt(Path(root)/f"{case}-TEs.txt")  # in ms
    order = np.argsort(TEs)
    return data[..., order], mask, np.array(TEs)[order]

def show_echo_montage(data, TEs, z=40, cmap="gray"):
    '''
    Displays a montage of echo images for a given slice z.

    Parameters:
    - data: 4D array (X,Y,Z,E) of echo images
    - TEs: 1D array of echo times corresponding to the last dimension of data
    - z: slice index to display
    - cmap: colormap for display
    '''
    nE = data.shape[-1]
    cols = min(nE, 6)
    rows = int(np.ceil(nE/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(nE):
        ax = axes[i]
        ax.imshow(data[:,:,z,i], cmap=cmap)
        ax.set_title(f"TE {TEs[i]:.1f} ms")
        ax.axis("off")
    for j in range(nE, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

def plot_voxel_curves(data, TEs, voxels):
    '''
    Plots the signal decay curves for specified voxel coordinates across echo times.

    Parameters:
    - data: 4D array (X,Y,Z,E) of echo images
    - TEs: 1D array of echo times corresponding to the last dimension of data
    - voxels: list of (x,y,z) tuples specifying voxel coordinates to plot
    '''
    plt.figure(figsize=(5,4))
    for (x,y,z) in voxels:
        plt.plot(TEs, data[x,y,z,:], marker="o", label=f"({x},{y},{z})")
    plt.xlabel("TE (ms)")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid()
    plt.title("Voxel signal decay curves")
    plt.show()

def plot_roi_curve(data, TEs, mask):
    '''
    Plots the mean signal decay curve for a region of interest (ROI) defined by a mask.

    Parameters:
    - data: 4D array (X,Y,Z,E) of echo images
    - TEs: 1D array of echo times corresponding to the last dimension of data
    - mask: 3D boolean array indicating voxels to include in the ROI
    '''
    roi = data[mask]
    mean = roi.mean(axis=0)
    std  = roi.std(axis=0)
    plt.figure(figsize=(5,4))
    plt.errorbar(TEs, mean, yerr=std, marker="o", capsize=3)
    plt.xlabel("TE (ms)")
    plt.ylabel("Signal (ROI mean ± SD)")
    plt.grid()
    plt.title("ROI signal decay curve")
    plt.show()

def monotonic_violations(data, mask):
    '''
    Counts the fraction of voxels where any later echo is higher than an earlier one.

    Parameters:
    - data: 4D array (X,Y,Z,E) of echo images
    - mask: 3D boolean array indicating voxels to analyze

    Returns:
    - fraction of non-monotonic voxels
    '''
    sig = data[mask]  # (Nvox, E)
    diffs = np.diff(sig, axis=1)
    viol = (diffs > 0).any(axis=1)
    return viol.mean()  # fraction

def full_analysis(root, case):
    '''
    Helper function to load a case, visualize the data, and analyze monotonicity.

    Parameters:
    - root: directory containing the data files
    - case: identifier for the case (e.g., "case01")
    '''
    data, mask, TEs = load_case(root, case)
    show_echo_montage(data, TEs, z=data.shape[2]//2)
    plot_voxel_curves(data, TEs, voxels=[(60,60,40), (70,70,40)])
    plot_roi_curve(data, TEs, mask)
    frac_viol = monotonic_violations(data, mask)
    print(f"Fraction non-monotonic voxels: {frac_viol*100:.2f}%")




# Load and explore a case
full_analysis("data", "case01")