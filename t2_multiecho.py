import numpy as np
import nibabel as nib
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import time

'''
This script implements fitting methods for T2 estimation from multi-echo data. It includes:
- Analytic two-echo method (using any pair of echoes)
- Linear least squares in log-domain (all echoes)
- Weighted linear least squares (log-domain, with S^2 weights)
- Bounded linear least squares (enforcing non-negativity of S0 and T2)
- Non-linear least squares (signal domain, bounded)

TO DO:
- Fix fit for bounded T2 log least squares, currently produces HUGE values, on order of 1e18
'''

# helper to ensure positivity for log-domain methods
def _guard_positive(s, eps=1e-6):
    return np.maximum(s, eps)

# Linear LS (log-domain)
def log_linear_ls(S, TEs, mask=None):
    S = _guard_positive(S)
    y = np.log(S) # (..., E)
    X1 = -TEs # (E,)
    w = np.ones_like(y)

    A00 = np.sum(w, axis=-1)
    A01 = np.sum(w * X1, axis=-1)
    A11 = np.sum(w * X1 * X1, axis=-1)
    b0  = np.sum(w * y, axis=-1)
    b1  = np.sum(w * X1 * y, axis=-1)

    det = np.where(np.abs(A00*A11 - A01*A01) < 1e-12, 1e-12, A00*A11 - A01*A01)
    lnS0 = (b0 * A11 - b1 * A01) / det
    slope = (A00 * b1 - A01 * b0) / det # negative inverse T2

    T2 = 1.0 / slope 
    S0 = np.exp(lnS0)

    if mask is not None:
        S0_out = np.full(mask.shape, np.nan); T2_out = np.full(mask.shape, np.nan)
        S0_out[mask] = S0[mask]; T2_out[mask] = T2[mask]
        return S0_out, T2_out
    return S0, T2

# Weighted LS (log-domain)
def weighted_log_linear_ls(S, TEs, weights=None, mask=None):
    S = _guard_positive(S)
    y = np.log(S)
    if weights is None:
        weights = S**2
    w = np.maximum(weights, 1e-12)
    X1 = -TEs
    A00 = np.sum(w, axis=-1)
    A01 = np.sum(w * X1, axis=-1)
    A11 = np.sum(w * X1 * X1, axis=-1)
    b0  = np.sum(w * y, axis=-1)
    b1  = np.sum(w * X1 * y, axis=-1)
    det = np.where(np.abs(A00*A11 - A01*A01) < 1e-12, 1e-12, A00*A11 - A01*A01)
    lnS0 = (b0 * A11 - b1 * A01) / det
    slope = (A00 * b1 - A01 * b0) / det    
    T2 = 1.0 / slope                        
    S0 = np.exp(lnS0)

    if mask is not None:
        S0_out = np.full(mask.shape, np.nan); T2_out = np.full(mask.shape, np.nan)
        S0_out[mask] = S0[mask]; T2_out[mask] = T2[mask]
        return S0_out, T2_out
    return S0, T2

# Bounded linear LS (“NNLS-like” for single compartment)
def bounded_log_ls(S, TEs, mask=None, T2_min=5.0, T2_max=5000.0, eps=1e-6):
    """
    Bounded log-domain LS via per-voxel constrained least squares.
    Model: log S = lnS0 + (-TE) * invT2, where invT2 = 1/T2 >= 0
    Bounds enforce T2 in [T2_min, T2_max].
    """
    S = np.maximum(S, eps)
    y = np.log(S)  # (..., E)
    # Design matrix: [1, -TE]
    X = np.vstack([np.ones_like(TEs), -TEs]).T  # (E,2)
    shp = y.shape[:-1]
    lnS0 = np.full(shp, np.nan)
    invT2 = np.full(shp, np.nan)
    invT2_min = 1.0 / T2_max   # lower bound (close to 0 but not 0)
    invT2_max = 1.0 / T2_min   # upper bound
    it = np.ndindex(shp)
    for idx in it:
        if mask is not None and not mask[idx]:
            continue
        yi = y[idx]
        def res(beta):
            return X @ beta - yi
        # Initial guess:
        # lnS0 ~ max(logS), invT2 ~ 1/100 ms^-1 (T2 ~100 ms)
        x0 = np.array([float(yi.max()), 1.0 / 100.0])
        sol = least_squares(
            res,
            x0=x0,
            bounds=(
                [-np.inf, invT2_min],
                [ np.inf, invT2_max]
            ),
            max_nfev=200
        )
        lnS0[idx], invT2[idx] = sol.x
    T2 = 1.0 / invT2
    S0 = np.exp(lnS0)
    return S0, T2
    
# Non-linear LS (signal domain, bounded)
def nlls_single_voxel(S, TEs, bounds=([0, 5], [np.inf, 5000])):
    S = _guard_positive(S)
    def model(te, S0, T2): return S0 * np.exp(-te / T2)
    if len(TEs) >= 2:
        TE1, TE2 = TEs[:2]
        S1, S2 = S[0], S[1]
        ratio = np.clip(S1/S2, 1e-6, 1e6)
        denom = np.log(ratio)
        denom = np.sign(denom)*1e-8 if abs(denom)<1e-8 else denom
        T2g = (TE2-TE1)/denom
        S0g = S1 * np.exp(TE1 / T2g)
    else:
        S0g, T2g = S.max(), 80.0
    popt, _ = curve_fit(model, TEs, S,
                        p0=[float(S0g), float(np.clip(T2g, 5, 500))],
                        bounds=bounds, maxfev=2000)
    return popt # S0, T2

def nlls_volume(Svol, TEs, mask=None, bounds=([0, 5], [np.inf, 5000])):
    shp = Svol.shape[:-1]
    S0 = np.full(shp, np.nan); T2 = np.full(shp, np.nan)
    it = np.ndindex(shp)
    for idx in it:
        if mask is not None and not mask[idx]:
            continue
        try:
            S0[idx], T2[idx] = nlls_single_voxel(Svol[idx], TEs, bounds=bounds)
        except Exception:
            S0[idx], T2[idx] = np.nan, np.nan
    return S0, T2

# ---------------------------------------------------------------------

root = "data"
img = nib.load(f"{root}/case01-qt2_reg.nii.gz")
data = img.get_fdata() # (X,Y,Z,E)
affine = img.affine
TEs  = np.loadtxt(f"{root}/case01-TEs.txt")
mask = nib.load(f"{root}/case01-mask.nii.gz").get_fdata().astype(bool)

# restrict the data upfront to mask for LS/WLS speed (still produce full-size outputs)
data_masked = data.copy()
data_masked[~mask] = 0  # harmless because we ignore non-mask voxels in output

start = time.time()
S0_ls,  T2_ls  = log_linear_ls(data_masked, TEs, mask=mask)
print(f"Log-linear LS took {time.time() - start:.2f} seconds.")

start = time.time()
S0_wls, T2_wls = weighted_log_linear_ls(data_masked, TEs, mask=mask)
print(f"Weighted log-linear LS took {time.time() - start:.2f} seconds.")

start = time.time()
S0_bls, T2_bls = bounded_log_ls(data_masked, TEs, mask=mask)
print(f"Bounded log-linear LS took {time.time() - start:.2f} seconds.")

start = time.time()
S0_nl,  T2_nl  = nlls_volume(data, TEs, mask=mask)  # uses mask internally
print(f"Non-linear LS took {time.time() - start:.2f} seconds.")

# Save maps
nib.Nifti1Image(T2_ls,  affine).to_filename("case01-T2-ls.nii.gz")
nib.Nifti1Image(T2_wls, affine).to_filename("case01-T2-wls.nii.gz")
nib.Nifti1Image(T2_bls, affine).to_filename("case01-T2-boundedls.nii.gz")
nib.Nifti1Image(T2_nl,  affine).to_filename("case01-T2-nlls.nii.gz")

nib.Nifti1Image(S0_ls,  affine).to_filename("case01-S0-ls.nii.gz")
nib.Nifti1Image(S0_wls, affine).to_filename("case01-S0-wls.nii.gz")
nib.Nifti1Image(S0_bls, affine).to_filename("case01-S0-boundedls.nii.gz")
nib.Nifti1Image(S0_nl,  affine).to_filename("case01-S0-nlls.nii.gz")

# Visualization with mask-based scaling
def show_map(arr, title, mask, cmap="viridis", slice_idx=None):
    if slice_idx is None:
        slice_idx = arr.shape[2]//2
    vals = arr[mask & np.isfinite(arr)]
    vmin, vmax = np.percentile(vals, [5, 95]) if vals.size else (0,1)
    plt.imshow(arr[:, :, slice_idx].T, cmap=cmap, origin="lower")
    plt.clim(vmin, vmax)
    plt.title(title); plt.colorbar()

plt.figure(figsize=(12,10))
plt.subplot(2,2,1); show_map(T2_ls, "T2 Log-Least Squares", mask)
plt.subplot(2,2,2); show_map(T2_wls, "T2 Weighted Log-Least Squares", mask)
plt.subplot(2,2,3); show_map(T2_bls, "T2 Bounded Log-Least Squares", mask)
plt.subplot(2,2,4); show_map(T2_nl, "T2 Non-Linear Least Squares", mask)
plt.tight_layout(); plt.show()

plt.figure(figsize=(12,10))
plt.subplot(2,2,1); show_map(S0_ls, "S0 Log-Least Squares", mask)
plt.subplot(2,2,2); show_map(S0_wls, "S0 Weighted Log-Least Squares", mask)
plt.subplot(2,2,3); show_map(S0_bls, "S0 Bounded Log-Least Squares", mask)
plt.subplot(2,2,4); show_map(S0_nl, "S0 Non-Linear Least Squares", mask)
plt.tight_layout(); plt.show()
