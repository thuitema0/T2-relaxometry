import numpy as np
from functools import partial  # retained if you plan to extend; safe to remove otherwise
import nibabel as nib
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt

def _guard_positive(s):
    return np.maximum(s, 1e-6)

def analytic_two_echo(S, TEs):
    # S: (..., 2)
    S = _guard_positive(S)
    TE1, TE2 = TEs
    S1, S2 = S[...,0], S[...,1]
    ratio = np.clip(S1 / S2, 1e-6, 1e6)
    denom = np.log(ratio)
    # Avoid divide-by-zero when echoes are nearly identical
    denom = np.where(np.abs(denom) < 1e-8, np.sign(denom) * 1e-8 + 1e-12, denom)
    T2 = (TE2 - TE1) / denom
    S0 = S1 * np.exp(TE1 / T2)
    return S0, T2

def log_linear_ls(S, TEs):
    S = _guard_positive(np.asarray(S))
    TEs = np.asarray(TEs)
    y = np.log(S)  # (..., E)

    # Design matrix columns: 1 and -TE
    X1 = -TEs  # (E,)
    w = np.ones_like(y)

    A00 = np.sum(w, axis=-1)
    A01 = np.sum(w * X1, axis=-1)
    A11 = np.sum(w * X1 * X1, axis=-1)
    b0  = np.sum(w * y, axis=-1)
    b1  = np.sum(w * X1 * y, axis=-1)

    det = A00 * A11 - A01 * A01
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)

    lnS0 = (b0 * A11 - b1 * A01) / det
    neg_invT2 = (A00 * b1 - A01 * b0) / det

    T2 = 1 / (-neg_invT2)
    S0 = np.exp(lnS0)
    return S0, T2

def weighted_log_linear_ls(S, TEs, weights=None):
    S = _guard_positive(np.asarray(S))
    TEs = np.asarray(TEs)
    y = np.log(S)
    if weights is None:
        # weight by S^2 to approximate inverse var in log-domain
        weights = S**2
    w = np.maximum(weights, 1e-12)  # ensure positive weights

    X1 = -TEs  # (E,)

    A00 = np.sum(w, axis=-1)
    A01 = np.sum(w * X1, axis=-1)
    A11 = np.sum(w * X1 * X1, axis=-1)
    b0  = np.sum(w * y, axis=-1)
    b1  = np.sum(w * X1 * y, axis=-1)

    det = A00 * A11 - A01 * A01
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)

    lnS0 = (b0 * A11 - b1 * A01) / det
    neg_invT2 = (A00 * b1 - A01 * b0) / det

    T2 = 1 / (-neg_invT2)
    S0 = np.exp(lnS0)
    return S0, T2

def nlls_single_voxel(S, TEs, bounds=([0, 5], [np.inf, 5000])):
    S = _guard_positive(np.asarray(S))
    TEs = np.asarray(TEs)
    def model(te, S0, T2): return S0 * np.exp(-te / T2)
    # initial guess from two echoes if possible
    if len(TEs) >= 2:
        S0g, T2g = analytic_two_echo(S[:2], TEs[:2])
    else:
        S0g, T2g = S.max(), 80.0
    popt, _ = curve_fit(
        model, TEs, S,
        p0=[float(S0g), float(np.clip(T2g, 5, 500))],
        bounds=bounds, maxfev=2000
    )
    return popt  # S0, T2

def nlls_volume(Svol, TEs, mask=None, bounds=([0, 5], [np.inf, 5000])):
    TEs = np.asarray(TEs)
    shp = Svol.shape[:-1]
    S0 = np.zeros(shp); T2 = np.zeros(shp)
    it = np.ndindex(shp)
    for idx in it:
        if mask is not None and not mask[idx]:
            continue
        s = Svol[idx]
        try:
            S0[idx], T2[idx] = nlls_single_voxel(s, TEs, bounds=bounds)
        except Exception:
            S0[idx], T2[idx] = np.nan, np.nan
    return S0, T2

def compare_algorithms(S, TEs):
    algos = {}
    if S.shape[-1] >= 2:
        algos["analytic_2echo"] = analytic_two_echo(S[..., :2], TEs[:2])
    algos["log_ls"] = log_linear_ls(S, TEs)
    algos["wlog_ls"] = weighted_log_linear_ls(S, TEs)
    algos["nlls"] = nlls_volume(S, TEs)
    return algos

# Load data
root = "data"

data = nib.load(f"{root}/case01-qt2_reg.nii.gz").get_fdata()  # (X,Y,Z,E)
TEs  = np.loadtxt(f"{root}/case01-TEs.txt")
mask = nib.load(f"{root}/case01-mask.nii.gz").get_fdata().astype(bool)

# Fast analytic using first two echoes
S0_2, T2_2 = analytic_two_echo(data[..., :2], TEs[:2])

# Log-linear LS (fast)
S0_lls, T2_lls = log_linear_ls(data, TEs)

# Weighted log-linear LS
S0_w, T2_w = weighted_log_linear_ls(data, TEs)

# Non-linear LS (slower; consider per-slice or downsampling)
S0_nl, T2_nl = nlls_volume(data, TEs, mask=mask, bounds=([0, 5], [np.inf, 5000]))

nib.Nifti1Image(T2_w, affine=np.eye(4)).to_filename("case01-T2-wlogls.nii.gz")
nib.Nifti1Image(S0_w, affine=np.eye(4)).to_filename("case01-S0-wlogls.nii.gz")

for f in ["case01-T2-wlogls.nii.gz", "case01-S0-wlogls.nii.gz"]:
    img_nii = nib.load(f)
    arr = img_nii.get_fdata()
    print(f, "shape", arr.shape, "min/mean/max", np.nanmin(arr), np.nanmean(arr), np.nanmax(arr))

    plt.imshow(arr[:, :, arr.shape[2]//2].T, cmap="inferno", origin="lower")
    if "T2" in f:
        # Use percentiles within mask (if available) for a sensible display range
        if mask is not None:
            vmin, vmax = np.percentile(arr[mask], [1, 99])
        else:
            vmin, vmax = np.percentile(arr[np.isfinite(arr)], [1, 99])
        plt.clim(vmin, vmax)
    plt.title(f)
    plt.colorbar()
    plt.show()
    