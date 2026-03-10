from pathlib import Path
import traceback
import time
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.optimize import least_squares


# ============================================================
# User configuration
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\18828\Desktop\研究生打工\biomedical modelling\cw2\relaxometry\cmbi_data\cmbi_data"
)

OUTPUT_ROOT = Path(
    r"C:\Users\18828\Desktop\研究生打工\biomedical modelling\cw2\relaxometry\task2_preterm_mono_outputs"
)

# Set to None to run all subjects found in DATA_ROOT
SUBJECT_IDS = None

# Methods to run
METHODS_TO_RUN = ["weighted_ls", "nnls_grid", "nlls"]

# Numerical settings
EPS = 1e-8
MIN_SIGNAL = 1e-6
T2_MIN_MS = 5.0
T2_MAX_MS = 500.0
WEIGHT_MODE = "signal2"       # for weighted LS
WEIGHT_FLOOR = 0.05

CHUNK_SIZE = 8000

# NNLS mono-exponential grid
NNLS_T2_GRID = np.concatenate([
    np.arange(10, 82, 2, dtype=np.float32),
    np.arange(85, 155, 5, dtype=np.float32),
    np.arange(160, 505, 10, dtype=np.float32),
])

# NLLS settings
NLLS_MAX_NFEV = 100


# ============================================================
# Logging
# ============================================================

def log(msg: str, logfile: Path = None):
    print(msg, flush=True)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ============================================================
# File helpers
# ============================================================

def load_nifti(path: Path):
    nii = nib.load(str(path))
    arr = nii.get_fdata(dtype=np.float32)
    return nii, arr


def save_map(reference_nii: nib.Nifti1Image, array_3d: np.ndarray, out_path: Path):
    out_nii = nib.Nifti1Image(np.asarray(array_3d, dtype=np.float32), reference_nii.affine)
    out_nii.set_data_dtype(np.float32)
    nib.save(out_nii, str(out_path))


def find_te_file():
    candidates = [
        DATA_ROOT / "TEs.txt",
        DATA_ROOT.parent / "TEs.txt",
        DATA_ROOT.parent.parent / "TEs.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find TEs.txt near DATA_ROOT.")


def discover_subject_ids():
    ids = []
    for p in sorted(DATA_ROOT.glob("*-qt2_reg.nii*")):
        ids.append(p.name.split("-qt2_reg")[0])
    if len(ids) == 0:
        for p in sorted(DATA_ROOT.glob("*-qt2.nii*")):
            name = p.name
            if "-qt2_par" in name or "-qt2_seg" in name or "-qt2_fullp" in name:
                continue
            ids.append(name.split("-qt2")[0])
    ids = sorted(list(set(ids)))
    if len(ids) == 0:
        raise RuntimeError("No subjects found in DATA_ROOT.")
    return ids


def find_subject_files(subject_id: str):
    qt2_reg = sorted(DATA_ROOT.glob(f"{subject_id}-qt2_reg.nii*"))
    qt2 = sorted(DATA_ROOT.glob(f"{subject_id}-qt2.nii*"))
    mask1 = sorted(DATA_ROOT.glob(f"{subject_id}-mask1.nii*"))
    mask0 = sorted(DATA_ROOT.glob(f"{subject_id}-mask.nii*"))
    seg1 = sorted(DATA_ROOT.glob(f"{subject_id}-qt2_seg1.nii*"))
    seg0 = sorted(DATA_ROOT.glob(f"{subject_id}-qt2_seg.nii*"))

    img_file = qt2_reg[0] if len(qt2_reg) > 0 else (qt2[0] if len(qt2) > 0 else None)
    mask_file = mask1[0] if len(mask1) > 0 else (mask0[0] if len(mask0) > 0 else None)
    seg_file = seg1[0] if len(seg1) > 0 else (seg0[0] if len(seg0) > 0 else None)

    if img_file is None:
        raise FileNotFoundError(f"Missing qt2_reg/qt2 image for {subject_id}")
    if mask_file is None:
        raise FileNotFoundError(f"Missing mask for {subject_id}")
    if seg_file is None:
        raise FileNotFoundError(f"Missing seg for {subject_id}")

    return {
        "img_file": img_file,
        "mask_file": mask_file,
        "seg_file": seg_file,
    }


# ============================================================
# Shared utilities
# ============================================================

def build_weights(signals: np.ndarray, mode: str = "signal2") -> np.ndarray:
    if mode == "uniform":
        return np.ones_like(signals, dtype=np.float32)

    max_signal = np.max(signals, axis=1, keepdims=True)
    max_signal = np.clip(max_signal, EPS, None)

    ratio = signals / max_signal
    ratio = np.clip(ratio, WEIGHT_FLOOR, None)

    if mode == "signal":
        w = ratio
    elif mode == "signal2":
        w = ratio ** 2
    else:
        raise ValueError(f"Unknown WEIGHT_MODE: {mode}")

    return w.astype(np.float32)


def rebuild_map(mask: np.ndarray, vec: np.ndarray, fill_value=np.nan):
    out = np.full(mask.shape, fill_value, dtype=np.float32)
    out[mask] = vec.astype(np.float32)
    return out


def summarize_case(subject_id, method, num_voxels, num_valid, runtime_sec, t2_vec, s0_vec, nrmse_vec):
    valid_t2 = t2_vec[np.isfinite(t2_vec)]
    valid_t2_in_range = valid_t2[(valid_t2 >= T2_MIN_MS) & (valid_t2 <= T2_MAX_MS)]
    valid_s0 = s0_vec[np.isfinite(s0_vec)]
    valid_nrmse = nrmse_vec[np.isfinite(nrmse_vec)]

    return {
        "subject_id": subject_id,
        "method": method,
        "num_voxels_in_mask": int(num_voxels),
        "num_valid_fit_voxels": int(num_valid),
        "valid_fit_fraction": float(num_valid / max(num_voxels, 1)),
        "runtime_sec": float(runtime_sec),
        "t2_mean_ms_in_range": float(np.mean(valid_t2_in_range)) if valid_t2_in_range.size > 0 else np.nan,
        "t2_median_ms_in_range": float(np.median(valid_t2_in_range)) if valid_t2_in_range.size > 0 else np.nan,
        "t2_std_ms_in_range": float(np.std(valid_t2_in_range)) if valid_t2_in_range.size > 0 else np.nan,
        "fraction_t2_gt_500": float(np.mean(valid_t2 > 500.0)) if valid_t2.size > 0 else np.nan,
        "s0_mean": float(np.mean(valid_s0)) if valid_s0.size > 0 else np.nan,
        "nrmse_mean": float(np.mean(valid_nrmse)) if valid_nrmse.size > 0 else np.nan,
    }


# ============================================================
# Method 1: weighted LS (log-domain)
# ============================================================

def fit_weighted_ls(signals: np.ndarray, tes: np.ndarray):
    N, E = signals.shape
    x = tes[None, :]

    s0 = np.full((N,), np.nan, dtype=np.float32)
    t2 = np.full((N,), np.nan, dtype=np.float32)
    nrmse = np.full((N,), np.nan, dtype=np.float32)
    valid = np.zeros((N,), dtype=np.uint8)

    ok_signal = np.all(np.isfinite(signals), axis=1) & (np.max(signals, axis=1) > MIN_SIGNAL)
    if not np.any(ok_signal):
        return s0, t2, nrmse, valid

    y = signals[ok_signal]
    logy = np.log(np.clip(y, MIN_SIGNAL, None))
    w = build_weights(y, mode=WEIGHT_MODE)

    W = np.sum(w, axis=1)
    xw = np.sum(w * x, axis=1)
    yw = np.sum(w * logy, axis=1)
    x2w = np.sum(w * (x ** 2), axis=1)
    xyw = np.sum(w * x * logy, axis=1)
    den = W * x2w - xw * xw

    a = np.full((y.shape[0],), np.nan, dtype=np.float32)
    b = np.full((y.shape[0],), np.nan, dtype=np.float32)

    ok = den > EPS
    a[ok] = (x2w[ok] * yw[ok] - xw[ok] * xyw[ok]) / den[ok]
    b[ok] = (W[ok] * xyw[ok] - xw[ok] * yw[ok]) / den[ok]

    s0_fit = np.exp(a)
    t2_fit = -1.0 / b

    fit_ok = (
        np.isfinite(s0_fit) & np.isfinite(t2_fit) &
        (s0_fit > 0) & (t2_fit >= T2_MIN_MS) & (t2_fit <= T2_MAX_MS * 4) & (b < 0)
    )

    pred = s0_fit[:, None] * np.exp(-x / np.clip(t2_fit[:, None], EPS, None))
    rss = np.sum((pred - y) ** 2, axis=1)
    denom = np.sum(y ** 2, axis=1)
    nrmse_fit = np.sqrt(rss / np.clip(denom, EPS, None))

    idx = np.where(ok_signal)[0]
    s0[idx[fit_ok]] = s0_fit[fit_ok]
    t2[idx[fit_ok]] = t2_fit[fit_ok]
    nrmse[idx[fit_ok]] = nrmse_fit[fit_ok]
    valid[idx[fit_ok]] = 1

    return s0, t2, nrmse, valid


# ============================================================
# Method 2: mono-exponential NNLS via T2 grid search
# ============================================================

def fit_nnls_grid_chunk(signals: np.ndarray, tes: np.ndarray):
    """
    For each candidate T2, solve nonnegative S0 analytically:
        min || y - S0 * exp(-TE/T2) ||^2  s.t. S0 >= 0
    Then pick the T2 with minimum RSS.
    """
    N, E = signals.shape

    s0 = np.full((N,), np.nan, dtype=np.float32)
    t2 = np.full((N,), np.nan, dtype=np.float32)
    nrmse = np.full((N,), np.nan, dtype=np.float32)
    valid = np.zeros((N,), dtype=np.uint8)

    ok_signal = np.all(np.isfinite(signals), axis=1) & (np.max(signals, axis=1) > MIN_SIGNAL)
    if not np.any(ok_signal):
        return s0, t2, nrmse, valid

    y = signals[ok_signal]                           # [Nv, E]
    B = np.exp(-tes[:, None] / NNLS_T2_GRID[None, :]).astype(np.float32)  # [E, K]

    num = y @ B                                      # [Nv, K]
    den = np.sum(B * B, axis=0)[None, :]             # [1, K]
    s0_all = np.maximum(0.0, num / np.clip(den, EPS, None))

    y_energy = np.sum(y * y, axis=1, keepdims=True)  # [Nv, 1]
    rss = y_energy - 2.0 * s0_all * num + (s0_all ** 2) * den

    best_idx = np.argmin(rss, axis=1)
    best_rss = rss[np.arange(rss.shape[0]), best_idx]
    best_s0 = s0_all[np.arange(s0_all.shape[0]), best_idx]
    best_t2 = NNLS_T2_GRID[best_idx]

    fit_ok = np.isfinite(best_s0) & np.isfinite(best_t2) & (best_s0 > 0)

    nrmse_fit = np.sqrt(best_rss / np.clip(y_energy[:, 0], EPS, None))

    idx = np.where(ok_signal)[0]
    s0[idx[fit_ok]] = best_s0[fit_ok]
    t2[idx[fit_ok]] = best_t2[fit_ok]
    nrmse[idx[fit_ok]] = nrmse_fit[fit_ok]
    valid[idx[fit_ok]] = 1

    return s0, t2, nrmse, valid


def fit_nnls_grid(signals: np.ndarray, tes: np.ndarray):
    N = signals.shape[0]
    s0 = np.full((N,), np.nan, dtype=np.float32)
    t2 = np.full((N,), np.nan, dtype=np.float32)
    nrmse = np.full((N,), np.nan, dtype=np.float32)
    valid = np.zeros((N,), dtype=np.uint8)

    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        chunk_outputs = fit_nnls_grid_chunk(signals[start:end], tes)
        s0[start:end] = chunk_outputs[0]
        t2[start:end] = chunk_outputs[1]
        nrmse[start:end] = chunk_outputs[2]
        valid[start:end] = chunk_outputs[3]

    return s0, t2, nrmse, valid


# ============================================================
# Method 3: non-linear least squares
# ============================================================

def mono_model(params, tes):
    s0, t2 = params
    return s0 * np.exp(-tes / t2)


def mono_residuals(params, tes, y):
    return mono_model(params, tes) - y


def mono_jacobian(params, tes, y):
    s0, t2 = params
    e = np.exp(-tes / t2)
    ds0 = e
    dt2 = s0 * e * (tes / (t2 ** 2))
    return np.stack([ds0, dt2], axis=1)


def fit_nlls(signals: np.ndarray, tes: np.ndarray, init_s0: np.ndarray, init_t2: np.ndarray, logfile: Path = None):
    N = signals.shape[0]
    s0 = np.full((N,), np.nan, dtype=np.float32)
    t2 = np.full((N,), np.nan, dtype=np.float32)
    nrmse = np.full((N,), np.nan, dtype=np.float32)
    valid = np.zeros((N,), dtype=np.uint8)

    ok_signal = np.all(np.isfinite(signals), axis=1) & (np.max(signals, axis=1) > MIN_SIGNAL)
    indices = np.where(ok_signal)[0]

    for count, idx in enumerate(indices, start=1):
        y = signals[idx].astype(np.float64)

        s0_0 = init_s0[idx] if np.isfinite(init_s0[idx]) and init_s0[idx] > 0 else float(np.max(y))
        t2_0 = init_t2[idx] if np.isfinite(init_t2[idx]) and init_t2[idx] > 0 else 60.0

        try:
            res = least_squares(
                mono_residuals,
                x0=np.array([s0_0, t2_0], dtype=np.float64),
                jac=mono_jacobian,
                bounds=([0.0, T2_MIN_MS], [np.inf, T2_MAX_MS]),
                args=(tes.astype(np.float64), y),
                method="trf",
                max_nfev=NLLS_MAX_NFEV,
            )

            if res.success:
                s0_hat, t2_hat = res.x
                pred = mono_model(res.x, tes.astype(np.float64))
                rss = np.sum((pred - y) ** 2)
                denom = np.sum(y ** 2)

                if np.isfinite(s0_hat) and np.isfinite(t2_hat) and s0_hat > 0 and T2_MIN_MS <= t2_hat <= T2_MAX_MS:
                    s0[idx] = float(s0_hat)
                    t2[idx] = float(t2_hat)
                    nrmse[idx] = float(np.sqrt(rss / max(denom, EPS)))
                    valid[idx] = 1

        except Exception:
            pass

        if logfile is not None and count % 5000 == 0:
            log(f"NLLS progress: {count}/{len(indices)} valid-signal voxels processed", logfile)

    return s0, t2, nrmse, valid


# ============================================================
# Main processing
# ============================================================

def process_subject(subject_id: str, tes: np.ndarray, logfile: Path):
    files = find_subject_files(subject_id)

    img_nii, img = load_nifti(files["img_file"])
    _, mask = load_nifti(files["mask_file"])

    mask = mask > 0

    if img.ndim != 4:
        raise ValueError(f"{subject_id}: expected 4D qt2 image, got {img.shape}")
    if img.shape[:3] != mask.shape:
        raise ValueError(f"{subject_id}: image/mask shape mismatch")
    if img.shape[-1] != len(tes):
        raise ValueError(f"{subject_id}: number of echoes {img.shape[-1]} does not match len(TEs)={len(tes)}")

    signals = img[mask]
    out_dir = OUTPUT_ROOT / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n===== {subject_id} =====", logfile)
    log(f"Image file   : {files['img_file']}", logfile)
    log(f"Mask file    : {files['mask_file']}", logfile)
    log(f"Seg file     : {files['seg_file']}", logfile)
    log(f"Image shape  : {img.shape}", logfile)
    log(f"Masked voxels: {signals.shape[0]}", logfile)

    case_rows = []

    weighted_init_s0 = np.full((signals.shape[0],), np.nan, dtype=np.float32)
    weighted_init_t2 = np.full((signals.shape[0],), np.nan, dtype=np.float32)

    for method in METHODS_TO_RUN:
        log(f"\n--- Running {method} ---", logfile)
        start_time = time.time()

        if method == "weighted_ls":
            s0_vec, t2_vec, nrmse_vec, valid_vec = fit_weighted_ls(signals, tes)
            weighted_init_s0 = s0_vec.copy()
            weighted_init_t2 = t2_vec.copy()

        elif method == "nnls_grid":
            s0_vec, t2_vec, nrmse_vec, valid_vec = fit_nnls_grid(signals, tes)

        elif method == "nlls":
            s0_vec, t2_vec, nrmse_vec, valid_vec = fit_nlls(
                signals, tes,
                init_s0=weighted_init_s0,
                init_t2=weighted_init_t2,
                logfile=logfile,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        runtime_sec = time.time() - start_time

        s0_map = rebuild_map(mask, s0_vec)
        t2_map = rebuild_map(mask, t2_vec)
        nrmse_map = rebuild_map(mask, nrmse_vec)
        valid_map = rebuild_map(mask, valid_vec.astype(np.float32), fill_value=0.0)

        save_map(img_nii, s0_map, out_dir / f"{subject_id}_s0_{method}.nii.gz")
        save_map(img_nii, t2_map, out_dir / f"{subject_id}_t2_{method}.nii.gz")
        save_map(img_nii, nrmse_map, out_dir / f"{subject_id}_nrmse_{method}.nii.gz")
        save_map(img_nii, valid_map, out_dir / f"{subject_id}_valid_{method}.nii.gz")

        row = summarize_case(
            subject_id=subject_id,
            method=method,
            num_voxels=signals.shape[0],
            num_valid=int(np.sum(valid_vec)),
            runtime_sec=runtime_sec,
            t2_vec=t2_vec,
            s0_vec=s0_vec,
            nrmse_vec=nrmse_vec,
        )
        case_rows.append(row)

        pd.DataFrame([row]).to_csv(out_dir / f"{subject_id}_{method}_summary.csv", index=False)

        log(f"{method} finished in {runtime_sec:.3f} s", logfile)
        log(f"valid-fit fraction = {row['valid_fit_fraction']:.4f}", logfile)
        log(f"T2 median (in range) = {row['t2_median_ms_in_range']}", logfile)
        log(f"NRMSE mean = {row['nrmse_mean']}", logfile)

    return case_rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logfile = OUTPUT_ROOT / "run_log.txt"
    with open(logfile, "w", encoding="utf-8") as f:
        f.write("task2 preterm mono fit all log\n")

    te_file = find_te_file()
    tes = np.loadtxt(str(te_file), dtype=np.float32)
    if tes.ndim != 1:
        tes = tes.reshape(-1)

    subject_ids = SUBJECT_IDS if SUBJECT_IDS is not None else discover_subject_ids()

    log(f"DATA_ROOT = {DATA_ROOT}", logfile)
    log(f"TE file   = {te_file}", logfile)
    log(f"TEs (ms)  = {tes}", logfile)
    log(f"Subjects  = {subject_ids}", logfile)
    log(f"Methods   = {METHODS_TO_RUN}", logfile)

    all_rows = []

    for sid in subject_ids:
        try:
            all_rows.extend(process_subject(sid, tes, logfile))
        except Exception as e:
            log(f"[ERROR] {sid} failed: {e}", logfile)
            log(traceback.format_exc(), logfile)

    if len(all_rows) > 0:
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(OUTPUT_ROOT / "all_methods_all_subjects_summary.csv", index=False)
        log(f"Saved: {OUTPUT_ROOT / 'all_methods_all_subjects_summary.csv'}", logfile)


if __name__ == "__main__":
    main()