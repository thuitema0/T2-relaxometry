from __future__ import annotations

import gzip
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover
    nib = None


DEFAULT_THRESHOLDS: dict[str, float] = {
    "mono_rel_inc_thresh": 0.05,
    "early_echo_fraction": 0.5,
    "r2_poor_thresh": 0.95,
    "echo_z_thresh": 2.5,
    "max_voxels_for_r2": 5000,
    "eps": 1e-6,
}

_NIFTI_DTYPE_MAP: dict[int, np.dtype] = {
    2: np.dtype(np.uint8),
    4: np.dtype(np.int16),
    8: np.dtype(np.int32),
    16: np.dtype(np.float32),
    64: np.dtype(np.float64),
    256: np.dtype(np.int8),
    512: np.dtype(np.uint16),
    768: np.dtype(np.uint32),
}


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    case_dir: Path
    qt2_reg: Path
    tes: Path | None
    mask: Path
    seg: Path | None = None
    par: Path | None = None
    par_lobe: Path | None = None


def _case_sort_key(case_id: str) -> tuple[int, str]:
    digits = ''.join(ch for ch in case_id if ch.isdigit())
    if digits:
        return (int(digits), case_id)
    return (10**9, case_id)


def _find_first_existing(case_dir: Path, filenames: list[str]) -> Path | None:
    for name in filenames:
        path = case_dir / name
        if path.exists():
            return path
    return None


def _find_tes_path(case_dir: Path, root_dir: Path | None = None, case_id: str | None = None) -> Path | None:
    case_names = [f"{case_id}-TEs.txt", f"{case_id}-tes.txt"] if case_id else []
    case_tes = _find_first_existing(case_dir, case_names)
    if case_tes is not None:
        return case_tes

    # Support shared TE files (e.g., cmbi_data/TEs.txt for all Epicure cases).
    search_dirs: list[Path] = [case_dir]
    if root_dir is not None:
        search_dirs.extend([root_dir, root_dir.parent])
    search_dirs.extend([case_dir.parent, case_dir.parent.parent])

    seen: set[Path] = set()
    for d in search_dirs:
        d = d.resolve()
        if d in seen:
            continue
        seen.add(d)
        shared = d / "TEs.txt"
        if shared.exists():
            return shared
    return None


def discover_cases(root_dir: str | Path) -> list[CasePaths]:
    root = Path(root_dir).resolve()
    qt2_files = sorted(root.rglob('*-qt2_reg.nii.gz'))
    cases: list[CasePaths] = []
    seen_keys: set[tuple[str, Path]] = set()
    for qt2_path in qt2_files:
        case_id = qt2_path.name.split('-', 1)[0]
        case_dir = qt2_path.parent
        case_key = (case_id, case_dir)
        if case_key in seen_keys:
            continue
        seen_keys.add(case_key)

        tes_path = _find_tes_path(case_dir, root_dir=root, case_id=case_id)
        mask_path = _find_first_existing(case_dir, [f'{case_id}-mask.nii.gz', f'{case_id}-mask1.nii.gz'])
        if mask_path is None:
            continue

        seg_path = _find_first_existing(case_dir, [f'{case_id}-seg.nii.gz', f'{case_id}-qt2_seg1.nii.gz'])
        par_path = _find_first_existing(case_dir, [f'{case_id}-par.nii.gz', f'{case_id}-qt2_par1.nii.gz'])
        par_lobe_path = _find_first_existing(case_dir, [f'{case_id}-par_lobe.nii.gz', f'{case_id}-qt2_par2.nii.gz'])
        if par_lobe_path is None:
            par_lobe_path = seg_path

        cases.append(
            CasePaths(
                case_id=case_id,
                case_dir=case_dir,
                qt2_reg=qt2_path,
                tes=tes_path,
                mask=mask_path,
                seg=seg_path,
                par=par_path,
                par_lobe=par_lobe_path,
            )
        )
    cases.sort(key=lambda c: _case_sort_key(c.case_id))
    return cases


def read_tes(path: str | Path) -> np.ndarray:
    te = np.loadtxt(str(path), dtype=np.float64)
    return np.asarray(te).reshape(-1)


def _load_nifti_fallback(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 352:
        raise ValueError(f"NIfTI header too short: {path}")

    endian = "<"
    sizeof_hdr = struct.unpack("<i", raw[0:4])[0]
    if sizeof_hdr != 348:
        sizeof_hdr_be = struct.unpack(">i", raw[0:4])[0]
        if sizeof_hdr_be == 348:
            endian = ">"
        else:
            raise ValueError(f"Invalid NIfTI header size in {path}: {sizeof_hdr}")

    dim = struct.unpack(endian + "8h", raw[40:56])
    ndim = int(dim[0])
    if ndim < 1 or ndim > 7:
        raise ValueError(f"Unsupported NIfTI ndim={ndim} in {path}")
    shape = tuple(int(v) for v in dim[1 : 1 + ndim])
    datatype = int(struct.unpack(endian + "h", raw[70:72])[0])
    vox_offset = int(round(float(struct.unpack(endian + "f", raw[108:112])[0])))
    scl_slope = float(struct.unpack(endian + "f", raw[112:116])[0])
    scl_inter = float(struct.unpack(endian + "f", raw[116:120])[0])

    if datatype not in _NIFTI_DTYPE_MAP:
        raise ValueError(f"Unsupported datatype {datatype} in {path}")
    dtype = _NIFTI_DTYPE_MAP[datatype].newbyteorder(endian)
    count = int(np.prod(shape))
    arr = np.frombuffer(raw, dtype=dtype, count=count, offset=vox_offset)
    arr = np.asarray(arr).reshape(shape, order="F").astype(np.float32, copy=False)

    if scl_slope not in (0.0, 1.0):
        arr = arr * np.float32(scl_slope)
    if scl_inter != 0.0:
        arr = arr + np.float32(scl_inter)
    return arr


def load_nifti(path: str | Path) -> np.ndarray:
    if nib is not None:
        try:
            img = nib.load(str(path))
            return np.asarray(img.get_fdata(dtype=np.float32))
        except Exception:
            pass
    return _load_nifti_fallback(path)


def _ensure_3d(arr: np.ndarray, *, name: str) -> np.ndarray:
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[3] == 1:
        return arr[..., 0]
    raise ValueError(f"{name} must be 3D or 4D with singleton last dim, got {arr.shape}")


def _resolve_case_paths(case_dir: str | Path, case_id: str) -> CasePaths:
    cdir = Path(case_dir).resolve()
    qt2_path = cdir / f'{case_id}-qt2_reg.nii.gz'
    if not qt2_path.exists():
        raise FileNotFoundError(qt2_path)

    tes_path = _find_tes_path(cdir, case_id=case_id)
    mask_path = _find_first_existing(cdir, [f'{case_id}-mask.nii.gz', f'{case_id}-mask1.nii.gz'])
    if mask_path is None:
        raise FileNotFoundError(cdir / f'{case_id}-mask*.nii.gz')

    seg_path = _find_first_existing(cdir, [f'{case_id}-seg.nii.gz', f'{case_id}-qt2_seg1.nii.gz'])
    par_path = _find_first_existing(cdir, [f'{case_id}-par.nii.gz', f'{case_id}-qt2_par1.nii.gz'])
    par_lobe_path = _find_first_existing(cdir, [f'{case_id}-par_lobe.nii.gz', f'{case_id}-qt2_par2.nii.gz'])
    if par_lobe_path is None:
        par_lobe_path = seg_path

    return CasePaths(
        case_id=case_id,
        case_dir=cdir,
        qt2_reg=qt2_path,
        tes=tes_path,
        mask=mask_path,
        seg=seg_path,
        par=par_path,
        par_lobe=par_lobe_path,
    )


def load_case(case_dir: str | Path, case_id: str) -> dict[str, Any]:
    paths = _resolve_case_paths(case_dir, case_id)
    data = load_nifti(paths.qt2_reg).astype(np.float32, copy=False)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D qt2 data for {case_id}, got {data.shape}")

    mask = _ensure_3d(load_nifti(paths.mask), name=f"{case_id} mask")
    mask = np.asarray(mask > 0, dtype=bool)

    if paths.tes is not None:
        tes = read_tes(paths.tes).astype(np.float64, copy=False)
    else:
        tes = np.arange(data.shape[3], dtype=np.float64)

    seg = load_nifti(paths.seg) if paths.seg is not None else None
    par = load_nifti(paths.par) if paths.par is not None else None
    par_lobe = load_nifti(paths.par_lobe) if paths.par_lobe is not None else None
    if par is not None:
        par = _ensure_3d(par, name=f"{case_id} par")
    if par_lobe is not None:
        par_lobe = _ensure_3d(par_lobe, name=f"{case_id} par_lobe")

    return {
        "case_id": case_id,
        "case_dir": paths.case_dir,
        "paths": paths,
        "data": data,
        "mask": mask,
        "tes": tes,
        "seg": seg,
        "par": par,
        "par_lobe": par_lobe,
    }


def audit_case(
    data: np.ndarray,
    mask: np.ndarray,
    tes: np.ndarray,
    *,
    case_id: str | None = None,
    eps: float = 1e-9,
) -> dict[str, Any]:
    if data.ndim != 4:
        raise ValueError(f"data must be 4D, got {data.shape}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got {mask.shape}")
    if data.shape[:3] != mask.shape:
        shape_match = False
    else:
        shape_match = True
    time_match = data.shape[3] == int(tes.size)

    te_diff = np.diff(tes)
    te_non_decreasing = bool(np.all(te_diff >= -eps))
    te_duplicates = int(np.sum(np.isclose(te_diff, 0.0, atol=eps)))
    te_min_step = float(np.min(te_diff)) if te_diff.size else math.nan

    finite_data = np.isfinite(data)
    finite_rate = 100.0 * float(finite_data.mean())
    nonpositive_rate = 100.0 * float(np.mean(data[mask, :] <= 0)) if mask.any() else math.nan

    return {
        "case_id": case_id,
        "shape_xyz_t": tuple(int(v) for v in data.shape),
        "mask_shape_xyz": tuple(int(v) for v in mask.shape),
        "shape_match": shape_match,
        "time_match": time_match,
        "te_count": int(tes.size),
        "te_non_decreasing": te_non_decreasing,
        "te_duplicate_count": te_duplicates,
        "te_min_step_ms": te_min_step,
        "brain_voxels": int(mask.sum()),
        "finite_data_rate_pct": finite_rate,
        "nonpositive_sample_rate_pct": nonpositive_rate,
    }


def fit_log_linear_r2(curves: np.ndarray, tes: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    if curves.ndim != 2:
        raise ValueError(f"curves must be 2D [n_vox, n_te], got {curves.shape}")
    if curves.shape[1] != tes.size:
        raise ValueError(f"Curve length {curves.shape[1]} does not match TEs {tes.size}")
    valid = np.all(curves > eps, axis=1)
    r2 = np.full(curves.shape[0], np.nan, dtype=np.float64)
    if not np.any(valid):
        return r2

    y = np.log(curves[valid].astype(np.float64))
    x = tes.astype(np.float64)
    x_mean = float(x.mean())
    x_center = x - x_mean
    denom = float(np.dot(x_center, x_center))
    if denom < eps:
        return r2

    beta = y.dot(x_center) / denom
    alpha = y.mean(axis=1) - beta * x_mean
    pred = alpha[:, None] + beta[:, None] * x[None, :]
    residual = y - pred
    ss_res = np.sum(residual * residual, axis=1)
    ss_tot = np.sum((y - y.mean(axis=1, keepdims=True)) ** 2, axis=1) + eps
    r2_valid = 1.0 - ss_res / ss_tot
    r2[valid] = r2_valid
    return r2


def compute_global_decay_profile(
    data: np.ndarray,
    mask: np.ndarray,
    tes: np.ndarray,
    *,
    z_thresh: float = 2.5,
    eps: float = 1e-6,
) -> dict[str, Any]:
    if data.ndim != 4:
        raise ValueError(f"data must be 4D, got {data.shape}")
    if data.shape[:3] != mask.shape:
        raise ValueError(f"data/mask shape mismatch: {data.shape[:3]} vs {mask.shape}")
    if data.shape[3] != tes.size:
        raise ValueError("data and TE count mismatch")

    if not np.any(mask):
        n_te = data.shape[3]
        return {
            "median_curve": np.full(n_te, np.nan),
            "rel_steps": np.full(max(n_te - 1, 0), np.nan),
            "step_z_scores": np.full(max(n_te - 1, 0), np.nan),
            "anomaly_indices": np.array([], dtype=np.int64),
            "anomaly_transitions": [],
        }

    signals = data[mask, :]
    med = np.median(signals, axis=0).astype(np.float64)
    rel_steps = np.diff(med) / np.maximum(med[:-1], eps)

    if rel_steps.size == 0:
        z = np.array([], dtype=np.float64)
        anomalies = np.array([], dtype=np.int64)
    else:
        center = float(np.median(rel_steps))
        mad = float(np.median(np.abs(rel_steps - center)))
        robust_std = max(1.4826 * mad, eps)
        z = (rel_steps - center) / robust_std
        anomalies = np.where(np.abs(z) >= z_thresh)[0].astype(np.int64)

    transitions = [
        f"{tes[i]:.0f}->{tes[i+1]:.0f}ms ({rel_steps[i] * 100:+.1f}%, z={z[i]:+.2f})"
        for i in anomalies
    ]
    return {
        "median_curve": med,
        "rel_steps": rel_steps,
        "step_z_scores": z,
        "anomaly_indices": anomalies,
        "anomaly_transitions": transitions,
    }


def compute_qc_metrics(
    data: np.ndarray,
    mask: np.ndarray,
    tes: np.ndarray,
    thresholds: dict[str, float] | None = None,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        cfg.update(thresholds)

    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.shape}")
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.shape}")
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Data/mask mismatch: {data.shape[:3]} vs {mask.shape}")
    if data.shape[3] != tes.size:
        raise ValueError(f"Data/TE mismatch: {data.shape[3]} vs {tes.size}")

    eps = float(cfg["eps"])
    rel_thresh = float(cfg["mono_rel_inc_thresh"])
    early_fraction = float(cfg["early_echo_fraction"])
    r2_poor_thresh = float(cfg["r2_poor_thresh"])
    max_voxels_for_r2 = int(cfg["max_voxels_for_r2"])
    echo_z_thresh = float(cfg["echo_z_thresh"])

    signals = data[mask, :].astype(np.float64, copy=False)
    n_vox = signals.shape[0]
    n_te = signals.shape[1]
    if n_vox == 0:
        return {
            "brain_voxels": 0,
            "te_count": int(tes.size),
            "te_duplicate_count": int(np.sum(np.isclose(np.diff(tes), 0.0, atol=eps))),
            "mono_violation_rate_pct": math.nan,
            "early_mono_violation_rate_pct": math.nan,
            "nonpositive_voxel_rate_pct": math.nan,
            "r2_median": math.nan,
            "r2_poor_rate_pct": math.nan,
            "echo_anomaly_count": 0,
            "echo_anomaly_transitions": "",
            "median_s0": math.nan,
            "median_last": math.nan,
            "median_decay_ratio_s0_over_last": math.nan,
        }

    rel_steps = np.diff(signals, axis=1) / np.maximum(signals[:, :-1], eps)
    mono_violation = np.any(rel_steps > rel_thresh, axis=1)
    mono_violation_rate = 100.0 * float(np.mean(mono_violation))

    early_count = int(max(2, min(n_te, round(n_te * early_fraction))))
    early_rel_steps = np.diff(signals[:, :early_count], axis=1) / np.maximum(
        signals[:, : early_count - 1], eps
    )
    early_violation = np.any(early_rel_steps > rel_thresh, axis=1)
    early_rate = 100.0 * float(np.mean(early_violation))

    nonpositive_rate = 100.0 * float(np.mean(np.any(signals <= 0.0, axis=1)))

    valid_for_r2 = np.where(np.all(signals > eps, axis=1))[0]
    if valid_for_r2.size:
        rng = np.random.default_rng(seed)
        k = min(max_voxels_for_r2, int(valid_for_r2.size))
        chosen = rng.choice(valid_for_r2, size=k, replace=False)
        r2_values = fit_log_linear_r2(signals[chosen, :], tes, eps=eps)
        r2_median = float(np.nanmedian(r2_values))
        r2_poor_rate = 100.0 * float(np.mean(r2_values < r2_poor_thresh))
    else:
        r2_median = math.nan
        r2_poor_rate = math.nan

    profile = compute_global_decay_profile(data, mask, tes, z_thresh=echo_z_thresh, eps=eps)
    anomaly_count = int(len(profile["anomaly_indices"]))
    anomaly_transitions = "; ".join(profile["anomaly_transitions"])

    med_curve = profile["median_curve"]
    med_s0 = float(med_curve[0]) if med_curve.size else math.nan
    med_last = float(med_curve[-1]) if med_curve.size else math.nan
    decay_ratio = med_s0 / max(med_last, eps) if np.isfinite(med_s0) and np.isfinite(med_last) else math.nan

    return {
        "brain_voxels": int(n_vox),
        "te_count": int(tes.size),
        "te_duplicate_count": int(np.sum(np.isclose(np.diff(tes), 0.0, atol=eps))),
        "mono_violation_rate_pct": mono_violation_rate,
        "early_mono_violation_rate_pct": early_rate,
        "nonpositive_voxel_rate_pct": nonpositive_rate,
        "r2_median": r2_median,
        "r2_poor_rate_pct": r2_poor_rate,
        "echo_anomaly_count": anomaly_count,
        "echo_anomaly_transitions": anomaly_transitions,
        "median_s0": med_s0,
        "median_last": med_last,
        "median_decay_ratio_s0_over_last": decay_ratio,
    }


def largest_roi_labels(
    roi_map: np.ndarray,
    mask: np.ndarray,
    *,
    top_k: int = 3,
    min_voxels: int = 100,
) -> list[int]:
    roi_int = np.asarray(np.rint(roi_map), dtype=np.int32)
    valid = roi_int[(roi_int > 0) & mask]
    if valid.size == 0:
        return []
    labels, counts = np.unique(valid, return_counts=True)
    keep = counts >= int(min_voxels)
    labels = labels[keep]
    counts = counts[keep]
    if labels.size == 0:
        return []
    order = np.argsort(counts)[::-1]
    selected = labels[order][: int(top_k)]
    return [int(v) for v in selected.tolist()]


def sample_curves(
    data: np.ndarray,
    mask: np.ndarray,
    roi_map: np.ndarray,
    n_per_roi: int = 100,
    seed: int = 0,
    *,
    tes: np.ndarray | None = None,
    roi_labels: list[int] | None = None,
) -> pd.DataFrame:
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.shape}")
    if mask.ndim != 3 or roi_map.ndim != 3:
        raise ValueError("mask and roi_map must be 3D")
    if data.shape[:3] != mask.shape or mask.shape != roi_map.shape:
        raise ValueError("data/mask/roi_map shape mismatch")

    t = data.shape[3]
    te_axis = np.arange(t, dtype=np.float64) if tes is None else np.asarray(tes, dtype=np.float64).reshape(-1)
    if te_axis.size != t:
        raise ValueError(f"TE size mismatch: {te_axis.size} vs {t}")

    roi_int = np.asarray(np.rint(roi_map), dtype=np.int32)
    labels = roi_labels if roi_labels is not None else largest_roi_labels(roi_int, mask, top_k=3, min_voxels=100)
    if not labels:
        return pd.DataFrame(
            columns=["roi_label", "roi_voxel_id", "x", "y", "z", "te_idx", "te_ms", "signal", "log_signal"]
        )

    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for label in labels:
        vox = np.argwhere((roi_int == int(label)) & mask)
        if vox.size == 0:
            continue
        pick = min(int(n_per_roi), int(vox.shape[0]))
        if pick <= 0:
            continue
        idx = rng.choice(vox.shape[0], size=pick, replace=False)
        chosen = vox[idx]
        curves = data[chosen[:, 0], chosen[:, 1], chosen[:, 2], :].astype(np.float64, copy=False)

        rows = pick * t
        frame = pd.DataFrame(
            {
                "roi_label": np.full(rows, int(label), dtype=np.int32),
                "roi_voxel_id": np.repeat(np.arange(pick, dtype=np.int32), t),
                "x": np.repeat(chosen[:, 0], t),
                "y": np.repeat(chosen[:, 1], t),
                "z": np.repeat(chosen[:, 2], t),
                "te_idx": np.tile(np.arange(t, dtype=np.int32), pick),
                "te_ms": np.tile(te_axis, pick),
                "signal": curves.reshape(-1),
            }
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            frame["log_signal"] = np.where(frame["signal"] > 0, np.log(frame["signal"]), np.nan)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["roi_label", "roi_voxel_id", "x", "y", "z", "te_idx", "te_ms", "signal", "log_signal"]
        )
    return pd.concat(frames, ignore_index=True)


def _score_case_risk(metrics: dict[str, Any]) -> tuple[float, str]:
    mono = float(metrics.get("mono_violation_rate_pct", 0.0)) / 100.0
    early = float(metrics.get("early_mono_violation_rate_pct", 0.0)) / 100.0
    poor_r2 = float(metrics.get("r2_poor_rate_pct", 0.0)) / 100.0
    nonpos = min(float(metrics.get("nonpositive_voxel_rate_pct", 0.0)) / 2.0, 1.0)
    anomalies = min(float(metrics.get("echo_anomaly_count", 0.0)) / 3.0, 1.0)
    duplicate_penalty = 0.1 if int(metrics.get("te_duplicate_count", 0)) > 0 else 0.0

    score = (
        0.30 * mono
        + 0.20 * early
        + 0.25 * poor_r2
        + 0.15 * nonpos
        + 0.10 * anomalies
        + duplicate_penalty
    )
    score = float(min(max(score, 0.0), 1.0))
    if score < 0.33:
        level = "low"
    elif score < 0.66:
        level = "medium"
    else:
        level = "high"
    return score, level


def _build_interpretation(metrics: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    distortions: list[str] = []
    impacts: list[str] = []
    recommendations: list[str] = []

    if int(metrics.get("te_duplicate_count", 0)) > 0:
        distortions.append("TE schedule inconsistency (duplicate echo time).")
        impacts.append("Can destabilize slope-based estimates and overweight a repeated echo.")
        recommendations.append("Deduplicate or merge repeated TEs before fitting.")

    if float(metrics.get("early_mono_violation_rate_pct", 0.0)) > 30.0:
        distortions.append("Early-echo non-monotonic behavior.")
        impacts.append("Suggests motion/registration mismatch, stimulated echoes, or multi-component mixing.")
        recommendations.append("Use robust loss and inspect short-TE outliers before model fitting.")

    if float(metrics.get("r2_poor_rate_pct", 0.0)) > 25.0:
        distortions.append("Substantial mono-exponential misfit.")
        impacts.append("Single-compartment T2 may be biased, especially in mixed tissue voxels.")
        recommendations.append("Prefer weighted/robust fitting and compare against multi-compartment models.")

    if float(metrics.get("nonpositive_voxel_rate_pct", 0.0)) > 0.1:
        distortions.append("Non-positive signal in masked voxels.")
        impacts.append("Log-linear transforms become unstable and can underestimate late-TE signal.")
        recommendations.append("Apply positivity handling (epsilon floor or robust nonlinear fit).")

    if int(metrics.get("echo_anomaly_count", 0)) > 0:
        distortions.append("Echo-specific global anomalies.")
        impacts.append("May indicate acquisition instability affecting specific TEs.")
        recommendations.append("Evaluate excluding anomalous echoes in sensitivity analyses.")

    if not distortions:
        distortions.append("No dominant distortion signature in configured checks.")
        impacts.append("Data appears compatible with basic mono-exponential assumptions at cohort level.")
        recommendations.append("Proceed with baseline fitting and keep QA thresholds documented.")

    return distortions, impacts, recommendations


def make_case_report(
    metrics: dict[str, Any],
    curves: pd.DataFrame | None,
    figs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score, level = _score_case_risk(metrics)
    distortions, impacts, recommendations = _build_interpretation(metrics)
    return {
        "case_id": metrics.get("case_id"),
        "risk_score": score,
        "risk_level": level,
        "key_distortions": distortions,
        "fit_impacts": impacts,
        "recommendations": recommendations,
        "n_curve_rows": int(0 if curves is None else len(curves)),
        "figures": {} if figs is None else dict(figs),
    }


def run_synthetic_sanity_checks(seed: int = 0, *, verbose: bool = False) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    tes = np.linspace(19.0, 150.0, 21, dtype=np.float64)
    n_vox = 600

    s0 = rng.uniform(900.0, 1200.0, size=(n_vox, 1))
    t2 = rng.uniform(50.0, 95.0, size=(n_vox, 1))
    clean = s0 * np.exp(-tes.reshape(1, -1) / t2)
    baseline = np.clip(clean + rng.normal(0.0, 8.0, size=clean.shape), 1e-3, None)

    # Inject one clear anomaly: short-TE bump and late-TE noise floor.
    anomalous = baseline.copy()
    affected = rng.choice(n_vox, size=n_vox // 3, replace=False)
    anomalous[affected, 5] *= 1.25
    anomalous[:, -3:] += 25.0

    # Duplicate one TE entry to test metadata anomaly handling.
    tes_dup = np.insert(tes, 10, tes[10])
    anomalous_dup = np.insert(anomalous, 10, anomalous[:, 10], axis=1)

    baseline_4d = baseline.reshape(n_vox, 1, 1, tes.size)
    anomalous_4d = anomalous.reshape(n_vox, 1, 1, tes.size)
    anomalous_dup_4d = anomalous_dup.reshape(n_vox, 1, 1, tes_dup.size)
    mask = np.ones((n_vox, 1, 1), dtype=bool)

    m_base = compute_qc_metrics(baseline_4d, mask, tes, seed=seed)
    m_bad = compute_qc_metrics(anomalous_4d, mask, tes, seed=seed)
    m_dup = compute_qc_metrics(anomalous_dup_4d, mask, tes_dup, seed=seed)

    # Directional assertions: anomalies should worsen QC metrics.
    assert m_base["r2_median"] > 0.95
    assert m_bad["mono_violation_rate_pct"] > m_base["mono_violation_rate_pct"]
    assert m_bad["r2_poor_rate_pct"] > m_base["r2_poor_rate_pct"]
    assert m_dup["te_duplicate_count"] >= 1

    if verbose:
        print("Synthetic sanity checks passed.")
        print("baseline:", m_base)
        print("anomalous:", m_bad)
        print("anomalous+dupTE:", m_dup)

    return {"baseline": m_base, "anomalous": m_bad, "anomalous_dup_te": m_dup}


def summarize_case_reports(reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in reports:
        rows.append(
            {
                "case_id": r.get("case_id"),
                "risk_score": r.get("risk_score"),
                "risk_level": r.get("risk_level"),
                "top_distortion": (r.get("key_distortions") or [""])[0],
                "top_recommendation": (r.get("recommendations") or [""])[0],
            }
        )
    df = pd.DataFrame(rows)
    if "risk_score" in df:
        df = df.sort_values(["risk_score", "case_id"], ascending=[False, True]).reset_index(drop=True)
    return df
