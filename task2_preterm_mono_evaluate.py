from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib


# ============================================================
# User configuration
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\18828\Desktop\研究生打工\biomedical modelling\cw2\relaxometry\cmbi_data\cmbi_data"
)

FIT_ROOT = Path(
    r"C:\Users\18828\Desktop\研究生打工\biomedical modelling\cw2\relaxometry\task2_preterm_mono_outputs"
)

OUT_ROOT = FIT_ROOT / "evaluation"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

METHODS = ["weighted_ls", "nnls_grid", "nlls"]

# Updated mapping for the preterm / Epicure dataset
# Based on current fitted T2 behaviour:
# label 1 -> CSF (highest T2)
# label 2 -> GM  (middle T2)
# label 3 -> WM  (lowest T2)
TISSUE_LABELS = {
    "WM": [3],
    "GM": [2],
    "CSF": [1],
}


# ============================================================
# Helpers
# ============================================================

def load_nifti(path: Path):
    nii = nib.load(str(path))
    arr = nii.get_fdata(dtype=np.float32)
    return nii, arr


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
    return sorted(list(set(ids)))


def find_subject_seg_mask(subject_id: str):
    mask1 = sorted(DATA_ROOT.glob(f"{subject_id}-mask1.nii*"))
    mask0 = sorted(DATA_ROOT.glob(f"{subject_id}-mask.nii*"))
    seg1 = sorted(DATA_ROOT.glob(f"{subject_id}-qt2_seg1.nii*"))
    seg0 = sorted(DATA_ROOT.glob(f"{subject_id}-qt2_seg.nii*"))

    mask_file = mask1[0] if len(mask1) > 0 else (mask0[0] if len(mask0) > 0 else None)
    seg_file = seg1[0] if len(seg1) > 0 else (seg0[0] if len(seg0) > 0 else None)

    if mask_file is None or seg_file is None:
        raise FileNotFoundError(f"Missing mask or seg for {subject_id}")

    return mask_file, seg_file


def convert_seg_to_hard_labels(seg: np.ndarray):
    if seg.ndim == 3:
        return np.round(seg).astype(np.int32)
    if seg.ndim == 4:
        return np.argmax(seg, axis=3).astype(np.int32)
    raise ValueError(f"Unsupported seg ndim: {seg.ndim}")


def summarize(values: np.ndarray):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ============================================================
# Main
# ============================================================

subjects = discover_subject_ids()
label_rows = []
tissue_rows = []

for sid in subjects:
    try:
        mask_file, seg_file = find_subject_seg_mask(sid)
        _, mask = load_nifti(mask_file)
        _, seg = load_nifti(seg_file)

        mask = mask > 0
        hard = convert_seg_to_hard_labels(seg)

        labels, counts = np.unique(hard[mask], return_counts=True)
        for lab, cnt in zip(labels, counts):
            label_rows.append({
                "subject_id": sid,
                "label": int(lab),
                "voxel_count": int(cnt),
            })

        for method in METHODS:
            case_dir = FIT_ROOT / sid
            t2_file = case_dir / f"{sid}_t2_{method}.nii.gz"
            nrmse_file = case_dir / f"{sid}_nrmse_{method}.nii.gz"
            valid_file = case_dir / f"{sid}_valid_{method}.nii.gz"

            if not t2_file.exists():
                continue

            _, t2 = load_nifti(t2_file)
            _, nrmse = load_nifti(nrmse_file)
            _, valid = load_nifti(valid_file)

            valid_fit = valid > 0.5

            for tissue_name, labels_used in TISSUE_LABELS.items():
                tissue_mask = mask & np.isin(hard, labels_used) & valid_fit

                t2_stats = summarize(t2[tissue_mask])
                nrmse_stats = summarize(nrmse[tissue_mask])

                tissue_rows.append({
                    "subject_id": sid,
                    "method": method,
                    "tissue": tissue_name,
                    "labels_used": ",".join(map(str, labels_used)),
                    "num_valid_voxels": t2_stats["n"],
                    "t2_mean_ms": t2_stats["mean"],
                    "t2_median_ms": t2_stats["median"],
                    "t2_std_ms": t2_stats["std"],
                    "t2_q25_ms": t2_stats["q25"],
                    "t2_q75_ms": t2_stats["q75"],
                    "nrmse_mean": nrmse_stats["mean"],
                    "nrmse_median": nrmse_stats["median"],
                })

    except Exception as e:
        print(f"[ERROR] {sid}: {e}")

label_df = pd.DataFrame(label_rows)
tissue_df = pd.DataFrame(tissue_rows)

label_df.to_csv(OUT_ROOT / "label_counts_all_subjects.csv", index=False)
tissue_df.to_csv(OUT_ROOT / "tissue_summary_all_methods.csv", index=False)

# Method-level aggregate
summary_csv = FIT_ROOT / "all_methods_all_subjects_summary.csv"
if summary_csv.exists():
    method_df = pd.read_csv(summary_csv)
    agg = (
        method_df.groupby("method", as_index=False)
        .agg(
            num_subjects=("subject_id", "nunique"),
            mean_runtime_sec=("runtime_sec", "mean"),
            std_runtime_sec=("runtime_sec", "std"),
            mean_valid_fit_fraction=("valid_fit_fraction", "mean"),
            mean_t2_mean_ms=("t2_mean_ms_in_range", "mean"),
            mean_t2_median_ms=("t2_median_ms_in_range", "mean"),
            mean_fraction_t2_gt_500=("fraction_t2_gt_500", "mean"),
            mean_nrmse=("nrmse_mean", "mean"),
        )
    )
    agg.to_csv(OUT_ROOT / "method_level_summary.csv", index=False)

# Tissue-level aggregate
if len(tissue_df) > 0:
    tissue_agg = (
        tissue_df.groupby(["method", "tissue"], as_index=False)
        .agg(
            num_subjects=("subject_id", "nunique"),
            mean_num_valid_voxels=("num_valid_voxels", "mean"),
            mean_t2_mean_ms=("t2_mean_ms", "mean"),
            mean_t2_median_ms=("t2_median_ms", "mean"),
            mean_t2_std_ms=("t2_std_ms", "mean"),
            mean_nrmse=("nrmse_mean", "mean"),
        )
    )
    tissue_agg.to_csv(OUT_ROOT / "tissue_level_summary.csv", index=False)

print("Saved:")
print(OUT_ROOT / "label_counts_all_subjects.csv")
print(OUT_ROOT / "tissue_summary_all_methods.csv")
if (OUT_ROOT / "method_level_summary.csv").exists():
    print(OUT_ROOT / "method_level_summary.csv")
if (OUT_ROOT / "tissue_level_summary.csv").exists():
    print(OUT_ROOT / "tissue_level_summary.csv")