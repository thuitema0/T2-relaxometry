from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2: visualize T2 fitted brain slices with overlay"
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Path to the folder containing NIfTI images and masks",
    )
    parser.add_argument(
        "--fit-root", type=str, required=True,
        help="Path to the folder containing fitted maps (output of fit_all)",
    )
    parser.add_argument(
        "--subjects", type=str, nargs="*", default=None,
        help="Subject IDs to visualize (default: auto-discover all)",
    )
    parser.add_argument(
        "--methods", type=str, nargs="*",
        default=["ls", "weighted_ls", "nnls_grid", "nlls"],
        help="Methods to visualize (default: ls weighted_ls nnls_grid nlls)",
    )
    parser.add_argument(
        "--map", type=str, default="t2", choices=["t2", "nrmse", "s0"],
        help="Which parameter map to show (default: t2)",
    )
    parser.add_argument(
        "--t2-vmin", type=float, default=20,
        help="Min T2 for colour scale (default: 20)",
    )
    parser.add_argument(
        "--t2-vmax", type=float, default=200,
        help="Max T2 for colour scale (default: 200)",
    )
    parser.add_argument(
        "--overlay-alpha", type=float, default=0.65,
        help="Overlay transparency (default: 0.65)",
    )
    return parser.parse_args()


# ============================================================
# Global paths (set from CLI in main)
# ============================================================

DATA_ROOT: Path = None  # type: ignore
FIT_ROOT: Path = None  # type: ignore
OUT_ROOT: Path = None  # type: ignore
METHODS_TO_SHOW: list = None  # type: ignore
MAP_TO_SHOW: str = "t2"
T2_VMIN: float = 20
T2_VMAX: float = 200
OVERLAY_ALPHA: float = 0.65

BACKGROUND_MODE = "mean_first_3_echoes"
SLICE_FRACTIONS = [0.30, 0.50, 0.70]


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


def find_subject_image_and_mask(subject_id: str):
    qt2_reg = sorted(DATA_ROOT.glob(f"{subject_id}-qt2_reg.nii*"))
    qt2 = sorted(DATA_ROOT.glob(f"{subject_id}-qt2.nii*"))
    mask1 = sorted(DATA_ROOT.glob(f"{subject_id}-mask1.nii*"))
    mask0 = sorted(DATA_ROOT.glob(f"{subject_id}-mask.nii*"))

    img_file = qt2_reg[0] if len(qt2_reg) > 0 else (qt2[0] if len(qt2) > 0 else None)
    mask_file = mask1[0] if len(mask1) > 0 else (mask0[0] if len(mask0) > 0 else None)

    if img_file is None or mask_file is None:
        raise FileNotFoundError(f"Missing image or mask for {subject_id}")

    return img_file, mask_file


def find_method_map(subject_id: str, method: str, map_name: str):
    case_dir = FIT_ROOT / subject_id
    f = case_dir / f"{subject_id}_{map_name}_{method}.nii.gz"
    if not f.exists():
        raise FileNotFoundError(f"Missing map: {f}")
    return f


def choose_background(img4d: np.ndarray):
    if BACKGROUND_MODE == "first_echo":
        return img4d[..., 0].astype(np.float32)
    if BACKGROUND_MODE == "mean_first_3_echoes":
        num = min(3, img4d.shape[3])
        return np.mean(img4d[..., :num], axis=3).astype(np.float32)
    raise ValueError(f"Unknown BACKGROUND_MODE: {BACKGROUND_MODE}")


def choose_slices(mask: np.ndarray):
    z_has_mask = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    if len(z_has_mask) == 0:
        raise ValueError("Empty mask")

    z_min, z_max = int(z_has_mask.min()), int(z_has_mask.max())
    slices = []
    for frac in SLICE_FRACTIONS:
        z = int(round(z_min + frac * (z_max - z_min)))
        z = max(z_min, min(z, z_max))
        slices.append(z)
    return slices


def robust_range(arr: np.ndarray, mask: np.ndarray, p_low=1, p_high=99):
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, p_low)), float(np.percentile(vals, p_high))


def rotate(slice2d: np.ndarray):
    return np.rot90(slice2d)


def make_figure(subject_id: str, method: str):
    img_file, mask_file = find_subject_image_and_mask(subject_id)
    map_file = find_method_map(subject_id, method, MAP_TO_SHOW)

    _, img4d = load_nifti(img_file)
    _, mask = load_nifti(mask_file)
    _, param_map = load_nifti(map_file)

    mask = mask > 0
    bg = choose_background(img4d)

    if bg.shape != mask.shape or param_map.shape != mask.shape:
        raise ValueError(f"{subject_id}: shape mismatch")

    slices = choose_slices(mask)
    bg_vmin, bg_vmax = robust_range(bg, mask)

    if MAP_TO_SHOW == "t2":
        map_vmin, map_vmax = T2_VMIN, T2_VMAX
        cmap_color = "turbo"
        map_title = "T2 (ms)"
    elif MAP_TO_SHOW == "nrmse":
        vals = param_map[mask]
        vals = vals[np.isfinite(vals)]
        map_vmin = 0.0
        map_vmax = float(np.percentile(vals, 99)) if vals.size > 0 else 0.1
        cmap_color = "magma"
        map_title = "NRMSE"
    else:
        vals = param_map[mask]
        vals = vals[np.isfinite(vals)]
        map_vmin = float(np.percentile(vals, 1)) if vals.size > 0 else 0.0
        map_vmax = float(np.percentile(vals, 99)) if vals.size > 0 else 1.0
        cmap_color = "viridis"
        map_title = "S0"

    fig, axes = plt.subplots(3, len(slices), figsize=(4 * len(slices), 10))

    if len(slices) == 1:
        axes = np.array(axes).reshape(3, 1)

    overlay_im = None
    graymap_im = None

    for col, z in enumerate(slices):
        bg_slice = rotate(bg[:, :, z])
        map_slice = rotate(param_map[:, :, z])
        mask_slice = rotate(mask[:, :, z])

        bg_show = bg_slice.copy()
        bg_show[~mask_slice] = np.nan

        map_show = map_slice.copy()
        map_show[~mask_slice] = np.nan

        ax = axes[0, col]
        ax.imshow(bg_show, cmap="gray", vmin=bg_vmin, vmax=bg_vmax)
        ax.set_title(f"{subject_id} | anatomy | z={z}")
        ax.axis("off")

        ax = axes[1, col]
        graymap_im = ax.imshow(map_show, cmap="gray", vmin=map_vmin, vmax=map_vmax)
        ax.set_title(f"{subject_id} | {method} | {map_title} grayscale")
        ax.axis("off")

        ax = axes[2, col]
        ax.imshow(bg_show, cmap="gray", vmin=bg_vmin, vmax=bg_vmax)
        overlay_im = ax.imshow(
            map_show,
            cmap=cmap_color,
            vmin=map_vmin,
            vmax=map_vmax,
            alpha=OVERLAY_ALPHA,
        )
        ax.set_title(f"{subject_id} | {method} | {map_title} overlay")
        ax.axis("off")

    cbar1 = fig.colorbar(graymap_im, ax=axes[1, :], fraction=0.02, pad=0.02)
    cbar1.set_label(f"{map_title} grayscale")

    cbar2 = fig.colorbar(overlay_im, ax=axes[2, :], fraction=0.02, pad=0.02)
    cbar2.set_label(f"{map_title} overlay")

    fig.suptitle(f"{subject_id} - {method} brain slices", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_file = OUT_ROOT / f"{subject_id}_{method}_{MAP_TO_SHOW}_brain_slices.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_file}")


def main():
    global DATA_ROOT, FIT_ROOT, OUT_ROOT, METHODS_TO_SHOW
    global MAP_TO_SHOW, T2_VMIN, T2_VMAX, OVERLAY_ALPHA
    args = parse_args()
    DATA_ROOT = Path(args.data_root)
    FIT_ROOT = Path(args.fit_root)
    OUT_ROOT = FIT_ROOT / "brain_slice_figures"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METHODS_TO_SHOW = args.methods
    MAP_TO_SHOW = args.map
    T2_VMIN = args.t2_vmin
    T2_VMAX = args.t2_vmax
    OVERLAY_ALPHA = args.overlay_alpha

    subject_ids = args.subjects if args.subjects is not None else discover_subject_ids()

    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"FIT_ROOT  = {FIT_ROOT}")
    print(f"OUT_ROOT  = {OUT_ROOT}")
    print(f"Subjects  = {subject_ids}")
    print(f"Methods   = {METHODS_TO_SHOW}")

    for sid in subject_ids:
        for method in METHODS_TO_SHOW:
            try:
                make_figure(sid, method)
            except Exception as e:
                print(f"[ERROR] {sid} | {method}: {e}")


if __name__ == "__main__":
    main()