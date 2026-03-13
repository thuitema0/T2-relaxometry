from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from t2_analyser import T2Analyser


DATA_DIR = Path("/Users/zhuyingjie/Desktop/COMP0018/T2-relaxometry/preterm_data")

FILES = {
    "Mono T2": "Epicure66734_NLLS_Mono_T2.nii.gz",
    "Mono AIC": "Epicure66734_NLLS_Mono_AIC.nii.gz",
    "Bi AIC (F25,S150)": "Epicure66734_NLLS_Bi_F25_S150_AIC.nii.gz",
    "Delta AIC (Mono-Bi)": "Epicure66734_AIC_Comparison_Delta_AIC_Mono_minus_Bi.nii.gz",
}

OUT_PNG = DATA_DIR / "Epicure66734_model_maps_debug.png"


def load_map(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim != 3:
        raise ValueError(f"{path.name} is not 3D. Shape = {data.shape}")
    return data


def describe_map(name: str, data: np.ndarray) -> None:
    finite = np.isfinite(data)
    nonzero = finite & (data != 0)
    print(f"\n{name}")
    print(f"  shape: {data.shape}")
    print(f"  finite voxels: {finite.sum()}")
    print(f"  nonzero finite voxels: {nonzero.sum()}")
    if nonzero.any():
        vals = data[nonzero]
        print(f"  min: {vals.min():.4g}")
        print(f"  max: {vals.max():.4g}")
        print(f"  median: {np.median(vals):.4g}")
    else:
        print("  WARNING: no nonzero finite voxels found")


def choose_best_slice(volumes: dict[str, np.ndarray]) -> int:
    # Use union of all valid nonzero voxels
    shape = next(iter(volumes.values())).shape
    union_mask = np.zeros(shape, dtype=bool)

    for data in volumes.values():
        union_mask |= np.isfinite(data) & (data != 0)

    counts = [union_mask[:, :, z].sum() for z in range(shape[2])]
    best_z = int(np.argmax(counts))
    print(f"\nChosen slice z = {best_z}, valid voxels in slice = {counts[best_z]}")
    return best_z


def robust_limits(slice2d: np.ndarray, low=2, high=98):
    vals = slice2d[np.isfinite(slice2d) & (slice2d != 0)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(vals, low)
    vmax = np.percentile(vals, high)
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def symmetric_limits(slice2d: np.ndarray, pct=98):
    vals = slice2d[np.isfinite(slice2d) & (slice2d != 0)]
    if vals.size == 0:
        return -1.0, 1.0
    vmax = np.percentile(np.abs(vals), pct)
    vmax = max(float(vmax), 1e-6)
    return -vmax, vmax


def prepare_slice(data: np.ndarray, z: int) -> np.ma.MaskedArray:
    sl = data[:, :, z].T
    mask = ~np.isfinite(sl) | (sl == 0)
    return np.ma.masked_array(sl, mask=mask)


def main():
    volumes = {}
    for label, fname in FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        volumes[label] = load_map(path)
        describe_map(label, volumes[label])

    z = choose_best_slice(volumes)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    panels = [
        ("Mono T2", "viridis", False),
        ("Mono AIC", "magma", False),
        ("Bi AIC (F25,S150)", "magma", False),
        ("Delta AIC (Mono-Bi)", "coolwarm", True),
    ]

    for ax, (label, cmap, symmetric) in zip(axes.ravel(), panels):
        sl = prepare_slice(volumes[label], z)

        if sl.count() == 0:
            ax.set_title(f"{label}\nNo valid voxels in slice")
            ax.axis("off")
            continue

        if symmetric:
            vmin, vmax = symmetric_limits(sl)
        else:
            vmin, vmax = robust_limits(sl)

        im = ax.imshow(sl, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Epicure66734 model maps (slice z={z})", fontsize=14)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nSaved figure to: {OUT_PNG}")


# if __name__ == "__main__":
#     main()

# =========================
# NEW ANALYSIS / PLOTTING MAIN
# =========================

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def load_nii(path):
    img = nib.load(str(path))
    return img.get_fdata()


def masked_slice(vol, z):
    sl = vol[:, :, z].T
    mask = ~np.isfinite(sl) | (sl == 0)
    return np.ma.masked_array(sl, mask=mask)


def robust_limits(arr, low=2, high=98):
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals) & (vals != 0)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(vals, low)
    vmax = np.percentile(vals, high)
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def symmetric_limits(arr, pct=98):
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals) & (vals != 0)]
    if vals.size == 0:
        return -1.0, 1.0
    vmax = np.percentile(np.abs(vals), pct)
    vmax = max(float(vmax), 1e-6)
    return -vmax, vmax


def choose_best_slice(mask_3d):
    counts = [mask_3d[:, :, z].sum() for z in range(mask_3d.shape[2])]
    return int(np.argmax(counts))


def plot_t2_histogram(root, case, out_dir):
    wls_t2 = load_nii(root / f"{case}_WLS_T2.nii.gz")
    nnls_t2 = load_nii(root / f"{case}_NNLS_T2.nii.gz")
    mono_t2 = load_nii(root / f"{case}_NLLS_Mono_T2.nii.gz")

    def clean(vals):
        vals = vals[np.isfinite(vals) & (vals > 0) & (vals < 500)]
        return vals

    wls_vals = clean(wls_t2)
    nnls_vals = clean(nnls_t2)
    mono_vals = clean(mono_t2)

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 250, 120)
    plt.hist(wls_vals, bins=bins, alpha=0.45, density=True, label="WLS T2")
    plt.hist(nnls_vals, bins=bins, alpha=0.45, density=True, label="NNLS T2")
    plt.hist(mono_vals, bins=bins, alpha=0.45, density=True, label="NLLS Mono T2")
    plt.xlabel("T2 (ms)")
    plt.ylabel("Density")
    plt.title(f"{case}: T2 histogram")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path = out_dir / f"{case}_T2_histogram.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved: {out_path}")


def choose_voxel_from_delta_aic(root, case):
    delta = load_nii(root / f"{case}_AIC_Comparison_Delta_AIC_Mono_minus_Bi.nii.gz")
    mono_s0 = load_nii(root / f"{case}_NLLS_Mono_S0.nii.gz")
    mono_t2 = load_nii(root / f"{case}_NLLS_Mono_T2.nii.gz")
    bi_s0f = load_nii(root / f"{case}_NLLS_Bi_F25_S150_S0_fast.nii.gz")
    bi_s0s = load_nii(root / f"{case}_NLLS_Bi_F25_S150_S0_slow.nii.gz")
    bi_t2f = load_nii(root / f"{case}_NLLS_Bi_F25_S150_T2_fast.nii.gz")
    bi_t2s = load_nii(root / f"{case}_NLLS_Bi_F25_S150_T2_slow.nii.gz")

    valid = (
        np.isfinite(delta)
        & np.isfinite(mono_s0) & np.isfinite(mono_t2)
        & np.isfinite(bi_s0f) & np.isfinite(bi_s0s)
        & np.isfinite(bi_t2f) & np.isfinite(bi_t2s)
        & (mono_s0 > 0) & (mono_t2 > 0)
        & (bi_s0f >= 0) & (bi_s0s >= 0)
        & (bi_t2f > 0) & (bi_t2s > 0)
    )

    if not np.any(valid):
        raise RuntimeError("No valid voxels found for mono/bi comparison.")

    delta_valid = np.where(valid, delta, -np.inf)
    x, y, z = np.unravel_index(np.argmax(delta_valid), delta_valid.shape)
    return (x, y, z)


def plot_voxel_signal_decay(root, case, analyser, out_dir, voxel=None):
    te = analyser.tes
    data_4d = analyser.data

    mono_s0 = load_nii(root / f"{case}_NLLS_Mono_S0.nii.gz")
    mono_t2 = load_nii(root / f"{case}_NLLS_Mono_T2.nii.gz")

    bi_s0f = load_nii(root / f"{case}_NLLS_Bi_F25_S150_S0_fast.nii.gz")
    bi_s0s = load_nii(root / f"{case}_NLLS_Bi_F25_S150_S0_slow.nii.gz")
    bi_t2f = load_nii(root / f"{case}_NLLS_Bi_F25_S150_T2_fast.nii.gz")
    bi_t2s = load_nii(root / f"{case}_NLLS_Bi_F25_S150_T2_slow.nii.gz")

    delta = load_nii(root / f"{case}_AIC_Comparison_Delta_AIC_Mono_minus_Bi.nii.gz")

    if voxel is None:
        voxel = choose_voxel_from_delta_aic(root, case)

    x, y, z = voxel
    signal = data_4d[x, y, z, :]

    s0_m = mono_s0[x, y, z]
    t2_m = mono_t2[x, y, z]

    s0_f = bi_s0f[x, y, z]
    s0_s = bi_s0s[x, y, z]
    t2_f = bi_t2f[x, y, z]
    t2_s = bi_t2s[x, y, z]

    te_dense = np.linspace(float(np.min(te)), float(np.max(te)), 400)
    mono_curve_dense = s0_m * np.exp(-te_dense / t2_m)
    bi_curve_dense = s0_f * np.exp(-te_dense / t2_f) + s0_s * np.exp(-te_dense / t2_s)

    mono_curve_pts = s0_m * np.exp(-te / t2_m)
    bi_curve_pts = s0_f * np.exp(-te / t2_f) + s0_s * np.exp(-te / t2_s)

    plt.figure(figsize=(8, 5))
    plt.plot(te, signal, "o", label="Measured signal")
    plt.plot(te_dense, mono_curve_dense, "-", linewidth=2, label=f"Mono fit: S0={s0_m:.1f}, T2={t2_m:.1f} ms")
    plt.plot(
        te_dense,
        bi_curve_dense,
        "--",
        linewidth=2,
        label=f"Bi fit: Sf={s0_f:.1f}, T2f={t2_f:.1f} ms; Ss={s0_s:.1f}, T2s={t2_s:.1f} ms",
    )
    plt.scatter(te, mono_curve_pts, s=24)
    plt.scatter(te, bi_curve_pts, s=24)
    plt.xlabel("TE (ms)")
    plt.ylabel("Signal")
    plt.title(f"{case}: voxel ({x}, {y}, {z}), ΔAIC={delta[x, y, z]:.2f}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path = out_dir / f"{case}_voxel_signal_decay_mono_vs_bi.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved: {out_path}")
    print(f"Chosen voxel: ({x}, {y}, {z})")


def plot_fast_slow_t2_maps(root, case, out_dir, z=None):
    t2_fast = load_nii(root / f"{case}_NLLS_Bi_F25_S150_T2_fast.nii.gz")
    t2_slow = load_nii(root / f"{case}_NLLS_Bi_F25_S150_T2_slow.nii.gz")

    valid_mask = np.isfinite(t2_fast) & np.isfinite(t2_slow) & (t2_fast > 0) & (t2_slow > 0)
    if z is None:
        z = choose_best_slice(valid_mask)

    sl_fast = masked_slice(t2_fast, z)
    sl_slow = masked_slice(t2_slow, z)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    vmin_f, vmax_f = robust_limits(sl_fast)
    im0 = axes[0].imshow(sl_fast, origin="lower", cmap="viridis", vmin=vmin_f, vmax=vmax_f)
    axes[0].set_title(f"Fast T2 (slice z={z})")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    vmin_s, vmax_s = robust_limits(sl_slow)
    im1 = axes[1].imshow(sl_slow, origin="lower", cmap="magma", vmin=vmin_s, vmax=vmax_s)
    axes[1].set_title(f"Slow T2 (slice z={z})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(f"{case}: bi-exponential T2 components")
    out_path = out_dir / f"{case}_fast_vs_slow_T2_maps.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved: {out_path}")


def plot_fast_fraction_map(root, case, out_dir, z=None):
    s0_fast = load_nii(root / f"{case}_NLLS_Bi_F25_S150_S0_fast.nii.gz")
    s0_slow = load_nii(root / f"{case}_NLLS_Bi_F25_S150_S0_slow.nii.gz")

    denom = s0_fast + s0_slow
    frac_fast = np.full_like(s0_fast, np.nan, dtype=float)
    valid = np.isfinite(s0_fast) & np.isfinite(s0_slow) & (denom > 0)
    frac_fast[valid] = s0_fast[valid] / denom[valid]

    if z is None:
        z = choose_best_slice(valid)

    sl = masked_slice(frac_fast, z)

    plt.figure(figsize=(5, 5))
    im = plt.imshow(sl, origin="lower", cmap="plasma", vmin=0, vmax=1)
    plt.title(f"{case}: fast component fraction (slice z={z})")
    plt.axis("off")
    plt.colorbar(im, shrink=0.8, label="Sf / (Sf + Ss)")
    plt.tight_layout()
    out_path = out_dir / f"{case}_fast_component_fraction_map.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved: {out_path}")


def main():
    root = Path("/Users/zhuyingjie/Desktop/COMP0018/T2-relaxometry/preterm_data")
    case = "Epicure66734"
    out_dir = root / "extra_plots"
    out_dir.mkdir(exist_ok=True)

    data_path = root / f"{case}-qt2_reg.nii.gz"
    te_path = root / "TEs.txt"
    mask_path = root / f"{case}-mask1.nii.gz"

    analyser = T2Analyser(str(data_path), str(te_path), str(mask_path))

    # 1) T2 histogram
    plot_t2_histogram(root, case, out_dir)

    # 2) voxel signal decay with mono and bi fits
    plot_voxel_signal_decay(root, case, analyser, out_dir, voxel=None)
    # To force a specific voxel instead of auto-selecting the strongest bi-favoured voxel:
    # plot_voxel_signal_decay(root, case, analyser, out_dir, voxel=(48, 50, 29))

    # 3) fast vs slow T2 component maps
    plot_fast_slow_t2_maps(root, case, out_dir, z=None)

    # 4) fast component fraction map
    plot_fast_fraction_map(root, case, out_dir, z=None)


if __name__ == "__main__":
    main()