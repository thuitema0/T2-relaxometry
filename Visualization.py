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


if __name__ == "__main__":
    main()

