import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import time
from pathlib import Path
import multiprocessing


def _guard_positive(s, eps=1e-6):
    return np.maximum(s, eps)

def analytic_two_echo(S, TEs):
    """Analytic two-echo T2 estimation."""
    S = _guard_positive(S)
    TE1, TE2 = TEs
    S1, S2 = S[..., 0], S[..., 1]
    ratio = np.clip(S1 / S2, 1e-6, 1e6)
    denom = np.log(ratio)
    denom = np.where(np.abs(denom) < 1e-8, np.sign(denom) * 1e-8 + 1e-12, denom)
    T2 = (TE2 - TE1) / denom
    S0 = S1 * np.exp(TE1 / T2)
    return S0, T2

def log_linear_ls(S, TEs, mask=None):
    S = _guard_positive(S)
    y = np.log(S)
    X1 = -TEs
    w = np.ones_like(y)

    A00 = np.sum(w, axis=-1)
    A01 = np.sum(w * X1, axis=-1)
    A11 = np.sum(w * X1 * X1, axis=-1)
    b0  = np.sum(w * y, axis=-1)
    b1  = np.sum(w * X1 * y, axis=-1)

    det = np.where(np.abs(A00*A11 - A01*A01) < 1e-12, 1e-12, A00*A11 - A01*A01)
    lnS0 = (b0 * A11 - b1 * A01) / det
    slope = (A00 * b1 - A01 * b0) / det 
    T2 = np.divide(1.0, slope, out=np.full_like(slope, np.nan), where=np.abs(slope) > 1e-9)
    S0 = np.exp(lnS0)

    if mask is not None:
        S0_out = np.full(mask.shape, np.nan); T2_out = np.full(mask.shape, np.nan)
        S0_out[mask] = S0[mask]; T2_out[mask] = T2[mask]
        return S0_out, T2_out
    return S0, T2

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
    T2 = np.divide(1.0, slope, out=np.full_like(slope, np.nan), where=np.abs(slope) > 1e-9)
    S0 = np.exp(lnS0)

    if mask is not None:
        S0_out = np.full(mask.shape, np.nan); T2_out = np.full(mask.shape, np.nan)
        S0_out[mask] = S0[mask]; T2_out[mask] = T2[mask]
        return S0_out, T2_out
    return S0, T2

def bounded_log_ls(S, TEs, mask=None, T2_min=5.0, T2_max=5000.0, eps=1e-6):
    S = _guard_positive(S, eps)
    y = np.log(S)
    X = np.vstack([np.ones_like(TEs), -TEs]).T
    shp = y.shape[:-1]
    lnS0 = np.full(shp, np.nan); invT2 = np.full(shp, np.nan)
    
    invT2_min = 1.0 / T2_max
    invT2_max = 1.0 / T2_min
    
    it = np.ndindex(shp)
    for idx in it:
        if mask is not None and not mask[idx]:
            continue
        yi = y[idx]
        def res(beta): return X @ beta - yi
        x0 = [float(yi.max()), 1.0/80.0]
        try:
            res_opt = least_squares(res, x0, bounds=([-np.inf, invT2_min], [np.inf, invT2_max]), max_nfev=100)
            lnS0[idx], invT2[idx] = res_opt.x
        except:
            pass
    T2 = np.divide(1.0, invT2, out=np.full_like(invT2, np.nan), where=np.abs(invT2) > 1e-12)
    S0 = np.exp(lnS0)
    return S0, T2

def nlls_single_voxel(S, TEs, bounds=([0, 5], [np.inf, 5000])):
    S = _guard_positive(S)
    def model(te, S0, T2): return S0 * np.exp(-te / T2)
    if len(TEs) >= 2:
        TE1, TE2 = TEs[:2]
        S1, S2 = S[0], S[1]
        ratio = np.clip(S1/S2, 1e-6, 1e6)
        denom = np.log(ratio)
        if abs(denom) < 1e-8:
            denom = np.sign(denom) * 1e-8 + 1e-12
        T2g = (TE2 - TE1) / denom
        S0g = S1 * np.exp(TE1 / T2g)
    else:
        S0g, T2g = S.max(), 80.0
    try:
        popt, _ = curve_fit(model, TEs, S,
                            p0=[float(S0g), float(np.clip(T2g, 5, 500))],
                            bounds=bounds, maxfev=2000)
    except:
        popt = [np.nan, np.nan]
    return popt

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

def _fit_bi_pixel(args):
    """Worker function for two compartment."""
    tes, sig, t2f_g, t2s_g = args
    
    p0 = [sig[0]/2, t2f_g, sig[0]/2, t2s_g]
    bounds = ([0, 5, 0, 60], [np.inf, 40, np.inf, 2500])
    def model(t, s0f, t2f, s0s, t2s):
        return (s0f * np.exp(-t / t2f)) + (s0s * np.exp(-t / t2s))
    try:
        popt, _ = curve_fit(model, tes, sig, p0=p0, bounds=bounds, maxfev=2000)
        return popt
    except:
        return [np.nan, np.nan, np.nan, np.nan]

class T2Analyser:
    def __init__(self, data_path, te_path, mask_path=None):
        img = nib.load(data_path)
        self.data = img.get_fdata()
        self.affine = img.affine
        self.tes = np.loadtxt(te_path)
        
        if mask_path:
            self.mask = nib.load(mask_path).get_fdata().astype(bool)
        else:
            self.mask = np.ones(self.data.shape[:3], dtype=bool)
            
        self.results = {} 

    @staticmethod
    def mono_exponential(te, s0, t2):
        return s0 * np.exp(-te / t2)

    @staticmethod
    def bi_exponential(te, s0_f, t2_f, s0_s, t2_s):
        """Two-compartment model (Fast and Slow components)."""
        return (s0_f * np.exp(-te / t2_f)) + (s0_s * np.exp(-te / t2_s))


    def fit_analytic(self, echo_indices=[0, 1]):
        """Task 2: Analytic fit using two specific echoes."""
        print(f"Running Analytic Fit on echoes {echo_indices}...")
        start = time.time()
        data_subset = self.data[..., echo_indices]
        tes_subset = self.tes[echo_indices]
        
        s0, t2 = analytic_two_echo(data_subset, tes_subset)
        
        if self.mask is not None:
            s0[~self.mask] = 0; t2[~self.mask] = 0
        
        elapsed = time.time() - start
        print(f"Analytic fit took {elapsed:.2f} seconds")
        self.results[f'Analytic_{echo_indices[0]}&{echo_indices[1]}'] = {
            'T2': t2, 'S0': s0, 'time': elapsed
        }

    def fit_lls(self):
        print("Running Linear Least Squares...")
        start = time.time()
        s0, t2 = log_linear_ls(self.data, self.tes, mask=self.mask)
        elapsed = time.time() - start
        print(f"LLS fit took {elapsed:.2f} seconds")
        self.results['LLS'] = {'T2': t2, 'S0': s0, 'time': elapsed}

    def fit_wls(self):
        print("Running Weighted Linear Least Squares...")
        start = time.time()
        s0, t2 = weighted_log_linear_ls(self.data, self.tes, mask=self.mask)
        elapsed = time.time() - start
        print(f"WLS fit took {elapsed:.2f} seconds")
        self.results['WLS'] = {'T2': t2, 'S0': s0, 'time': elapsed}

    def fit_nnls(self):
        print("Running Non-Negative (Bounded) Linear Least Squares...")
        start = time.time()
        s0, t2 = bounded_log_ls(self.data, self.tes, mask=self.mask)
        elapsed = time.time() - start
        print(f"NNLS fit took {elapsed:.2f} seconds")
        self.results['NNLS'] = {'T2': t2, 'S0': s0, 'time': elapsed}

    def fit_nlls_mono(self):
        print("Running Non-Linear Least Squares (Mono)...")
        start = time.time()
        s0, t2 = nlls_volume(self.data, self.tes, mask=self.mask)
        elapsed = time.time() - start
        print(f"NLLS Mono fit took {elapsed:.2f} seconds")
        self.results['NLLS_Mono'] = {'T2': t2, 'S0': s0, 'time': elapsed}

    # --- TASK 3 ---

    def fit_nlls_bi(self, t2_fast_guess=20, t2_slow_guess=80, suffix=""):
        """Fits the two-compartment model to the data."""
        print(f"Running Non-Linear Least Squares (Bi-Exponential) [Guess: {t2_fast_guess}, {t2_slow_guess}]...")
        start = time.time()
        s0_f = np.full(self.mask.shape, np.nan); t2_f = np.full(self.mask.shape, np.nan)
        s0_s = np.full(self.mask.shape, np.nan); t2_s = np.full(self.mask.shape, np.nan)
        
        indices = np.argwhere(self.mask)


        # Execute over multiple cores
        signals = self.data[indices[:,0], indices[:,1], indices[:,2], :]
        args = [(self.tes, sig, t2_fast_guess, t2_slow_guess) for sig in signals]
        
        n_cores = multiprocessing.cpu_count()
        print(f"  ...parallelising over {n_cores} cores...")
        
        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.map(_fit_bi_pixel, args)
            

        results = np.array(results)
        s0_f[indices[:,0], indices[:,1], indices[:,2]] = results[:,0]
        t2_f[indices[:,0], indices[:,1], indices[:,2]] = results[:,1]
        s0_s[indices[:,0], indices[:,1], indices[:,2]] = results[:,2]
        t2_s[indices[:,0], indices[:,1], indices[:,2]] = results[:,3]
        
        elapsed = time.time() - start
        print(f"NLLS Bi fit took {elapsed:.2f} seconds")
        self.results[f'NLLS_Bi{suffix}'] = {
            'T2_fast': t2_f, 'S0_fast': s0_f, 
            'T2_slow': t2_s, 'S0_slow': s0_s, 
            'time': elapsed
        }

    def calculate_maps_aic(self, bi_key="NLLS_Bi"):
        print(f"Calculating AIC maps for {bi_key}...")
        if 'NLLS_Mono' not in self.results or bi_key not in self.results:
            print(f"Skipping AIC: Run fit_nlls_mono and fit_nlls_bi ({bi_key}) first.")
            return

        n = len(self.tes)
        # Reshape TEs to (1, 1, 1, E)
        tes_4d = self.tes.reshape(1, 1, 1, -1)

        # Mono AIC (k=2)
        res_mono = self.results['NLLS_Mono']
        s0_m = res_mono['S0'][..., np.newaxis]
        t2_m = res_mono['T2'][..., np.newaxis]
        t2_m[t2_m == 0] = 1e-9
        model_mono = s0_m * np.exp(-tes_4d / t2_m)
        rss_mono = np.sum((self.data - model_mono)**2, axis=-1)
        rss_mono[rss_mono <= 0] = 1e-10
        aic_mono = n * np.log(rss_mono / n) + 2 * 2
        aic_mono[~self.mask] = np.nan
        self.results['NLLS_Mono']['AIC'] = aic_mono

        # Bi AIC (k=4)
        res_bi = self.results[bi_key]
        s0f = res_bi['S0_fast'][..., np.newaxis]; t2f = res_bi['T2_fast'][..., np.newaxis]
        s0s = res_bi['S0_slow'][..., np.newaxis]; t2s = res_bi['T2_slow'][..., np.newaxis]
        t2f[t2f == 0] = 1e-9; t2s[t2s == 0] = 1e-9
        model_bi = (s0f * np.exp(-tes_4d / t2f)) + (s0s * np.exp(-tes_4d / t2s))
        rss_bi = np.sum((self.data - model_bi)**2, axis=-1)
        rss_bi[rss_bi <= 0] = 1e-10
        aic_bi = n * np.log(rss_bi / n) + 2 * 4
        aic_bi[~self.mask] = np.nan
        self.results[bi_key]['AIC'] = aic_bi
        

        self.results[f'AIC_Comparison_{bi_key}'] = {'Delta_AIC_Mono_minus_Bi': aic_mono - aic_bi}
        self.results['AIC_Comparison'] = {'Delta_AIC_Mono_minus_Bi': aic_mono - aic_bi}

    def roi_analysis(self, seg_path, labels={'CSF': 1, 'GM': 2, 'WM': 3}):
        print(f"--- ROI Analysis using {seg_path} ---")
        if not Path(seg_path).exists():
            print(f"Segmentation file not found: {seg_path}")
            return

        seg = nib.load(seg_path).get_fdata()
        
        if seg.ndim == 4:
            print(f"  Segmentation is 4D {seg.shape}. Converting to 3D label map via argmax.")
            seg = np.argmax(seg, axis=-1)
            
        seg = seg.astype(int)
        
        for tissue, label_id in labels.items():
            mask_roi = (seg == label_id) & self.mask
            if not np.any(mask_roi):
                continue
            
            avg_sig = np.mean(self.data[mask_roi], axis=0)
            print(f"Bootstrapping {tissue}...")
            
            # Bootstrap Mono
            p_mono, ci_mono = self._bootstrap_fit(
                self.mono_exponential, self.tes, avg_sig, 
                p0=[avg_sig[0], 80], bounds=([0, 5], [np.inf, 5000])
            )
            
            # Bootstrap Bi
            p0_bi = [avg_sig[0]/2, 20, avg_sig[0]/2, 80]
            bounds_bi = ([0, 5, 0, 40], [np.inf, 40, np.inf, 5000])
            p_bi, ci_bi = self._bootstrap_fit(
                self.bi_exponential, self.tes, avg_sig, 
                p0=p0_bi, bounds=bounds_bi
            )
            if 'AIC_Comparison' in self.results:
                daic_map = self.results['AIC_Comparison']['Delta_AIC_Mono_minus_Bi']
                avg_daic = np.nanmean(daic_map[mask_roi])
                
                print(f"  {tissue} Avg ΔAIC (Mono-Bi): {avg_daic:.2f}")
            
            print(f"  {tissue} Mono T2: {p_mono[1]:.2f} ms, 95% CI [{ci_mono[1][0]:.2f}, {ci_mono[1][1]:.2f}]")
            print(f"  {tissue} Bi T2s: Fast {p_bi[1]:.2f} ms [{ci_bi[1][0]:.2f}, {ci_bi[1][1]:.2f}], Slow {p_bi[3]:.2f} ms [{ci_bi[3][0]:.2f}, {ci_bi[3][1]:.2f}]")

    def _bootstrap_fit(self, model_func, x, y, p0, bounds, n_boot=200):
        try:
            popt, _ = curve_fit(model_func, x, y, p0=p0, bounds=bounds, maxfev=5000)
        except RuntimeError:
            return np.full(len(p0), np.nan), np.full((len(p0), 2), np.nan)
            
        y_pred = model_func(x, *popt)
        residuals = y - y_pred
        
        boot_params = []
        for _ in range(n_boot):
            res_resampled = np.random.choice(residuals, size=len(residuals), replace=True)
            y_boot = y_pred + res_resampled
            try:
                p_boot, _ = curve_fit(model_func, x, y_boot, p0=popt, bounds=bounds, maxfev=5000)
                boot_params.append(p_boot)
            except:
                continue
        
        boot_params = np.array(boot_params)
        if len(boot_params) == 0:
             return popt, np.full((len(p0), 2), np.nan)
             
        ci_lower = np.percentile(boot_params, 2.5, axis=0)
        ci_upper = np.percentile(boot_params, 97.5, axis=0)
        return popt, np.column_stack((ci_lower, ci_upper))

    def show_results(self, slice_idx=None, percentiles=[1, 5, 10], save_prefix=None):
            """Standardized plotting for all fitted models with multiple contrast levels."""
            if not self.results:
                print("No results to plot. Run a fit method first.")
                return

            z = slice_idx if slice_idx is not None else self.data.shape[2] // 2
            
            plot_list = []
            for name, maps in self.results.items():
                # Handle Bi-Exponential splitting for visualisation
                if 'T2_fast' in maps and 'T2_slow' in maps:
                    # Fast Component
                    fast_map = {'T2': maps['T2_fast'], 'S0': maps['S0_fast']}
                    if 'time' in maps: fast_map['time'] = maps['time']
                    plot_list.append((f"{name} (Fast)", fast_map))
                    
                    # Slow Component
                    slow_map = {'T2': maps['T2_slow'], 'S0': maps['S0_slow']}
                    if 'time' in maps: slow_map['time'] = maps['time']
                    plot_list.append((f"{name} (Slow)", slow_map))
                elif 'T2' in maps:
                    plot_list.append((name, maps))
            
            num_methods = len(plot_list)
            if num_methods == 0:
                print("No map results found to plot.")
                return

            # Loop through percentiles
            for p_cut in percentiles:
                lower, upper = p_cut, 100 - p_cut
                print(f"Plotting with contrast range: {lower}% - {upper}%")

                fig, axes = plt.subplots(2, num_methods, figsize=(4 * num_methods, 8))
                if num_methods == 1: 
                    axes = np.array([[axes[0]], [axes[1]]])
                elif axes.ndim == 1:
                     axes = axes.reshape(2, num_methods)

                for i, (name, maps) in enumerate(plot_list):
                    t2_plot = maps['T2']
                    s0_plot = maps['S0']

                    # Scaling for T2
                    valid_t2 = t2_plot[self.mask & np.isfinite(t2_plot)]
                    vmin_t2, vmax_t2 = np.percentile(valid_t2, [lower, upper]) if valid_t2.size > 0 else (0, 200)

                    # Row 0: T2 Maps
                    ax_t2 = axes[0, i]
                    im_t2 = ax_t2.imshow(t2_plot[:, :, z].T, cmap='viridis', origin='lower', vmin=vmin_t2, vmax=vmax_t2)
                    ax_t2.set_title(f"{name}\nT2 ({maps.get('time', 0):.2f}s)")
                    ax_t2.axis('off')
                    plt.colorbar(im_t2, ax=ax_t2, fraction=0.046, pad=0.04)

                    # Scaling for S0
                    valid_s0 = s0_plot[self.mask & np.isfinite(s0_plot)]
                    vmin_s0, vmax_s0 = np.percentile(valid_s0, [lower, upper]) if valid_s0.size > 0 else (0, 1000)
                    
                    # Row 1: S0 Maps
                    ax_s0 = axes[1, i]
                    im_s0 = ax_s0.imshow(s0_plot[:, :, z].T, cmap='magma', origin='lower', vmin=vmin_s0, vmax=vmax_s0)
                    ax_s0.set_title(f"{name}\nS0")
                    ax_s0.axis('off')
                    plt.colorbar(im_s0, ax=ax_s0, fraction=0.046, pad=0.04)

                plt.suptitle(f"Contrast: {lower}% - {upper}% Percentile")
                plt.tight_layout()
                
                if save_prefix:
                    out_name = f"{save_prefix}_contrast_{lower}-{upper}.png"
                    plt.savefig(out_name, dpi=150)
                    print(f"Saved plot to {out_name}")

                plt.show()

    def save_all(self, prefix="case01"):
        """Saves all generated maps to NIfTI files."""
        for name, maps in self.results.items():
            for key, val in maps.items():
                if isinstance(val, np.ndarray) and val.ndim >= 2:
                    fname = f"{prefix}_{name}_{key}.nii.gz"
                    nib.Nifti1Image(val, self.affine).to_filename(fname)


if __name__ == "__main__":
    root = "cmbi_data0"
    case = "case01"
    
    data_path = f"{root}/{case}-qt2_reg.nii.gz"
    te_path   = f"{root}/{case}-TEs.txt"
    mask_path = f"{root}/{case}-mask.nii.gz"


    analyser = T2Analyser(data_path, te_path, mask_path)
    analyser.data[~analyser.mask] = 0 

    print(f"--- Processing {case} from {root} ---")

    analyser.fit_analytic(echo_indices=[0, 1])
    analyser.fit_lls()        # Linear Least Squares
    analyser.fit_wls()        # Weighted Linear Least Squares
    analyser.fit_nnls()       # Non-Negative (Bounded) LS
    analyser.fit_nlls_mono()  # Non-Linear (1-compartment)
    

    
    t2f_grid = [10, 25, 40]   # Low, Mid, High for Myelin water
    t2s_grid = [60, 150, 400] # Low, Mid, High for Intra/Extra water & CSF
    # t2f_grid = [40]   # Low, Mid, High for Myelin water
    # t2s_grid = [400] # Low, Mid, High for Intra/Extra water & CSF

    # Grid Search
    for gf in t2f_grid:
        for gs in t2s_grid:
            print(f"\n--- Testing Bi-Fit: Initial Fast={gf}ms, Slow={gs}ms ---")
            suffix = f"_F{gf}_S{gs}"
            
            analyser.fit_nlls_bi(
                t2_fast_guess=gf, 
                t2_slow_guess=gs, 
                suffix=suffix
            )
            analyser.calculate_maps_aic(bi_key=f"NLLS_Bi{suffix}")

    seg_path = f"{root}/{case}-seg.nii.gz"
    analyser.roi_analysis(seg_path)

    analyser.show_results(save_prefix=f"{root}/{case}_grid")
    analyser.save_all(prefix=f"{root}/{case}")
    