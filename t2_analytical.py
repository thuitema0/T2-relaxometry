import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

'''
This script implements the analytic two-echo T2 estimation method. It uses the 
first two echoes from the multi-echo dataset to compute S0 and T2 maps. The choice 
of which echoes to use can significantly affect the variance and bias of the estimates, 
so we can experiment with different pairs of echoes (e.g., 1&3, 2&4) to see how it 
impacts the results.

Widely separated TEs leads to better conditioning but 
more noise sensitivity; closely spaced TEs → less sensitivity to T2 
(higher variance/bias). Try different idx pairs and compare.
'''

def _guard_positive(s, eps=1e-6):
    '''Ensure values are positive to avoid log issues.'''
    return np.maximum(s, eps)

def analytic_two_echo(S, TEs):
    """S: (..., 2), TEs: (2,)"""
    S = _guard_positive(S)
    TE1, TE2 = TEs
    S1, S2 = S[..., 0], S[..., 1]
    ratio = np.clip(S1 / S2, 1e-6, 1e6)
    denom = np.log(ratio)
    denom = np.where(np.abs(denom) < 1e-8, np.sign(denom) * 1e-8 + 1e-12, denom)
    T2 = (TE2 - TE1) / denom
    S0 = S1 * np.exp(TE1 / T2)
    return S0, T2



root = "data"
data = nib.load(f"{root}/case01-qt2_reg.nii.gz").get_fdata()  # (X,Y,Z,E)
TEs  = np.loadtxt(f"{root}/case01-TEs.txt")
# choose any two echoes (example: first two)
idx = [0, 1]
S0_2, T2_2 = analytic_two_echo(data[..., idx], TEs[idx])
nib.Nifti1Image(T2_2, affine=np.eye(4)).to_filename("case01-T2-analytic2.nii.gz")
nib.Nifti1Image(S0_2, affine=np.eye(4)).to_filename("case01-S0-analytic2.nii.gz")
print("Saved analytic 2-echo maps. Echo choice strongly affects variance/bias.")



# Show the images:
slice_idx = 50  # slice index

# Display T2 and S0 maps for the chosen slice
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(T2_2[:, :, slice_idx], cmap='viridis')
plt.title("T2 Map (Analytic 2 Echo)")
plt.colorbar()
# Use percentiles within mask (if available) for a sensible display range
vmin, vmax = np.percentile(T2_2[np.isfinite(T2_2)], [5, 95])
plt.clim(vmin, vmax)

plt.subplot(1, 2, 2)
plt.imshow(S0_2[:, :, slice_idx], cmap='viridis')
plt.title("S0 Map (Analytic 2 Echo)")
plt.colorbar()
# Use percentiles within mask (if available) for a sensible display range
vmin, vmax = np.percentile(S0_2[np.isfinite(S0_2)], [5, 95])
plt.clim(vmin, vmax)

plt.show()

