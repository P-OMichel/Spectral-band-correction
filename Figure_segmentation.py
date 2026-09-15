"""Segmentation and ECDF knee Correction of Spectral Signatures.
"""

from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import whittaker_als_baseline, label, watershed
from Functions.time_frequency import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation


def segment_blobs(M, f_M, f_int, factor_high=3, factor_low=2, average='mean', 
                  lam=1e4, p=0.01, max_iter=15, method='baseline', log_fit=True, eps=1e-12):
    '''
    Inputs:
    - M: spectrogram matrix (linear scale)
    - f_M: frequency vector for matrix M
    - f_int: frequency interval in which the signature is mostly contained
    - factor_high: multiplicative factor for high bins intensity
    - factor_low: multiplicative factor for background bins intensity
    - average: str --> whether mean or median is used to do the psd projection
    - log_fit: bool --> if True, baseline is fitted on log(psd) and converted back to linear
    - eps: float --> small offset to avoid log(0)
    '''
    # --- get PSD
    if average == 'mean':
        psd = np.mean(M, axis=1)
    elif average == 'median':
        psd = np.median(M, axis=1)
    else:
        raise ValueError("average mode is incorrect: select 'mean' or 'median'")

    # --- Get baseline of PSD
    if log_fit:
        # Fit baseline in log-space, then exponentiate back to linear
        log_psd = np.log(np.maximum(psd, eps))
        log_baseline = whittaker_als_baseline(log_psd, lam, p, max_iter)
        psd_baseline = np.exp(log_baseline)
    else:
        psd_baseline = whittaker_als_baseline(psd, lam, p, max_iter)

    # --- Get threshold above which blobs are considered high and under which pixels are background
    if method == 'baseline':
        T_high = factor_high * psd_baseline
        T_low = factor_low * psd_baseline
    elif method == 'quantile':
        med = np.quantile(M, 0.5)
        T_high = np.full_like(psd, factor_high * med)
        T_low = np.full_like(psd, factor_low * med)
    else:
        raise ValueError("method name is incorrect: select 'baseline' or 'quantile'")

    # --- get mask of high intensity bins
    mask = M > (T_high[:, np.newaxis])

    # --- refine mask to region within f_int
    freq_mask = (f_M >= f_int[0]) & (f_M <= f_int[-1])
    mask_f_int = mask * freq_mask[:, None]

    # --- Watershed extension
    structure = np.ones((3, 3), dtype=int)
    markers, num_features = label(mask_f_int, structure=structure)
    
    topography = -M
    background_mask = M < (T_low[:, np.newaxis])
    
    watershed_markers = markers.copy()
    watershed_markers[background_mask] = -1
    labels = watershed(topography, markers=watershed_markers, mask=~background_mask)
    
    mask_watershed = (labels > 0).astype(int)

    return mask_f_int, mask_watershed, psd, psd_baseline, T_low, T_high

# --- generate synthetic signal
T = 20  # Duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 1]

t, y = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

f_spectro, t_spectro, spectro = spectrogram(y, fs, nfft_factor=2)

# Focus strictly on frequencies >= 20 Hz
mask_f = f_spectro >= 20
f_M = f_spectro[mask_f]
M = spectro[mask_f, :]
spectro_log = np.log2(M + 1e-11)
I_orig = spectro_log.copy()

# --- Segmentation of the pixels to correct
f_int = [25, 35]
mask_f_int, seg_mask, psd, baseline, T_low, T_high = segment_blobs(M, f_M, f_int)
seg_bool = seg_mask.astype(bool)
# Extended mask via morphological dilation (2 iterations)
expanded_mask = binary_dilation(seg_bool, iterations=2)
border_ring = expanded_mask & ~seg_bool


fig, axis = plt.subplots(1, constrained_layout = True)
vmin = np.percentile(I_orig, 5)
vmax = np.percentile(I_orig, 99.5)
axis.pcolormesh(t_spectro, f_M, I_orig, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axis.set_title('Spectrogram', fontsize = 14)
axis.axhline(f_int[0], color = 'white', linestyle = '--', linewidth = 1.5)
axis.axhline(f_int[-1], color = 'white', linestyle = '--', linewidth = 1.5)

plt.show()

fig, axis = plt.subplots(1, constrained_layout = True)
axis.plot(psd, label = 'PSD')
axis.plot(baseline, label = 'Baseline')

plt.show()


fig, axes = plt.subplots(2, 2, constrained_layout = True)
axes[0,0].pcolormesh(t_spectro, f_M, mask_f_int, shading='nearest', cmap='jet')
axes[0,0].set_title(r'Mask of high intensity pixels within f_{int}')
axes[1,0].pcolormesh(t_spectro, f_M, I_orig, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes[1,0].contour(t_spectro, f_M, mask_f_int, linestyle = '-', colors='black', linewidths=2)
axes[0,1].pcolormesh(t_spectro, f_M, seg_bool, shading='nearest', cmap='jet')
axes[0,1].set_title('Mask after expansion')
axes[1,1].pcolormesh(t_spectro, f_M, I_orig, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes[1,1].contour(t_spectro, f_M, seg_bool, linestyle = '-', colors='black', linewidths=2)

plt.show()