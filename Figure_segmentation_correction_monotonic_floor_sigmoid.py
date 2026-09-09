"""Segmentation and ECDF knee Correction of Spectral Signatures.
"""

from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from Functions.correct_spectrogram import apply_monotonic_floor_sigmoid, correct_by_logit_anchored_sigmoid
from Functions.utils import get_ecdf, compute_sef95


# ======================================================================================================================================================================
#                                                       Single example
# ======================================================================================================================================================================

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
seg_mask, psd, baseline, T_low, T_high = segment_blobs(M, f_M, f_int, factor_high=2, factor_low=1.5)
seg_bool = seg_mask.astype(bool)
# Extended mask via morphological dilation (2 iterations)
expanded_mask = binary_dilation(seg_bool, iterations=2)

# --- Correction
I_out, ref_vals = apply_monotonic_floor_sigmoid(I_orig, seg_bool)
I_out, I_mono, ref_vals = correct_by_logit_anchored_sigmoid(I_orig, seg_bool)

# --- Compute ECDFs
x_in, y_in = get_ecdf(I_orig[seg_bool])
x_out, y_out = get_ecdf(I_out[seg_bool])
x_mono, y_mono = get_ecdf(I_mono[seg_bool])
# Both reference distributions
x_ref_out, y_ref_out = get_ecdf(ref_vals)


# --- Display Spectrogram corrections
fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
vmin = np.percentile(I_orig, 5)
vmax = np.percentile(I_orig, 99.5)
plot_configs = [
    (axes[0],I_orig,'Original Spectrogram (≥ 20 Hz)',seg_bool,None),
    (axes[1],I_out,'Alpha-Blended ECDF Correction (Standard Mask)',seg_bool,None),
]

for ax, data, title, c_mask, ext_c_mask in plot_configs:
  pcm = ax.pcolormesh(t_spectro, f_M, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
  # Primary segmented blob contour
  ax.contour(t_spectro, f_M, c_mask, colors='black', linewidths=0.8)
  # Extended boundary contour
  if ext_c_mask is not None:
    ax.contour(t_spectro, f_M, ext_c_mask, colors='white', linewidths=1.5, linestyles='--')
  ax.set_title(title, fontsize=11)
  ax.set_ylabel('Frequency (Hz)')
  ax.sharex(axes[0])
axes[1].set_xlabel('Time (s)')
cbar = fig.colorbar(pcm, ax=axes, orientation='vertical', pad=0.02, fraction=0.04, label=r'$\log_2(\text{Power})$')

axes[2].plot(np.mean(I_orig, axis = -1))
axes[2].plot(np.mean(I_out, axis = -1))
axes[2].plot(np.mean(I_mono, axis = -1))

plt.show()


# --- Display ECDFs
fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

ax.plot(x_ref_out,y_ref_out,color='gray',linestyle=':',linewidth=2.0,label='Reference Ring (Standard Mask)')
ax.plot(x_in, y_in, 'r-', linewidth=2.0, label='Original Inside Blobs')
ax.plot(x_out,y_out,color='tab:blue',linestyle='-',linewidth=1.8,label='Corrected: monotonic floor sigmoid')
ax.plot(x_mono,y_mono,color='tab:blue',linestyle='-',linewidth=1.8,label='Corrected: monotonic floor')

ax.set_title('ECDF Comparison Inside Detected Blobs', fontsize=12, fontweight='bold')
ax.set_xlabel(r'Intensity ($\log_2\text{ Power}$)', fontsize=11)
ax.set_ylabel('Empirical Cumulative Probability', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', frameon=True, fontsize=10)

plt.show()



# ======================================================================================================================================================================
#                                                      Evaluation on multiple generated curves
# ======================================================================================================================================================================



rmse_errors = []
sef_orig_list = []
sef_corr_list = []
psd_first = {}
psd_last = {}

# --- run routine on multiple genrated signals
factors = np.linspace(1.0, 2.0, 10)  # 10 evenly spaced factor values from 1.0 to 2.0 to apply to the beta bump
for k, f_val in enumerate(factors):
  current_factors = [1, 1, float(f_val)]

  # Generate signal and full spectrogram
  t_k, y_k = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, current_factors)
  f_full, t_full, spectro_full = spectrogram(y_k, fs, nfft_factor=2)

  # Full raw PSD profile across time
  psd_full_orig = np.median(spectro_full, axis=1)

  # Focus on >= 20 Hz
  mask_f20 = f_full >= 20
  f_eval = f_full[mask_f20]
  M_eval = spectro_full[mask_f20, :].copy()
  I_orig_eval = np.log2(M_eval + 1e-11)

  # Segmentation
  seg_mask_k, psd_k, baseline_k, _, _ = segment_blobs(M_eval, f_eval, f_int, lam = 1e4, p = 0.01)
  seg_bool_k = seg_mask_k.astype(bool)
  expanded_mask_k = binary_dilation(seg_bool_k, iterations=2)

  # Apply correction using Extended Mask
  if np.any(expanded_mask_k):
    I_corr_eval, _ = apply_monotonic_floor_sigmoid(I_orig_eval, expanded_mask_k)
  else:
    I_corr_eval = I_orig_eval.copy()

  # Reconstruct full spectrogram with corrected >= 20 Hz region
  spectro_full_corr = spectro_full.copy()
  spectro_full_corr[mask_f20, :] = np.clip(2**I_corr_eval - 1e-11, 0, None)
  psd_full_corr = np.median(spectro_full_corr, axis=1)

  # Evaluate reconstruction error inside f_int = [25, 35] Hz
  mask_int = (f_eval >= f_int[0]) & (f_eval <= f_int[-1])
  psd_corr_eval = np.median(spectro_full_corr[mask_f20, :], axis=1)

  # Root Mean Square Error (in log2 domain against Whittaker baseline)
  rmse = np.sqrt(np.mean((np.log2(psd_corr_eval[mask_int] + 1e-11)- np.log2(baseline_k[mask_int] + 1e-11))** 2))
  rmse_errors.append(rmse)

  # Calculate SEF95 on full (uncropped) frequency spectrum
  sef_orig_list.append(compute_sef95(f_full, psd_full_orig))
  sef_corr_list.append(compute_sef95(f_full, psd_full_corr))

  # Save first and last profiles for plotting
  if k == 0:
    psd_first = {
        'f_full': f_full,
        'orig': psd_full_orig,
        'corr': psd_full_corr,
        'f_eval': f_eval,
        'baseline': baseline_k,
    }
  elif k == len(factors) - 1:
    psd_last = {
        'f_full': f_full,
        'orig': psd_full_orig,
        'corr': psd_full_corr,
        'f_eval': f_eval,
        'baseline': baseline_k,
    }

# --- Display first and last example PSD before/after from generated signals
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

# First condition (Factor = 1.00)
axes[0].plot(psd_first['f_full'],np.log2(psd_first['orig'] + 1e-11),'r-',label='Original PSD')
axes[0].plot(psd_first['f_full'],np.log2(psd_first['corr'] + 1e-11),'b--',label='Corrected PSD (Ext. Mask)')
axes[0].plot(psd_first['f_eval'],np.log2(psd_first['baseline'] + 1e-11),'k:',linewidth=2,label='Target Baseline')
axes[0].axvspan(f_int[0], f_int[1], color='gray', alpha=0.15, label='Target ROI')
axes[0].set_xlim([0, 50])
axes[0].set_title(f'PSD: Bump Factor = {factors[0]:.2f}')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel(r'$\log_2(\text{PSD})$')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='upper right', fontsize=9)

# Last condition (Factor = 2.00)
axes[1].plot(psd_last['f_full'],np.log2(psd_last['orig'] + 1e-11),'r-',label='Original PSD')
axes[1].plot(psd_last['f_full'],np.log2(psd_last['corr'] + 1e-11),'b--',label='Corrected PSD (Ext. Mask)')
axes[1].plot(psd_last['f_eval'],np.log2(psd_last['baseline'] + 1e-11),'k:',linewidth=2,label='Target Baseline')
axes[1].axvspan(f_int[0], f_int[1], color='gray', alpha=0.15, label='Target ROI')
axes[1].set_xlim([0, 50])
axes[1].set_title(f'PSD: Bump Factor = {factors[-1]:.2f}')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel(r'$\log_2(\text{PSD})$')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='upper right', fontsize=9)

plt.show()



# --- Display reconstruction error
fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

ax.plot(factors,rmse_errors,'o-',color='tab:blue',linewidth=2,markersize=6,label='Reconstruction RMSE')
ax.set_title(r'Reconstruction Error vs. Bump Factor within $[25, 35]$ Hz',fontweight='bold')
ax.set_xlabel('30 Hz Bump Scaling Factor')
ax.set_ylabel(r'RMSE ($\log_2\text{ PSD vs. Baseline}$)')
ax.set_xticks(factors)
ax.set_xticklabels([f'{f:.2f}' for f in factors])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left')

plt.show()


# --- Display SEF correction
fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

ax.plot(factors,sef_orig_list,'s--',color='tab:red',linewidth=2,markersize=6,label='Original Signal (Uncropped)')
ax.plot(factors,sef_corr_list,'o-',color='tab:green',linewidth=2,markersize=6,label='Corrected Signal (Ext. Mask)')

ax.set_title(r'Spectral Edge Frequency ($\text{SEF}_{95}$) across Bump Factors',fontweight='bold')
ax.set_xlabel('30 Hz Bump Scaling Factor')
ax.set_ylabel(r'$\text{SEF}_{95}$ (Hz)')
ax.set_xticks(factors)
ax.set_xticklabels([f'{f:.2f}' for f in factors])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='best')

plt.show()