"""Segmentation and Alpha-Blended ECDF Correction of Spectral Signatures.

Comparison of Standard Mask vs. Extended Mask correction with ECDF analysis.
"""

from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
from scipy.ndimage import binary_dilation

# ==============================================================================
# 1. Simulate EEG Signal & Compute Spectrogram
# ==============================================================================
T = 20  # Duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 1]

t, y = get_mixed_OU_signals(
    T, dt, lbda_list, omega_list, sigma_list, factor_list
)

f_spectro, t_spectro, spectro = spectrogram(y, fs, nfft_factor=2)

# Focus strictly on frequencies >= 20 Hz
mask_f = f_spectro >= 20
f_M = f_spectro[mask_f]
M = spectro[mask_f, :]
spectro_log = np.log2(M + 1e-11)
I_orig = spectro_log.copy()

# ==============================================================================
# 2. Segmentation & Mask Expansion
# ==============================================================================
f_int = [25, 35]
seg_mask, psd, baseline, T_low, T_high = segment_blobs(M, f_M, f_int)
seg_bool = seg_mask.astype(bool)

# Extended mask via morphological dilation (2 iterations)
expanded_mask = binary_dilation(seg_bool, iterations=2)


# ==============================================================================
# 3. Alpha-Blended Local ECDF Correction Function
# ==============================================================================
def apply_alpha_blended_ecdf(I_in, mask, dilation_iter=2, alpha_cutoff=0.8):
  """Applies local ECDF matching with distance-transform alpha blending.

  Parameters:
      I_in : 2D ndarray, input spectrogram in log scale
      mask : 2D bool ndarray, target region to correct
      dilation_iter : int, border expansion steps to form reference ring
      alpha_cutoff : float, distance scaling factor for transition ring
  """
  # Construct local outside reference ring
  dilated = binary_dilation(mask, iterations=dilation_iter)
  border_ring = dilated & ~mask

  outside_vals = np.sort(I_in[border_ring])
  inside_vals = I_in[mask]

  # Quantile-to-quantile transfer (ECDF mapping)
  N_in = len(inside_vals)
  N_out = len(outside_vals)
  q_in = np.linspace(0, 1, N_in)
  q_out = np.linspace(0, 1, N_out)

  ecdf_map = interp1d(
      np.sort(inside_vals),
      np.interp(q_in, q_out, outside_vals),
      bounds_error=False,
      fill_value='extrapolate',
  )
  I_ecdf_direct = ecdf_map(inside_vals)

  # Distance transform from mask edge inward
  dist = ndi.distance_transform_edt(mask)
  d_max = np.max(dist) if np.max(dist) > 0 else 1.0
  alpha_weight = np.clip(dist / (d_max * alpha_cutoff + 1e-6), 0.0, 1.0)

  # Alpha blend: 0 at boundary (retaining exterior values), 1 at core
  I_out = I_in.copy()
  I_out[mask] = (1.0 - alpha_weight[mask]) * inside_vals + alpha_weight[
      mask
  ] * I_ecdf_direct
  return I_out, outside_vals


# Apply to standard mask
I_alpha_std, ref_vals_std = apply_alpha_blended_ecdf(
    I_orig, seg_bool, dilation_iter=2, alpha_cutoff=0.75
)

# Apply to extended mask
I_alpha_ext, ref_vals_ext = apply_alpha_blended_ecdf(
    I_orig, expanded_mask, dilation_iter=2, alpha_cutoff=0.80
)


# ==============================================================================
# 4. Figure 1: Spectrogram Comparison
# ==============================================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

vmin = np.percentile(I_orig, 5)
vmax = np.percentile(I_orig, 99.5)

plot_configs = [
    (
        axes[0],
        I_orig,
        'Original Spectrogram (≥ 20 Hz)',
        seg_bool,
        None,
    ),
    (
        axes[1],
        I_alpha_std,
        'Alpha-Blended ECDF Correction (Standard Mask)',
        seg_bool,
        None,
    ),
    (
        axes[2],
        I_alpha_ext,
        'Alpha-Blended ECDF Correction (Extended Mask)',
        seg_bool,
        expanded_mask,
    ),
]

for ax, data, title, c_mask, ext_c_mask in plot_configs:
  pcm = ax.pcolormesh(
      t_spectro, f_M, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax
  )
  # Primary segmented blob contour
  ax.contour(t_spectro, f_M, c_mask, colors='black', linewidths=0.8)
  # Extended boundary contour
  if ext_c_mask is not None:
    ax.contour(
        t_spectro,
        f_M,
        ext_c_mask,
        colors='white',
        linewidths=1.5,
        linestyles='--',
    )
  ax.set_title(title, fontsize=11)
  ax.set_ylabel('Frequency (Hz)')
  ax.sharex(axes[0])

axes[2].set_xlabel('Time (s)')

cbar = fig.colorbar(
    pcm,
    ax=axes,
    orientation='vertical',
    pad=0.02,
    fraction=0.04,
    label=r'$\log_2(\text{Power})$',
)
plt.show()


# ==============================================================================
# 5. Figure 2: ECDF Statistical Distribution Comparison
# ==============================================================================
def get_ecdf(vals):
  x = np.sort(vals.ravel())
  y = np.linspace(0, 1, len(x))
  return x, y

fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

# ECDFs within original core blob region (seg_bool)
x_in, y_in = get_ecdf(I_orig[seg_bool])
x_std, y_std = get_ecdf(I_alpha_std[seg_bool])
x_ext, y_ext = get_ecdf(I_alpha_ext[seg_bool])

# Both reference distributions
x_ref_std, y_ref_std = get_ecdf(ref_vals_std)
x_ref_ext, y_ref_ext = get_ecdf(ref_vals_ext)

ax.plot(
    x_ref_std,
    y_ref_std,
    color='gray',
    linestyle=':',
    linewidth=2.0,
    label='Reference Ring (Standard Mask)',
)
ax.plot(
    x_ref_ext,
    y_ref_ext,
    color='black',
    linestyle='--',
    linewidth=2.0,
    label='Reference Ring (Extended Mask)',
)

ax.plot(x_in, y_in, 'r-', linewidth=2.0, label='Original Inside Blobs')
ax.plot(
    x_std,
    y_std,
    color='tab:orange',
    linestyle='-',
    linewidth=1.8,
    label='Corrected: Alpha-Blended (Standard Mask)',
)
ax.plot(
    x_ext,
    y_ext,
    color='tab:blue',
    linestyle='-',
    linewidth=1.8,
    label='Corrected: Alpha-Blended (Extended Mask)',
)

ax.set_title(
    'ECDF Comparison Inside Detected Blobs', fontsize=12, fontweight='bold'
)
ax.set_xlabel(r'Intensity ($\log_2\text{ Power}$)', fontsize=11)
ax.set_ylabel('Empirical Cumulative Probability', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', frameon=True, fontsize=10)

plt.show()


# ==============================================================================
# 6. Multi-Factor Benchmark: 30 Hz Bump Factor Sweep (1.0 to 2.0)
# ==============================================================================

factors = np.linspace(1.0, 2.0, 10)  # 10 evenly spaced factor values from 1.0 to 2.0
rmse_errors = []
sef_orig_list = []
sef_corr_list = []

psd_first = {}
psd_last = {}


def compute_sef95(freqs, psd_profile):
  """Computes Spectral Edge Frequency (95%) from a 1D PSD profile."""
  cum_power = np.cumsum(psd_profile)
  total_power = cum_power[-1]
  if total_power <= 0:
    return freqs[0]
  idx = np.searchsorted(cum_power, 0.95 * total_power)
  idx = min(idx, len(freqs) - 1)
  return freqs[idx]


for k, f_val in enumerate(factors):
  current_factors = [1, 1, float(f_val)]

  # Generate signal and full spectrogram
  t_k, y_k = get_mixed_OU_signals(
      T, dt, lbda_list, omega_list, sigma_list, current_factors
  )
  f_full, t_full, spectro_full = spectrogram(y_k, fs, nfft_factor=2)

  # Full raw PSD profile across time
  psd_full_orig = np.median(spectro_full, axis=1)

  # Focus on >= 20 Hz
  mask_f20 = f_full >= 15
  f_eval = f_full[mask_f20]
  M_eval = spectro_full[mask_f20, :].copy()
  I_orig_eval = np.log2(M_eval + 1e-11)

  # Segmentation
  seg_mask_k, psd_k, baseline_k, _, _ = segment_blobs(M_eval, f_eval, f_int, lam = 1e4, p = 0.01)
  seg_bool_k = seg_mask_k.astype(bool)
  expanded_mask_k = binary_dilation(seg_bool_k, iterations=2)

  # Apply correction using Extended Mask
  if np.any(expanded_mask_k):
    I_corr_eval, _ = apply_alpha_blended_ecdf(
        I_orig_eval, expanded_mask_k, dilation_iter=2, alpha_cutoff=0.80
    )
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
  rmse = np.sqrt(
      np.mean(
          (
              np.log2(psd_corr_eval[mask_int] + 1e-11)
              - np.log2(baseline_k[mask_int] + 1e-11)
          )
          ** 2
      )
  )
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


# ==============================================================================
# 7. Figure 3: First vs. Last PSD Comparison
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

# First condition (Factor = 1.00)
axes[0].plot(
    psd_first['f_full'],
    np.log2(psd_first['orig'] + 1e-11),
    'r-',
    label='Original PSD',
)
axes[0].plot(
    psd_first['f_full'],
    np.log2(psd_first['corr'] + 1e-11),
    'b--',
    label='Corrected PSD (Ext. Mask)',
)
axes[0].plot(
    psd_first['f_eval'],
    np.log2(psd_first['baseline'] + 1e-11),
    'k:',
    linewidth=2,
    label='Target Baseline',
)
axes[0].axvspan(f_int[0], f_int[1], color='gray', alpha=0.15, label='Target ROI')
axes[0].set_xlim([0, 50])
axes[0].set_title(f'PSD: Bump Factor = {factors[0]:.2f}')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel(r'$\log_2(\text{PSD})$')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='upper right', fontsize=9)

# Last condition (Factor = 2.00)
axes[1].plot(
    psd_last['f_full'],
    np.log2(psd_last['orig'] + 1e-11),
    'r-',
    label='Original PSD',
)
axes[1].plot(
    psd_last['f_full'],
    np.log2(psd_last['corr'] + 1e-11),
    'b--',
    label='Corrected PSD (Ext. Mask)',
)
axes[1].plot(
    psd_last['f_eval'],
    np.log2(psd_last['baseline'] + 1e-11),
    'k:',
    linewidth=2,
    label='Target Baseline',
)
axes[1].axvspan(f_int[0], f_int[1], color='gray', alpha=0.15, label='Target ROI')
axes[1].set_xlim([0, 50])
axes[1].set_title(f'PSD: Bump Factor = {factors[-1]:.2f}')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel(r'$\log_2(\text{PSD})$')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='upper right', fontsize=9)

plt.show()


# ==============================================================================
# 8. Figure 4: Reconstruction Error (RMSE) across Bump Factors
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

ax.plot(
    factors,
    rmse_errors,
    'o-',
    color='tab:blue',
    linewidth=2,
    markersize=6,
    label='Reconstruction RMSE',
)
ax.set_title(
    r'Reconstruction Error vs. Bump Factor within $[25, 35]$ Hz',
    fontweight='bold',
)
ax.set_xlabel('30 Hz Bump Scaling Factor')
ax.set_ylabel(r'RMSE ($\log_2\text{ PSD vs. Baseline}$)')
ax.set_xticks(factors)
ax.set_xticklabels([f'{f:.2f}' for f in factors])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left')

plt.show()


# ==============================================================================
# 9. Figure 5: Spectral Edge Frequency (SEF95) Before vs. After
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

ax.plot(
    factors,
    sef_orig_list,
    's--',
    color='tab:red',
    linewidth=2,
    markersize=6,
    label='Original Signal (Uncropped)',
)
ax.plot(
    factors,
    sef_corr_list,
    'o-',
    color='tab:green',
    linewidth=2,
    markersize=6,
    label='Corrected Signal (Ext. Mask)',
)

ax.set_title(
    r'Spectral Edge Frequency ($\text{SEF}_{95}$) across Bump Factors',
    fontweight='bold',
)
ax.set_xlabel('30 Hz Bump Scaling Factor')
ax.set_ylabel(r'$\text{SEF}_{95}$ (Hz)')
ax.set_xticks(factors)
ax.set_xticklabels([f'{f:.2f}' for f in factors])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='best')

plt.show()