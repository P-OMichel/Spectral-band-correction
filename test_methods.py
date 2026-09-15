"""Segmentation and ECDF knee Correction of Spectral Signatures.
"""

from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from Functions.correct_spectrogram import apply_spatially_conditioned_quantile_blend, apply_monotonic_quantile_blend, apply_monotonic_quantile_blend_1, apply_alpha_blended_ecdf, apply_rank_preserved_alpha_ecdf_1, apply_monotonic_floor_ecdf, apply_rank_preserved_alpha_ecdf
from Functions.utils import get_ecdf, compute_sef95
from Functions.utils import plot_rank_inversion_heatmap, plot_global_rank_inversion_heatmap, plot_local_boundary_inversion_heatmap

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
seg_mask, psd, baseline, T_low, T_high = segment_blobs(M, f_M, f_int)
seg_bool = seg_mask.astype(bool)
# Extended mask via morphological dilation (2 iterations)
expanded_mask = binary_dilation(seg_bool, iterations=2)

# --- Correction
I_a_blend, ref_vals_a_blend = apply_monotonic_floor_ecdf(I_orig, seg_bool) # apply_spatially_conditioned_quantile_blend(I_orig, expanded_mask) # apply_rank_preserved_alpha_ecdf(I_orig, expanded_mask) #apply_alpha_blended_ecdf(I_orig, expanded_mask)
I_m_q_blend, ref_vals_m_q_blend = apply_monotonic_floor_ecdf(I_orig, expanded_mask) #apply_monotonic_quantile_blend(I_orig, expanded_mask) #apply_monotonic_quantile_blend(I_orig, expanded_mask) #apply_rank_preserved_alpha_ecdf(I_orig, expanded_mask)#apply_monotonic_quantile_blend(I_orig, seg_bool)

plot_rank_inversion_heatmap(I_orig, I_a_blend, expanded_mask)
plot_rank_inversion_heatmap(I_orig, I_m_q_blend, expanded_mask)

plot_global_rank_inversion_heatmap(I_orig, I_a_blend, expanded_mask)
plot_global_rank_inversion_heatmap(I_orig, I_m_q_blend, expanded_mask)

plot_local_boundary_inversion_heatmap(I_orig, I_a_blend, expanded_mask)
plot_local_boundary_inversion_heatmap(I_orig, I_m_q_blend, expanded_mask)

# --- Compute ECDFs
x_in, y_in = get_ecdf(I_orig[seg_bool])
x_a_blend, y_a_blend = get_ecdf(I_a_blend[seg_bool])
x_m_q_blend, y_m_q_blend = get_ecdf(I_m_q_blend[seg_bool])

# Both reference distributions
x_ref_out_a_blend, y_ref_out_a_blend = get_ecdf(ref_vals_a_blend)
x_ref_out_m_q_blend, y_ref_out_m_q_blend = get_ecdf(ref_vals_m_q_blend)


# --- Display Spectrogram corrections
fig, axes = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)
vmin = np.percentile(I_orig, 5)
vmax = np.percentile(I_orig, 99.5)
plot_configs = [
    (axes[0],I_orig,'Original Spectrogram (≥ 20 Hz)',seg_bool,None),
    (axes[1],I_a_blend,'Alpha-Blended ECDF Correction (Standard Mask)',seg_bool,None),
    (axes[2],I_m_q_blend,'Monotonic Quantile ECDF Correction (Standard Mask)',seg_bool,None),
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

axes[3].plot(np.mean(I_orig, axis = -1))
axes[3].plot(np.mean(I_a_blend, axis = -1))
axes[3].plot(np.mean(I_m_q_blend, axis = -1))

plt.show()


# --- Display ECDFs
fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

ax.plot(x_ref_out_a_blend,y_ref_out_a_blend,linewidth=2.0,label='Reference Ring (a blend)')
ax.plot(x_ref_out_m_q_blend,y_ref_out_m_q_blend,linewidth=2.0,label='Reference Ring (m q blend)')

ax.plot(x_in, y_in, 'r-', linewidth=2.0, label='Original Inside Blobs')
ax.plot(x_a_blend,y_a_blend,linewidth=1.8,label='Corrected: a blend')
ax.plot(x_m_q_blend,y_m_q_blend,linewidth=1.8,label='Corrected: m q blend')

ax.set_title('ECDF Comparison Inside Detected Blobs', fontsize=12, fontweight='bold')
ax.set_xlabel(r'Intensity ($\log_2\text{ Power}$)', fontsize=11)
ax.set_ylabel('Empirical Cumulative Probability', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', frameon=True, fontsize=10)

plt.show()