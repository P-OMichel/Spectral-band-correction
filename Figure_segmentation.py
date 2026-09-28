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

    return mask_f_int, mask_watershed, psd, psd_baseline, T_low, T_high, background_mask

# --- generate synthetic signal
T = 20  # Duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 1]

t, y = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

f_spectro, t_spectro, spectro = spectrogram(y, fs, nfft_factor=2)[:3]

# Focus strictly on frequencies >= 20 Hz
mask_f = f_spectro >= 20
f_M = f_spectro[mask_f]
M = spectro[mask_f, :]
spectro_log = np.log2(M + 1e-11)
I_orig = spectro_log.copy()

# --- Segmentation of the pixels to correct
f_int = [25, 35]
mask_f_int, seg_mask, psd, baseline, T_low, T_high, background_mask = segment_blobs(M, f_M, f_int)
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




import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# --- Helper to draw complete contours including outer image boundaries ---
def draw_closed_contour(ax, t, f, mask, **contour_kwargs):
    # Determine bin step sizes
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    df = f[1] - f[0] if len(f) > 1 else 1.0

    # Expand grid by 1 bin on each side
    t_padded = np.concatenate([[t[0] - dt], t, [t[-1] + dt]])
    f_padded = np.concatenate([[f[0] - df], f, [f[-1] + df]])

    # Pad mask with zeros at boundaries so edge transitions are preserved
    mask_padded = np.pad(mask.astype(float), pad_width=1, mode='constant', constant_values=0)

    # Shift coordinates by half a bin to align with pcolormesh cells
    t_grid = t_padded + dt / 2.0
    f_grid = f_padded + df / 2.0

    # Draw contour and clip tightly back to original spectrogram limits
    ax.contour(t_grid, f_grid, mask_padded, levels=[0.5], **contour_kwargs)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(f[0], f[-1])


# --- Set up master figure ---
fig = plt.figure(figsize=(14, 15))

# Outer grid: Row 0 (Overview + PSDs), Rows 1-3 (Steps 3, 4, 5)
# hspace increased for larger spacing between rows
outer_gs = gridspec.GridSpec(
    4, 1, figure=fig,
    height_ratios=[1.2, 1.0, 1.0, 1.0],
    hspace=0.6
)

# Row 0: Spectrogram (large) + PSD + Thresholds
gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer_gs[0],
    width_ratios=[2.2, 1.0, 1.0],
    wspace=0.25
)

ax_spec = fig.add_subplot(gs_top[0])
ax_step1 = fig.add_subplot(gs_top[1])
ax_step2 = fig.add_subplot(gs_top[2])

# Steps 3, 4, 5: Each has 2 columns (Mask | Spectrogram + Contour)
gs_step3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1], wspace=0.15)
ax_s3_l = fig.add_subplot(gs_step3[0])
ax_s3_r = fig.add_subplot(gs_step3[1])

gs_step4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[2], wspace=0.15)
ax_s4_l = fig.add_subplot(gs_step4[0])
ax_s4_r = fig.add_subplot(gs_step4[1])

gs_step5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[3], wspace=0.15)
ax_s5_l = fig.add_subplot(gs_step5[0])
ax_s5_r = fig.add_subplot(gs_step5[1])

# Percentiles for spectrogram color scaling
vmin = np.percentile(I_orig, 5)
vmax = np.percentile(I_orig, 99.5)


# ==========================================
# ROW 0: Overview & PSD Steps
# ==========================================
# Main Spectrogram
ax_spec.pcolormesh(t_spectro, f_M, I_orig, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
ax_spec.axhline(f_int[0], color='black', linestyle='--', linewidth=1.5)
ax_spec.axhline(f_int[-1], color='black', linestyle='--', linewidth=1.5)
ax_spec.text(t_spectro[-1] * 0.96, f_int[-1], r'$f_{upper}$', ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
ax_spec.text(t_spectro[-1] * 0.96, f_int[0], r'$f_{lower}$', ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
ax_spec.set_ylabel('Frequency (Hz)', fontsize=11)
ax_spec.set_title('Signal Spectrogram', fontsize=12)

# Step 1: Baseline
ax_step1.plot(f_M, psd, label='PSD', color='C0')
ax_step1.plot(f_M, baseline, label='Baseline', color='orange')
ax_step1.axvline(f_int[0], color='black', linestyle='--', linewidth=0.8)
ax_step1.axvline(f_int[-1], color='black', linestyle='--', linewidth=0.8)
ax_step1.set_xticks(f_int)
ax_step1.set_xticklabels([r'$f_{lower}$', r'$f_{upper}$'])
ax_step1.tick_params(axis='y', labelsize=8)
ax_step1.legend(loc='upper right', framealpha=0.6, fontsize=9)
ax_step1.set_title('Step 1: Baseline estimation', fontsize=11)

# Step 2: Thresholds
ax_step2.plot(f_M, psd, color='C0', alpha=0.5, linewidth=1)
ax_step2.plot(f_M, baseline, color='orange', alpha=0.7, linewidth=1)
ax_step2.plot(f_M, T_low, label=r'$T_{background}$', color='limegreen', linewidth=1.5)
ax_step2.plot(f_M, T_high, label=r'$T_{high}$', color='darkred', linewidth=1.5)
ax_step2.axvline(f_int[0], color='black', linestyle='--', linewidth=0.8)
ax_step2.axvline(f_int[-1], color='black', linestyle='--', linewidth=0.8)
ax_step2.set_xticks(f_int)
ax_step2.set_xticklabels([r'$f_{lower}$', r'$f_{upper}$'])
ax_step2.tick_params(axis='y', labelsize=8)
ax_step2.legend(loc='upper right', framealpha=0.6, fontsize=9)
ax_step2.set_title('Step 2: Thresholds determination', fontsize=11)


# ==========================================
# ROWS 1-3: Masks & Contours
# ==========================================
step_panels = [
    (ax_s3_l, ax_s3_r, mask_f_int.astype(float),
     r'Mask of high intensity pixels within $f_{int}$',
     'Spectrogram with mask contour'),
    
    (ax_s4_l, ax_s4_r, background_mask.astype(float),
     'Mask of background pixels',
     'Spectrogram with background mask contour'),
    
    (ax_s5_l, ax_s5_r, seg_bool.astype(float),
     'Mask after expansion',
     'Spectrogram with expanded mask contour')
]

for ax_l, ax_r, mask_data, title_l, title_r in step_panels:
    # Left: Binary Mask
    ax_l.pcolormesh(t_spectro, f_M, mask_data, shading='nearest', cmap='jet')
    ax_l.set_title(title_l, fontsize=10)
    
    # Right: Spectrogram + Boundary Contour
    ax_r.pcolormesh(t_spectro, f_M, I_orig, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    draw_closed_contour(ax_r, t_spectro, f_M, mask_data, colors='black', linewidths=1.8)
    ax_r.set_title(title_r, fontsize=10)

plt.subplots_adjust(top=0.96, bottom=0.04, left=0.06, right=0.98)
plt.show()