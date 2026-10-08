# '''
# File to generate the introduction figure with:
# - contaminated time series
# - spectrogram
# - segmentation
# - correction
# - corrected time series 
# '''
# from Functions.generate_OU import get_mixed_OU_signals
# from Functions.segment_spectrogram import segment_blobs
# from Functions.time_frequency import spectrogram
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy as sc
# from scipy.ndimage import binary_dilation
# from Functions.correct_spectrogram import apply_rank_preserved_alpha_ecdf
# import matplotlib.gridspec as gridspec

# import scipy.signal as signal

# def istft_from_corrected_spectrogram(
#     spectro_corr,
#     stft_orig_sliced,
#     fs,
#     nperseg_factor=1,
#     noverlap_factor=0.9,
#     nfft_factor=1,
#     scaling="psd",
# ):
#     # 1. Recalculate parameters matching the forward STFT
#     nperseg = int(nperseg_factor * fs)
#     noverlap = int(noverlap_factor * nperseg)
#     nfft = int(nfft_factor * nperseg)
#     window = signal.windows.hamming(nperseg, sym=True)

#     # 2. Get corrected magnitude from power spectrogram (Sxx = |stft|^2)
#     # Ensure no small negative values before sqrt
#     mag_corr = np.sqrt(np.maximum(spectro_corr, 0))

#     # 3. Retrieve the original phase from the sliced STFT
#     phase_orig = np.angle(stft_orig_sliced)

#     # 4. Form the corrected complex STFT for the lower frequencies ([:j, :])
#     stft_corr_sliced = mag_corr * np.exp(1j * phase_orig)

#     # 5. Restore full frequency bins required by iSTFT: (nfft // 2) + 1
#     n_freq_bins_full = (nfft // 2) + 1
#     j, n_time_steps = stft_corr_sliced.shape

#     if j < n_freq_bins_full:
#         # Pad upper frequencies above f_cut with zeros
#         stft_corr_full = np.pad(
#             stft_corr_sliced,
#             pad_width=((0, n_freq_bins_full - j), (0, 0)),
#             mode="constant",
#             constant_values=0,
#         )
#     else:
#         stft_corr_full = stft_corr_sliced

#     # 6. Inverse STFT to get the corrected time series
#     t_corr, y_corr = signal.istft(
#         stft_corr_full,
#         fs=fs,
#         window=window,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         nfft=nfft,
#         scaling=scaling,
#     )

#     return stft_corr_full, t_corr, y_corr

# # --- generate synthetic signal
# T = 20  # Duration (s)
# dt = 0.001
# fs = 1 / dt

# lbda_list = [1, 2, 1]
# omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
# sigma_list = [3, 2, 2]
# factor_list = [1, 1, 1]

# t, y = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

# f_spectro, t_spectro, spectro, stft = spectrogram(y, fs, nfft_factor=2)

# # PSD 
# psd_spectro = np.mean(np.log2(spectro + 0.00000001), axis = 1) #sc.signal.savgol_filter(np.mean(np.log2(spectro + 0.00000001), axis = 0), 2, 1)


# # Focus strictly on frequencies >= 20 Hz
# mask_f = f_spectro >= 20
# f_M = f_spectro[mask_f]
# M = spectro[mask_f, :]
# spectro_log = np.log2(M + 1e-11)
# I_orig = spectro_log.copy()

# # --- Segmentation of the pixels to correct
# f_int = [25, 35]
# seg_mask, psd, baseline, T_low, T_high = segment_blobs(M, f_M, f_int)
# seg_bool = seg_mask.astype(bool)
# # Extended mask via morphological dilation (2 iterations)
# expanded_mask = binary_dilation(seg_bool, iterations=2)
# ref_mask = binary_dilation(expanded_mask, iterations=2)
# border_ring = ref_mask & ~expanded_mask

# # correction
# I_a_r_blend, ref_vals_a_r_blend = apply_rank_preserved_alpha_ecdf(I_orig, expanded_mask, alpha_cutoff=0.3)

# spectro_corr = spectro.copy()
# eps = 1e-11  
# I_a_r_blend_linear = np.exp2(I_a_r_blend) - eps
# spectro_corr[mask_f, :] = I_a_r_blend_linear

# psd_corr = np.mean(np.log2(spectro_corr + 0.00000001), axis = 1)

# # corrected signal
# stft_corr, t_corr, y_corr = istft_from_corrected_spectrogram(spectro_corr=spectro_corr, stft_orig_sliced=stft, fs=fs)

# fig = plt.figure(constrained_layout=True)
# # Define relative widths: small, medium, large
# w_small = 1
# w_med = 2
# w_large = 4
# # Outer layout: 2 rows, 1 column
# outer_gs = gridspec.GridSpec(2, 1, figure=fig)
# # Row 0: widths correspond to [0,0] (med), [0,1] (small), [0,2] (large)
# gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], width_ratios=[w_small, w_med, w_large])
# # Row 1: widths correspond to [1,0] (large), [1,1] (med), [1,2] (small)
# gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], width_ratios=[w_large, w_small, w_med])
# # Create the 2x3 axes array
# axes = [[fig.add_subplot(gs_row0[0, i]) for i in range(3)],[fig.add_subplot(gs_row1[0, i]) for i in range(3)]]

# v_min = np.quantile(spectro_log, 0.05)
# v_max= np.quantile(spectro_log, 0.95)
# axes[0][0].plot(y)
# axes[0][0].set_axis_off()
# axes[0][1].plot(f_spectro, psd_spectro)
# axes[0][2].pcolormesh(t_spectro, f_M, spectro_log, shading = 'gouraud', cmap = 'jet', vmin = v_min, vmax = v_max)
# axes[0][2].contour(t_spectro, f_M, expanded_mask, colors = 'black', linewidth = 2)

# axes[1][0].pcolormesh(t_spectro, f_M, I_a_r_blend, shading = 'gouraud', cmap = 'jet', vmin = v_min, vmax = v_max)
# axes[1][0].contour(t_spectro, f_M, expanded_mask, colors = 'black', linewidth = 2)
# axes[1][1].plot(t, y_corr)
# axes[1][1].set_axis_off()
# axes[1][2].plot(f_spectro, psd_corr)

# plt.show()




"""File to generate the introduction figures with:
- contaminated time series & bandpass comparisons
- spectrograms, masks, and corrected representations
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.ndimage import binary_dilation
from matplotlib.colors import ListedColormap
from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram
from Functions.correct_spectrogram import apply_rank_preserved_alpha_ecdf


def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Applies a zero-phase Butterworth bandpass filter."""
    sos = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data)


def istft_from_corrected_spectrogram(spectro_orig, spectro_corr, stft_orig, fs, nperseg_factor=1, noverlap_factor=0.9, nfft_factor=2, scaling="psd"):
    """Reconstructs the time-domain signal by applying a magnitude gain mask to the original STFT."""
    nperseg = int(nperseg_factor * fs)
    noverlap = int(noverlap_factor * nperseg)
    nfft = int(nfft_factor * nperseg)
    window = signal.windows.hamming(nperseg, sym=True)

    # 1. Element-wise magnitude gain mask preserving original phase and low frequencies
    gain = np.sqrt(np.maximum(spectro_corr, 0) / (spectro_orig + 1e-12))
    stft_corr = stft_orig * gain

    # 2. Invert using exact matching transform parameters
    t_corr, y_corr = signal.istft(stft_corr, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling=scaling)

    return stft_corr, t_corr, y_corr


# --- Generate synthetic signal
T = 20  # Duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 4]

t, y = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

# --- Generate synthetic aperiodic background with knee at 1 Hz and 1/f^chi decay
N = len(t)
freqs = np.fft.rfftfreq(N, d=dt)

# Knee spectrum model: P(f) = 1 / (k + f^chi), with knee frequency f_k = 1 Hz and exponent chi = 2
f_knee = 1.0
chi = 2.0
psd_aperiodic = 1.0 / (f_knee**chi + np.maximum(freqs, 1e-6) ** chi)

# Generate random phase white noise shaped by the aperiodic amplitude spectrum
np.random.seed(42)
phases = np.random.uniform(0, 2 * np.pi, len(freqs))
fft_aperiodic = np.sqrt(psd_aperiodic) * np.exp(1j * phases)
fft_aperiodic[0] = 0  # Remove DC component

aperiodic_signal = np.fft.irfft(fft_aperiodic, n=N)
aperiodic_signal = (aperiodic_signal / np.std(aperiodic_signal)) * 2.5  # Scale amplitude

# Combine the OU oscillatory peaks with the aperiodic knee background
y = y + aperiodic_signal

f_spectro, t_spectro, spectro, stft = spectrogram(y, fs, nfft_factor=2)

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
expanded_mask = binary_dilation(seg_bool, iterations=2)
ref_mask = binary_dilation(expanded_mask, iterations=2)
border_ring = ref_mask & ~expanded_mask

# --- Correction
I_a_r_blend, ref_vals_a_r_blend = apply_rank_preserved_alpha_ecdf(I_orig, expanded_mask, alpha_cutoff=0.3)

spectro_corr = spectro.copy()
eps = 1e-11
I_a_r_blend_linear = np.exp2(I_a_r_blend) - eps
spectro_corr[mask_f, :] = I_a_r_blend_linear

# --- Corrected time-domain signal
stft_corr, t_corr, y_corr = istft_from_corrected_spectrogram(spectro, spectro_corr, stft, fs, nfft_factor=2)

# Align lengths between iSTFT output and original vector
n_pts = min(len(t), len(y_corr))
t_plot = t[:n_pts]
y_orig_plot = y[:n_pts]
y_corr_plot = y_corr[:n_pts]

# --- Filtered signals for time-domain superposition
y_f_int_orig = bandpass_filter(y_orig_plot, fs, f_int[0], f_int[1])
y_f_int_corr = bandpass_filter(y_corr_plot, fs, f_int[0], f_int[1])
y_20_45 = bandpass_filter(y_orig_plot, fs, 20, 45)

# ==============================================================================
# FIGURE 1: Time-Domain Visualizations
# ==============================================================================
fig_time, axes_time = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)

# 1. Original signal + filtered in f_int
axes_time[0].plot(t_plot, y_orig_plot, color="gray", alpha=0.55, label="Original Signal")
axes_time[0].plot(t_plot, y_f_int_orig, color="crimson", linewidth=1.2, label=f"Component in f_int [{f_int[0]}-{f_int[1]} Hz]")
axes_time[0].set_title("Original Contaminated Signal vs. Target Band Component")
axes_time[0].set_ylabel("Amplitude")
axes_time[0].legend(loc="upper right")
axes_time[0].grid(True, linestyle="--", alpha=0.5)

# 2. Corrected signal + filtered in f_int
axes_time[1].plot(t_plot, y_corr_plot, color="teal", alpha=0.55, label="Corrected Signal")
axes_time[1].plot(t_plot, y_f_int_corr, color="crimson", linewidth=1.2, label=f"Component in f_int [{f_int[0]}-{f_int[1]} Hz]")
axes_time[1].set_title("Corrected Signal vs. Residual Target Band Component")
axes_time[1].set_ylabel("Amplitude")
axes_time[1].legend(loc="upper right")
axes_time[1].grid(True, linestyle="--", alpha=0.5)

# 3. Band [20, 45] Hz + component in f_int
axes_time[2].plot(t_plot, y_20_45, color="steelblue", alpha=0.6, label="Band [20 - 45 Hz]")
axes_time[2].plot(t_plot, y_f_int_orig, color="darkred", linewidth=1.2, label=f"Component in f_int [{f_int[0]}-{f_int[1]} Hz]")
axes_time[2].set_title(f"Signal in [20, 45 Hz] vs. f_int [{f_int[0]}-{f_int[1]} Hz]")
axes_time[2].set_xlabel("Time (s)")
axes_time[2].set_ylabel("Amplitude")
axes_time[2].legend(loc="upper right")
axes_time[2].grid(True, linestyle="--", alpha=0.5)

fig_time.tight_layout()

# ==============================================================================
# FIGURE 2: Spectrogram and Mask Visualizations
# ==============================================================================
fig_spec, axes_spec = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)

v_min = np.quantile(spectro_log, 0.05)
v_max = np.quantile(spectro_log, 0.95)

# Common overlays
mask_overlay = np.where(expanded_mask, 1.0, np.nan)
ring_overlay = np.where(border_ring, 1.0, np.nan)

# 1. Raw Log Spectrogram
mesh0 = axes_spec[0, 0].pcolormesh(t_spectro, f_M, spectro_log, shading="gouraud", cmap="jet", vmin=v_min, vmax=v_max)
axes_spec[0, 0].set_title("Log Spectrogram (Original, ≥ 20 Hz)")
axes_spec[0, 0].set_ylabel("Frequency (Hz)")

# 2. Spectrogram + Segmentation Mask (white overlay, thicker black contour)
axes_spec[0, 1].pcolormesh(t_spectro, f_M, spectro_log, shading="gouraud", cmap="jet", vmin=v_min, vmax=v_max)
axes_spec[0, 1].pcolormesh(t_spectro, f_M, mask_overlay, cmap=ListedColormap(["white"]), alpha=0.35, shading="gouraud", zorder=2)
axes_spec[0, 1].contour(t_spectro, f_M, expanded_mask, levels=[0.5], colors="black", linewidths=2.8, zorder=3)
axes_spec[0, 1].set_title("Spectrogram + Dilation Mask")

# 3. Spectrogram + Dilation Mask + Border Ring (thicker contours, white mask overlay, cyan ring overlay)
# axes_spec[1, 0].pcolormesh(t_spectro, f_M, spectro_log, shading="gouraud", cmap="jet", vmin=v_min, vmax=v_max)
# # Mask overlay (white) & contour (black, thick)
# axes_spec[1, 0].pcolormesh(t_spectro, f_M, mask_overlay, cmap=ListedColormap(["white"]), alpha=0.35, shading="gouraud", zorder=2)
# axes_spec[1, 0].contour(t_spectro, f_M, expanded_mask, levels=[0.5], colors="black", linewidths=2.8, zorder=3)
# # Border ring overlay (cyan) & contour (cyan, thick)
# axes_spec[1, 0].pcolormesh(t_spectro, f_M, ring_overlay, cmap=ListedColormap(["cyan"]), alpha=0.35, shading="gouraud", zorder=2)
# axes_spec[1, 0].contour(t_spectro, f_M, border_ring, levels=[0.5], colors="cyan", linewidths=2.8, zorder=3)
# axes_spec[1, 0].set_title("Spectrogram + Mask & Border Ring")
# axes_spec[1, 0].set_xlabel("Time (s)")
# axes_spec[1, 0].set_ylabel("Frequency (Hz)")

# 3. Spectrogram + Dilation Mask + Border Ring
axes_spec[1, 0].pcolormesh(t_spectro, f_M, spectro_log, shading="gouraud", cmap="jet", vmin=v_min, vmax=v_max)
# Overlays: mask (white) and ring (cyan)
axes_spec[1, 0].pcolormesh(t_spectro, f_M, mask_overlay, cmap=ListedColormap(["white"]), alpha=0.35, shading="gouraud", zorder=2)
axes_spec[1, 0].pcolormesh(t_spectro, f_M, ring_overlay, cmap=ListedColormap(["cyan"]), alpha=0.35, shading="gouraud", zorder=2)
# Border ring contour in white
axes_spec[1, 0].contour(t_spectro, f_M, border_ring, levels=[0.5], colors="white", linewidths=2.8, zorder=3)
# Segmentation mask boundary drawn explicitly on top in black
axes_spec[1, 0].contour(t_spectro, f_M, expanded_mask, levels=[0.5], colors="black", linewidths=2.8, zorder=4)
axes_spec[1, 0].set_title("Spectrogram + Mask & Border Ring")
axes_spec[1, 0].set_xlabel("Time (s)")
axes_spec[1, 0].set_ylabel("Frequency (Hz)")

# 4. Corrected Spectrogram
mesh3 = axes_spec[1, 1].pcolormesh(t_spectro, f_M, I_a_r_blend, shading="gouraud", cmap="jet", vmin=v_min, vmax=v_max)
axes_spec[1, 1].contour(t_spectro, f_M, expanded_mask, levels=[0.5], colors="black", linewidths=2.8, zorder=3)
axes_spec[1, 1].set_title("Corrected Spectrogram")
axes_spec[1, 1].set_xlabel("Time (s)")

fig_spec.subplots_adjust(right=0.88)
cbar_ax = fig_spec.add_axes([0.90, 0.15, 0.02, 0.7])
fig_spec.colorbar(mesh0, cax=cbar_ax, label="Log2 Power")

# Compute time-averaged log2 PSD profiles across frequency
psd_orig = np.mean(np.log2(spectro + 1e-11), axis=1)
psd_corr = np.mean(np.log2(spectro_corr + 1e-11), axis=1)

# ==============================================================================
# FIGURE 3: PSD Comparison (Before vs. After Correction)
# ==============================================================================
fig_psd, ax_psd = plt.subplots(figsize=(8, 5))

ax_psd.plot(f_spectro, psd_orig, color="crimson", linewidth=1.5, label="Original PSD")
ax_psd.plot(f_spectro, psd_corr, color="teal", linewidth=1.5, linestyle="--", label="Corrected PSD")

# Highlight target frequency interval
ax_psd.axvspan(f_int[0], f_int[1], color="gray", alpha=0.2, label=f"Target Band [{f_int[0]}-{f_int[1]} Hz]")

ax_psd.set_title("Power Spectral Density: Before vs. After Correction")
ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_ylabel("Log2 Power")
ax_psd.set_xlim(0, 50)
ax_psd.grid(True, linestyle="--", alpha=0.5)
ax_psd.legend(loc="upper right")

fig_psd.tight_layout()

plt.show()