'''
File to generate the introduction figure with:
- contaminated time series
- spectrogram
- segmentation
- correction
- corrected time series 
'''
from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.ndimage import binary_dilation
from Functions.correct_spectrogram import apply_rank_preserved_alpha_ecdf
import matplotlib.gridspec as gridspec

import scipy.signal as signal

def istft_from_corrected_spectrogram(
    spectro_corr,
    stft_orig_sliced,
    fs,
    nperseg_factor=1,
    noverlap_factor=0.9,
    nfft_factor=1,
    scaling="psd",
):
    # 1. Recalculate parameters matching the forward STFT
    nperseg = int(nperseg_factor * fs)
    noverlap = int(noverlap_factor * nperseg)
    nfft = int(nfft_factor * nperseg)
    window = signal.windows.hamming(nperseg, sym=True)

    # 2. Get corrected magnitude from power spectrogram (Sxx = |stft|^2)
    # Ensure no small negative values before sqrt
    mag_corr = np.sqrt(np.maximum(spectro_corr, 0))

    # 3. Retrieve the original phase from the sliced STFT
    phase_orig = np.angle(stft_orig_sliced)

    # 4. Form the corrected complex STFT for the lower frequencies ([:j, :])
    stft_corr_sliced = mag_corr * np.exp(1j * phase_orig)

    # 5. Restore full frequency bins required by iSTFT: (nfft // 2) + 1
    n_freq_bins_full = (nfft // 2) + 1
    j, n_time_steps = stft_corr_sliced.shape

    if j < n_freq_bins_full:
        # Pad upper frequencies above f_cut with zeros
        stft_corr_full = np.pad(
            stft_corr_sliced,
            pad_width=((0, n_freq_bins_full - j), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        stft_corr_full = stft_corr_sliced

    # 6. Inverse STFT to get the corrected time series
    t_corr, y_corr = signal.istft(
        stft_corr_full,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
    )

    return stft_corr_full, t_corr, y_corr

# --- generate synthetic signal
T = 20  # Duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 1]

t, y = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

f_spectro, t_spectro, spectro, stft = spectrogram(y, fs, nfft_factor=2)

# PSD 
psd_spectro = np.mean(np.log2(spectro + 0.00000001), axis = 1) #sc.signal.savgol_filter(np.mean(np.log2(spectro + 0.00000001), axis = 0), 2, 1)


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
ref_mask = binary_dilation(expanded_mask, iterations=2)
border_ring = ref_mask & ~expanded_mask

# correction
I_a_r_blend, ref_vals_a_r_blend = apply_rank_preserved_alpha_ecdf(I_orig, expanded_mask, alpha_cutoff=0.3)

spectro_corr = spectro.copy()
eps = 1e-11  
I_a_r_blend_linear = np.exp2(I_a_r_blend) - eps
spectro_corr[mask_f, :] = I_a_r_blend_linear

psd_corr = np.mean(np.log2(spectro_corr + 0.00000001), axis = 1)

# corrected signal
stft_corr, t_corr, y_corr = istft_from_corrected_spectrogram(spectro_corr=spectro_corr, stft_orig_sliced=stft, fs=fs)

fig = plt.figure(constrained_layout=True)
# Define relative widths: small, medium, large
w_small = 1
w_med = 2
w_large = 4
# Outer layout: 2 rows, 1 column
outer_gs = gridspec.GridSpec(2, 1, figure=fig)
# Row 0: widths correspond to [0,0] (med), [0,1] (small), [0,2] (large)
gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], width_ratios=[w_small, w_med, w_large])
# Row 1: widths correspond to [1,0] (large), [1,1] (med), [1,2] (small)
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], width_ratios=[w_large, w_small, w_med])
# Create the 2x3 axes array
axes = [[fig.add_subplot(gs_row0[0, i]) for i in range(3)],[fig.add_subplot(gs_row1[0, i]) for i in range(3)]]

v_min = np.quantile(spectro_log, 0.05)
v_max= np.quantile(spectro_log, 0.95)
axes[0][0].plot(y)
axes[0][0].set_axis_off()
axes[0][1].plot(f_spectro, psd_spectro)
axes[0][2].pcolormesh(t_spectro, f_M, spectro_log, shading = 'gouraud', cmap = 'jet', vmin = v_min, vmax = v_max)
axes[0][2].contour(t_spectro, f_M, expanded_mask, colors = 'black', linewidth = 2)

axes[1][0].pcolormesh(t_spectro, f_M, I_a_r_blend, shading = 'gouraud', cmap = 'jet', vmin = v_min, vmax = v_max)
axes[1][0].contour(t_spectro, f_M, expanded_mask, colors = 'black', linewidth = 2)
axes[1][1].plot(t, y_corr)
axes[1][1].set_axis_off()
axes[1][2].plot(f_spectro, psd_corr)

plt.show()


