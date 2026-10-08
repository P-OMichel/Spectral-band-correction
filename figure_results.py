"""Generates comparative time-domain, spindle burst, spectrogram (>= 0.1 Hz),

and full-signal PSD comparison figures for a 3-bump signal before and after correction.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.ndimage import binary_closing, binary_dilation, label

from Functions.correct_spectrogram import apply_rank_preserved_alpha_ecdf
from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram


# ==============================================================================
# Helper Functions
# ==============================================================================
def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Applies a zero-phase Butterworth bandpass filter."""
    sos = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data)


def istft_from_corrected_spectrogram(
    spectro_orig,
    spectro_corr,
    stft_orig,
    fs,
    nperseg_factor=1,
    noverlap_factor=0.9,
    nfft_factor=2,
    scaling="psd",
):
    """Reconstructs time-domain signal by applying a magnitude gain mask to original STFT."""
    nperseg = int(nperseg_factor * fs)
    noverlap = int(noverlap_factor * nperseg)
    nfft = int(nfft_factor * nperseg)
    window = signal.windows.hamming(nperseg, sym=True)

    gain = np.sqrt(np.maximum(spectro_corr, 0) / (spectro_orig + 1e-12))
    stft_corr = stft_orig * gain

    t_corr, y_corr = signal.istft(
        stft_corr,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
    )
    return stft_corr, t_corr, y_corr


def extract_spindles(
    filtered_sig,
    envelope_threshold_q=0.74,
    close_gap_pts=120,
    min_len_pts=160,
):
    """Detects spindle intervals with slightly shortened duration parameters."""
    analytic = signal.hilbert(filtered_sig)
    envelope = np.abs(analytic)

    # Smooth envelope
    kernel_len = 41
    kernel = np.hanning(kernel_len)
    kernel /= kernel.sum()
    smooth_env = np.convolve(envelope, kernel, mode="same")

    # Thresholding
    thresh = np.quantile(smooth_env, envelope_threshold_q)
    binary_spindles = smooth_env > thresh

    # Morphological closing to bridge close sub-bursts
    binary_closed = binary_closing(binary_spindles, structure=np.ones(close_gap_pts, dtype=bool))

    labeled_mask, num_features = label(binary_closed)
    spindle_intervals = []
    for feat_idx in range(1, num_features + 1):
        idx = np.where(labeled_mask == feat_idx)[0]
        if len(idx) >= min_len_pts:
            spindle_intervals.append((idx[0], idx[-1]))

    return spindle_intervals, smooth_env


def get_scaled_burst(filt_burst, center_offset, scale_factor=2.5):
    """Centers burst oscillation around 0, scales for visual prominence, and offsets."""
    oscillation = filt_burst - np.mean(filt_burst)
    return oscillation * scale_factor + center_offset


# ==============================================================================
# 1. Signal Synthesis (Single 20s signal with 3 bumps)
# ==============================================================================
T = 20
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 3]

t, y_ou = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)
N_total = len(t)

# Aperiodic background with 1/f^chi decay and knee at 1 Hz
freqs = np.fft.rfftfreq(N_total, d=dt)
f_knee, chi = 1.0, 2.0
psd_aperiodic = 1.0 / (f_knee**chi + np.maximum(freqs, 1e-6) ** chi)

np.random.seed(42)
phases = np.random.uniform(0, 2 * np.pi, len(freqs))
fft_aperiodic = np.sqrt(psd_aperiodic) * np.exp(1j * phases)
fft_aperiodic[0] = 0

aperiodic_signal = np.fft.irfft(fft_aperiodic, n=N_total)
aperiodic_signal = (aperiodic_signal / np.std(aperiodic_signal)) * 2.5

# Total signal (3 bumps + aperiodic)
y_total = y_ou + aperiodic_signal

# ==============================================================================
# 2. Spectrogram (Displayed from >= 0.1 Hz)
# ==============================================================================
f_spectro, t_spectro, spectro, stft = spectrogram(y_total, fs, nfft_factor=2)

mask_disp = f_spectro >= 0.1
f_disp = f_spectro[mask_disp]
spectro_disp = spectro[mask_disp, :]
spectro_log = np.log2(spectro_disp + 1e-11)

v_min = np.quantile(spectro_log, 0.05)
v_max = np.quantile(spectro_log, 0.95)

# Target correction mask (>= 20 Hz for blob segmentation)
mask_f_corr = f_spectro >= 20
f_M = f_spectro[mask_f_corr]
M = spectro[mask_f_corr, :]

# ==============================================================================
# 3. 5-Second Window & Spindle Extraction
# ==============================================================================
f_spindles = [20, 40]
y_filt_spindles = bandpass_filter(y_total, fs, f_spindles[0], f_spindles[1])

# Select 5-second window
w_mask = (t >= 7.0) & (t <= 12.0)
t_w = t[w_mask]
y_w = y_total[w_mask]
filt_w = y_filt_spindles[w_mask]

spindle_scale = 2.5
spindles_w, _ = extract_spindles(
    filt_w,
    envelope_threshold_q=0.74,
    close_gap_pts=120,
    min_len_pts=160,
)

# ==============================================================================
# 4. Segmentation [20, 40] Hz, Alpha-ECDF Correction, and iSTFT
# ==============================================================================
f_int = [20, 40]
seg_mask, psd_val, baseline, T_low, T_high = segment_blobs(M, f_M, f_int)
expanded_mask = binary_dilation(seg_mask.astype(bool), iterations=2)

I_corr_log, _ = apply_rank_preserved_alpha_ecdf(
    np.log2(M + 1e-11), expanded_mask, alpha_cutoff=0.3
)

spectro_corr = spectro.copy()
spectro_corr[mask_f_corr, :] = np.exp2(I_corr_log) - 1e-11

# Displayed log spectrogram after correction (>= 0.1 Hz)
spectro_corr_disp_log = np.log2(spectro_corr[mask_disp, :] + 1e-11)

# Time-domain reconstruction
stft_corr, t_corr, y_corr = istft_from_corrected_spectrogram(
    spectro, spectro_corr, stft, fs, nfft_factor=2
)
n_pts = min(len(y_total), len(y_corr))
y_corr_aligned = y_corr[:n_pts]
y_total_aligned = y_total[:n_pts]

# Filter corrected signal for spindle representation
y_filt_spindles_corr = bandpass_filter(y_corr_aligned, fs, f_spindles[0], f_spindles[1])

y_w_corr = y_corr_aligned[w_mask[:n_pts]]
filt_w_corr = y_filt_spindles_corr[w_mask[:n_pts]]

# Welch PSD computation for the full signals
f_welch_orig, psd_welch_orig = signal.welch(y_total_aligned, fs=fs, nperseg=int(2 * fs))
f_welch_corr, psd_welch_corr = signal.welch(y_corr_aligned, fs=fs, nperseg=int(2 * fs))

# ==============================================================================
# 5. Plotting
# ==============================================================================

# --- FIGURE 1: Spectrograms (>= 0.1 Hz, Before & After Correction) ---
fig_spec, axes_spec = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True)

im0 = axes_spec[0].pcolormesh(
    t_spectro,
    f_disp,
    spectro_log,
    shading="gouraud",
    cmap="jet",
    vmin=v_min,
    vmax=v_max,
)
axes_spec[0].set_title("Original Spectrogram (≥ 0.1 Hz)")
axes_spec[0].set_ylabel("Frequency (Hz)")
axes_spec[0].set_ylim(0.1, 50)

im1 = axes_spec[1].pcolormesh(
    t_spectro,
    f_disp,
    spectro_corr_disp_log,
    shading="gouraud",
    cmap="jet",
    vmin=v_min,
    vmax=v_max,
)
axes_spec[1].set_title(f"Corrected Spectrogram (Segmented f_int: [{f_int[0]}, {f_int[1]}] Hz)")
axes_spec[1].set_xlabel("Time (s)")
axes_spec[1].set_ylabel("Frequency (Hz)")
axes_spec[1].set_ylim(0.1, 50)

fig_spec.subplots_adjust(right=0.88)
cbar_ax = fig_spec.add_axes([0.90, 0.15, 0.02, 0.7])
fig_spec.colorbar(im0, cax=cbar_ax, label="Log2 Power")

# --- FIGURE 2: Fused Time-Domain Window Visualizations (3 Panels) ---
fig_time, ax_time = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)

# Subplot 1: Original signal with detected spindles aligned to local mean
ax_time[0].plot(t_w, y_w, color="gray", alpha=0.55, label="Raw Signal")
for idx_start, idx_end in spindles_w:
    t_burst = t_w[idx_start:idx_end]
    mean_offset = np.mean(y_w[idx_start:idx_end])
    burst_scaled = get_scaled_burst(
        filt_w[idx_start:idx_end],
        mean_offset,
        scale_factor=spindle_scale,
    )
    ax_time[0].plot(
        t_burst,
        burst_scaled,
        color="crimson",
        linewidth=1.8,
        label=f"Spindles [25–35 Hz] ({spindle_scale}x Gain)"
        if idx_start == spindles_w[0][0]
        else "",
    )
ax_time[0].set_title("Original Signal with Detected Spindles (Aligned to Local Mean)")
ax_time[0].set_ylabel("Amplitude")
ax_time[0].legend(loc="upper right")
ax_time[0].grid(True, linestyle="--", alpha=0.5)

# Subplot 2: Original vs Corrected Signal (both aligned on 0)
ax_time[1].plot(t_w, y_w, color="gray", alpha=0.6, label="Original Signal")
ax_time[1].plot(t_w, y_w_corr, color="teal", alpha=0.85, linewidth=1.2, label="Corrected Signal")
ax_time[1].set_title("Original vs. Corrected Signal (Zero-Aligned)")
ax_time[1].set_ylabel("Amplitude")
ax_time[1].legend(loc="upper right")
ax_time[1].grid(True, linestyle="--", alpha=0.5)

# Subplot 3: Corrected signal with corresponding corrected spindles aligned to local mean
ax_time[2].plot(t_w, y_w_corr, color="teal", alpha=0.55, label="Corrected Signal")
for idx_start, idx_end in spindles_w:
    t_burst = t_w[idx_start:idx_end]
    mean_corr_offset = np.mean(y_w_corr[idx_start:idx_end])
    burst_corr_scaled = get_scaled_burst(
        filt_w_corr[idx_start:idx_end],
        mean_corr_offset,
        scale_factor=spindle_scale,
    )
    ax_time[2].plot(
        t_burst,
        burst_corr_scaled,
        color="darkred",
        linewidth=1.8,
        label=f"Corrected Spindles ({spindle_scale}x Gain)"
        if idx_start == spindles_w[0][0]
        else "",
    )
ax_time[2].set_title("Corrected Signal with Corresponding Corrected Spindles (Aligned to Local Mean)")
ax_time[2].set_xlabel("Time (s)")
ax_time[2].set_ylabel("Amplitude")
ax_time[2].legend(loc="upper right")
ax_time[2].grid(True, linestyle="--", alpha=0.5)

fig_time.tight_layout()

# --- FIGURE 3: Full-Signal Power Spectral Density (PSD) Comparison ---
fig_psd, ax_psd = plt.subplots(figsize=(10, 5))

ax_psd.set_ylim(0, 45)
ax_psd.plot(
    f_welch_orig,
    np.log2(psd_welch_orig + 1e-11),
    color="crimson",
    linewidth=1.6,
    label="Original Signal PSD",
)
ax_psd.plot(
    f_welch_corr,
    np.log2(psd_welch_corr + 1e-11),
    color="teal",
    linewidth=1.6,
    linestyle="--",
    label="Corrected Signal PSD",
)

ax_psd.axvspan(
    f_int[0],
    f_int[1],
    color="gray",
    alpha=0.2,
    label=f"Segmentation Band [{f_int[0]}-{f_int[1]} Hz]",
)

ax_psd.set_title("Full Signal PSD Comparison: Original vs. Corrected (Welch Method)")
ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_ylabel("Log2 Power (V²/Hz)")
ax_psd.set_xlim(0.1, 50)
ax_psd.grid(True, linestyle="--", alpha=0.5)
ax_psd.legend(loc="upper right")
fig_psd.tight_layout()

plt.show()