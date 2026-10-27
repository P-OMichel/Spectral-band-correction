"""Generates a publication-ready 4x4 comparative figure across parameter regimes.

Columns:
  1. Time-Domain Overlay with Separated Burst Comparison
  2. Original Spectrogram (0.1–45 Hz, no y-label)
  3. Corrected Spectrogram (0.1–45 Hz, no y-label)
  4. Welch PSD Overlay (0.1–45 Hz, y >= -10, no y-label)

Rows:
  - Row 1: sigma_3 = 1.0 (Fixed factor = 3.0)
  - Row 2: sigma_3 = 4.0 (Fixed factor = 3.0)
  - Row 3: factor_3 = 1.0 (Fixed sigma = 2.0)
  - Row 4: factor_3 = 5.0 (Fixed sigma = 2.0)
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
    return y_corr


def extract_spindles(
    filtered_sig,
    envelope_threshold_q=0.74,
    close_gap_pts=120,
    min_len_pts=160,
):
    analytic = signal.hilbert(filtered_sig)
    envelope = np.abs(analytic)

    # Smooth envelope
    kernel_len = 41
    kernel = np.hanning(kernel_len)
    kernel /= kernel.sum()
    smooth_env = np.convolve(envelope, kernel, mode="same")

    # Thresholding & morphological closing
    thresh = np.quantile(smooth_env, envelope_threshold_q)
    binary_spindles = smooth_env > thresh
    binary_closed = binary_closing(
        binary_spindles, structure=np.ones(close_gap_pts, dtype=bool)
    )

    labeled_mask, num_features = label(binary_closed)
    spindle_intervals = []
    for feat_idx in range(1, num_features + 1):
        idx = np.where(labeled_mask == feat_idx)[0]
        if len(idx) >= min_len_pts:
            spindle_intervals.append((idx[0], idx[-1]))

    return spindle_intervals


def process_case(
    sigma_3,
    factor_3,
    T=20,
    dt=0.001,
    f_spindles=[20, 40],
    seed=42,
):
    fs = 1 / dt
    lbda_list = [1, 2, 1]
    omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
    sigma_list = [3, 2, sigma_3]
    factor_list = [1, 1, factor_3]

    t, y_ou = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)
    N_total = len(t)

    # 1/f^chi background with knee at 1 Hz
    freqs = np.fft.rfftfreq(N_total, d=dt)
    f_knee, chi = 1.0, 2.0
    psd_aperiodic = 1.0 / (f_knee**chi + np.maximum(freqs, 1e-6) ** chi)

    np.random.seed(seed)
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    fft_aperiodic = np.sqrt(psd_aperiodic) * np.exp(1j * phases)
    fft_aperiodic[0] = 0

    aperiodic_signal = np.fft.irfft(fft_aperiodic, n=N_total)
    aperiodic_signal = (aperiodic_signal / np.std(aperiodic_signal)) * 2.5
    y_total = y_ou + aperiodic_signal

    # Spectrogram computation
    f_spectro, t_spectro, spectro, stft = spectrogram(y_total, fs, nfft_factor=2)

    # Correction band (>= 20 Hz)
    mask_f_corr = f_spectro >= 20
    f_M = f_spectro[mask_f_corr]
    M = spectro[mask_f_corr, :]

    seg_mask, _, _, _, _ = segment_blobs(M, f_M, f_spindles)
    expanded_mask = binary_dilation(seg_mask.astype(bool), iterations=2)
    I_corr_log, _ = apply_rank_preserved_alpha_ecdf(
        np.log2(M + 1e-11), expanded_mask, alpha_cutoff=0.3
    )

    spectro_corr = spectro.copy()
    spectro_corr[mask_f_corr, :] = np.exp2(I_corr_log) - 1e-11

    # Reconstruction via iSTFT
    y_corr = istft_from_corrected_spectrogram(spectro, spectro_corr, stft, fs, nfft_factor=2)
    n_pts = min(len(y_total), len(y_corr))
    y_total_aligned = y_total[:n_pts]
    y_corr_aligned = y_corr[:n_pts]

    # Bandpass filtering for spindles
    y_filt_orig = bandpass_filter(y_total_aligned, fs, f_spindles[0], f_spindles[1])
    y_filt_corr = bandpass_filter(y_corr_aligned, fs, f_spindles[0], f_spindles[1])

    # 5-second analysis window (7 to 12 s)
    w_mask = (t[:n_pts] >= 7.0) & (t[:n_pts] <= 12.0)
    t_w = t[:n_pts][w_mask]
    y_orig_w = y_total_aligned[w_mask]
    y_corr_w = y_corr_aligned[w_mask]
    filt_orig_w = y_filt_orig[w_mask]
    filt_corr_w = y_filt_corr[w_mask]

    # Spindle burst interval extraction
    spindles_w = extract_spindles(
        filt_orig_w,
        envelope_threshold_q=0.74,
        close_gap_pts=120,
        min_len_pts=160,
    )

    # Welch PSD computation (full signal)
    f_welch, psd_orig = signal.welch(y_total_aligned, fs=fs, nperseg=int(2 * fs))
    _, psd_corr = signal.welch(y_corr_aligned, fs=fs, nperseg=int(2 * fs))

    # Mask for display (0.1 to 45 Hz)
    mask_disp = (f_spectro >= 0.1) & (f_spectro <= 45.0)
    f_disp = f_spectro[mask_disp]
    spectro_orig_log = np.log2(spectro[mask_disp, :] + 1e-11)
    spectro_corr_log = np.log2(spectro_corr[mask_disp, :] + 1e-11)

    return {
        "t_w": t_w,
        "y_orig_w": y_orig_w,
        "y_corr_w": y_corr_w,
        "filt_orig_w": filt_orig_w,
        "filt_corr_w": filt_corr_w,
        "spindles_w": spindles_w,
        "t_spectro": t_spectro,
        "f_disp": f_disp,
        "spectro_orig_log": spectro_orig_log,
        "spectro_corr_log": spectro_corr_log,
        "f_welch": f_welch,
        "psd_orig_log": np.log2(psd_orig + 1e-11),
        "psd_corr_log": np.log2(psd_corr + 1e-11),
        "f_spindles": f_spindles,
    }


# ==============================================================================
# Define the 4 Parameter Regimes
# ==============================================================================
cases = [
    {
        "row_label": r"$\sigma_3 = 1.0$",
        "sigma_3": 1.0,
        "factor_3": 3.0,
    },
    {
        "row_label": r"$\sigma_3 = 4.0$",
        "sigma_3": 4.0,
        "factor_3": 3.0,
    },
    {
        "row_label": r"$\mathrm{factor}_3 = 1.0$",
        "sigma_3": 2.0,
        "factor_3": 1.0,
    },
    {
        "row_label": r"$\mathrm{factor}_3 = 5.0$",
        "sigma_3": 2.0,
        "factor_3": 5.0,
    },
]

# Run simulations
results = [process_case(c["sigma_3"], c["factor_3"]) for c in cases]

# Dynamic range across all spectrograms
all_logs = np.concatenate(
    [
        np.concatenate([r["spectro_orig_log"].ravel(), r["spectro_corr_log"].ravel()])
        for r in results
    ]
)
v_min = np.quantile(all_logs, 0.05)
v_max = np.quantile(all_logs, 0.95)

# ==============================================================================
# Multi-Panel Figure Layout (4 rows x 4 columns)
# ==============================================================================
fig, axes = plt.subplots(4, 4, figsize=(20, 12), sharex="col")
plt.subplots_adjust(left=0.08, right=0.96, hspace=0.32, wspace=0.20)

col_titles = [
    "Signal & Offset Bursts (7–12 s)",
    "Original Spectrogram (0.1–45 Hz)",
    "Corrected Spectrogram (0.1–45 Hz)",
    "PSD Comparison (Welch, 0.1–45 Hz)",
]

spindle_scale = 2.5

for row_idx, (c, res) in enumerate(zip(cases, results)):
    ax_time = axes[row_idx, 0]
    ax_orig = axes[row_idx, 1]
    ax_corr = axes[row_idx, 2]
    ax_psd = axes[row_idx, 3]

    # --- Column 1: Time Domain + Horizontally Aligned Bursts Below ---
    ax_time.plot(
        res["t_w"],
        res["y_orig_w"],
        color="gray",
        alpha=0.45,
        linewidth=0.9,
        label="Raw" if row_idx == 0 else "",
    )
    ax_time.plot(
        res["t_w"],
        res["y_corr_w"],
        color="teal",
        alpha=0.6,
        linewidth=1.0,
        label="Corrected" if row_idx == 0 else "",
    )

    # Offset baseline below the main time series
    y_min_val = min(np.min(res["y_orig_w"]), np.min(res["y_corr_w"]))
    burst_max_amp = (
        max(np.max(np.abs(res["filt_orig_w"])), np.max(np.abs(res["filt_corr_w"])))
        * spindle_scale
    )
    burst_baseline = y_min_val - (burst_max_amp + 1.5)

    ax_time.axhline(
        burst_baseline, color="black", linestyle=":", alpha=0.35, linewidth=0.8
    )

    for i, (idx_start, idx_end) in enumerate(res["spindles_w"]):
        t_burst = res["t_w"][idx_start:idx_end]
        is_first = (row_idx == 0) and (i == 0)

        osc_orig = (
            res["filt_orig_w"][idx_start:idx_end]
            - np.mean(res["filt_orig_w"][idx_start:idx_end])
        )
        burst_orig_offset = osc_orig * spindle_scale + burst_baseline
        ax_time.plot(
            t_burst,
            burst_orig_offset,
            color="crimson",
            linewidth=1.4,
            label=f"Orig Burst ({spindle_scale}x)" if is_first else "",
        )

        osc_corr = (
            res["filt_corr_w"][idx_start:idx_end]
            - np.mean(res["filt_corr_w"][idx_start:idx_end])
        )
        burst_corr_offset = osc_corr * spindle_scale + burst_baseline
        ax_time.plot(
            t_burst,
            burst_corr_offset,
            color="darkblue",
            linewidth=1.4,
            linestyle="--",
            label=f"Corr Burst ({spindle_scale}x)" if is_first else "",
        )

    # Clean vertical label with only the parameter on the far left
    ax_time.set_ylabel(c["row_label"], fontsize=11, fontweight="bold", labelpad=10)
    ax_time.grid(True, linestyle="--", alpha=0.4)
    if row_idx == 0:
        ax_time.legend(loc="upper right", framealpha=0.85, fontsize=7.5, ncol=2)

    # --- Column 2: Spectrogram Before (0.1–45 Hz) ---
    im_orig = ax_orig.pcolormesh(
        res["t_spectro"],
        res["f_disp"],
        res["spectro_orig_log"],
        shading="gouraud",
        cmap="jet",
        vmin=v_min,
        vmax=v_max,
    )
    ax_orig.set_ylim(0.1, 45)

    # --- Column 3: Spectrogram After (0.1–45 Hz) ---
    im_corr = ax_corr.pcolormesh(
        res["t_spectro"],
        res["f_disp"],
        res["spectro_corr_log"],
        shading="gouraud",
        cmap="jet",
        vmin=v_min,
        vmax=v_max,
    )
    ax_corr.set_ylim(0.1, 45)

    # --- Column 4: PSD Superposed (0.1–45 Hz, y >= -10) ---
    ax_psd.plot(
        res["f_welch"],
        res["psd_orig_log"],
        color="crimson",
        linewidth=1.4,
        label="Original" if row_idx == 0 else "",
    )
    ax_psd.plot(
        res["f_welch"],
        res["psd_corr_log"],
        color="teal",
        linewidth=1.4,
        linestyle="--",
        label="Corrected" if row_idx == 0 else "",
    )
    ax_psd.axvspan(
        res["f_spindles"][0],
        res["f_spindles"][1],
        color="gray",
        alpha=0.18,
        label="[20-40 Hz]" if row_idx == 0 else "",
    )
    ax_psd.set_xlim(0.1, 45)
    ax_psd.set_ylim(bottom=-10)
    ax_psd.grid(True, linestyle="--", alpha=0.4)
    if row_idx == 0:
        ax_psd.legend(loc="upper right", framealpha=0.85, fontsize=7.5)

    # Column titles on the top row
    if row_idx == 0:
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)

# Bottom row X labels
axes[-1, 0].set_xlabel("Time (s)", fontsize=9.5)
axes[-1, 1].set_xlabel("Time (s)", fontsize=9.5)
axes[-1, 2].set_xlabel("Time (s)", fontsize=9.5)
axes[-1, 3].set_xlabel("Frequency (Hz)", fontsize=9.5)

plt.show()