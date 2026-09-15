import numpy as np
from scipy.ndimage import binary_dilation

from Functions.correct_spectrogram import (
    apply_alpha_blended_ecdf,
    apply_monotonic_quantile_blend,
    apply_rank_preserved_alpha_ecdf,
)
from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram
from Functions.utils import compute_sef95

# ==================================================================================================
#                                   Configuration & Parameters
# ==================================================================================================

T = 20  # Duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
factor_list = [1, 1, 2]

# 10 sigma values: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
sigma_x_values = np.linspace(0.5, 5.0, 10)
n_seeds_per_sigma = 10  # 10 values * 10 realizations = 100 signals
total_signals = len(sigma_x_values) * n_seeds_per_sigma

f_int = [25, 35]

correction_methods = {
    "Alpha-Blended": lambda I, m: apply_alpha_blended_ecdf(I, m, alpha_cutoff=0.5)[0],
    "Monotonic Quantile": lambda I, m: apply_monotonic_quantile_blend(
        I, m, alpha_blend=0.8
    )[0],
    "Rank-Preserved Alpha": lambda I, m: apply_rank_preserved_alpha_ecdf(
        I, m, alpha_cutoff=0.5
    )[0],
}

# Determine PSD length via dummy spectrogram call
dummy_t, dummy_y = get_mixed_OU_signals(
    T, dt, lbda_list, omega_list, [3, 2, 0.5], factor_list
)
f_full, _, dummy_spectro = spectrogram(dummy_y, fs, nfft_factor=2)
n_freq_bins = len(f_full)

mask_f20 = f_full >= 20
f_eval = f_full[mask_f20]
mask_int = (f_eval >= f_int[0]) & (f_eval <= f_int[-1])

# ==================================================================================================
#                                 Preallocate Output Storage
# ==================================================================================================

dataset = {
    "metadata": {
        "sigma_x_values": sigma_x_values,
        "n_seeds_per_sigma": n_seeds_per_sigma,
        "total_signals": total_signals,
        "f_full": f_full,
        "f_int": f_int,
        "sigma_x_per_run": np.zeros(total_signals),
    },
    "original": {
        "psd": np.zeros((total_signals, n_freq_bins)),
        "baseline_eval": np.zeros((total_signals, np.sum(mask_f20))),
        "orig_vs_baseline_rmse": np.zeros(total_signals),
        "sef95": np.zeros(total_signals),
    },
    "methods": {
        name: {
            "psd_corrected": np.zeros((total_signals, n_freq_bins)),
            "rmse_vs_baseline": np.zeros(total_signals),
            "sef95": np.zeros(total_signals),
        }
        for name in correction_methods
    },
}

# ==================================================================================================
#                                     Data Generation Loop
# ==================================================================================================

run_idx = 0
for sig_idx, sigma_x in enumerate(sigma_x_values):
    current_sigma_list = [3, 2, float(sigma_x)]

    for seed_idx in range(n_seeds_per_sigma):
        dataset["metadata"]["sigma_x_per_run"][run_idx] = sigma_x

        # 1. Signal generation and spectrogram
        t_k, y_k = get_mixed_OU_signals(
            T, dt, lbda_list, omega_list, current_sigma_list, factor_list
        )
        _, _, spectro_full = spectrogram(y_k, fs, nfft_factor=2)

        psd_full_orig = np.mean(spectro_full, axis=1)
        sef_orig = compute_sef95(f_full, psd_full_orig)

        dataset["original"]["psd"][run_idx] = psd_full_orig
        dataset["original"]["sef95"][run_idx] = sef_orig

        # 2. Extract frequency band >= 20 Hz
        M_eval = spectro_full[mask_f20, :].copy()
        I_orig_eval = np.log2(M_eval + 1e-11)

        # 3. Segmentation and mask dilation
        seg_mask, psd_eval, baseline_eval, _, _ = segment_blobs(
            M_eval, f_eval, f_int, lam=1e4, p=0.01
        )
        expanded_mask = binary_dilation(seg_mask.astype(bool), iterations=2)

        dataset["original"]["baseline_eval"][run_idx] = baseline_eval

        # Error between uncorrected PSD and target Whittaker baseline inside f_int
        target_baseline_log = np.log2(baseline_eval[mask_int] + 1e-11)
        orig_log_int = np.log2(psd_eval[mask_int] + 1e-11)
        dataset["original"]["orig_vs_baseline_rmse"][run_idx] = np.sqrt(
            np.mean((orig_log_int - target_baseline_log) ** 2)
        )

        # 4. Apply each correction method
        has_blobs = np.any(expanded_mask)
        for name, fn in correction_methods.items():
            if has_blobs:
                I_corr_eval = fn(I_orig_eval, expanded_mask)
            else:
                I_corr_eval = I_orig_eval.copy()

            # Reconstruct corrected full spectrogram and PSD
            spectro_full_corr = spectro_full.copy()
            spectro_full_corr[mask_f20, :] = np.clip(
                2**I_corr_eval - 1e-11, 0, None
            )
            psd_full_corr = np.mean(spectro_full_corr, axis=1)

            # Error between corrected PSD and baseline inside f_int
            psd_corr_eval = np.mean(spectro_full_corr[mask_f20, :], axis=1)
            psd_corr_log_int = np.log2(psd_corr_eval[mask_int] + 1e-11)
            rmse_corr = np.sqrt(
                np.mean((psd_corr_log_int - target_baseline_log) ** 2)
            )

            sef_corr = compute_sef95(f_full, psd_full_corr)

            # Store metrics
            dataset["methods"][name]["psd_corrected"][run_idx] = psd_full_corr
            dataset["methods"][name]["rmse_vs_baseline"][run_idx] = rmse_corr
            dataset["methods"][name]["sef95"][run_idx] = sef_corr

        run_idx += 1

# ==================================================================================================
#                                         Persist Dataset
# ==================================================================================================

np.savez_compressed("spectrogram_correction_evaluation_dataset_sigma.npz", **dataset)
print("Benchmark completed: 100 runs evaluated and saved to 'spectrogram_correction_evaluation_dataset_sigma.npz'.")