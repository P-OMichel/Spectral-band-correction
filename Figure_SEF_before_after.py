import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from Functions.correct_spectrogram import (
    apply_alpha_blended_ecdf,
    apply_monotonic_quantile_blend,
    apply_rank_preserved_alpha_ecdf,
    apply_spatially_conditioned_quantile_blend,
)
from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram
from Functions.utils import compute_sef95

# ==================================================================================================
#                                      Configuration & Setup
# ==================================================================================================

T = 20  # Duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
sigma_list = [3, 2, 2]

f_int = [25, 35]
factors = np.linspace(1.0, 2.0, 10)

# Define all correction methods to evaluate
correction_methods = {
    "Alpha-Blended": {
        "fn": lambda I, m: apply_alpha_blended_ecdf(I, m, alpha_cutoff=0.5)[0],
        "color": "tab:blue",
        "marker": "o",
        "ls": "-",
    },
    "Monotonic Quantile": {
        "fn": lambda I, m: apply_monotonic_quantile_blend(I, m, alpha_blend=0.8)[0],
        "color": "tab:orange",
        "marker": "^",
        "ls": "--",
    },
    "Rank-Preserved Alpha": {
        "fn": lambda I, m: apply_rank_preserved_alpha_ecdf(I, m, alpha_cutoff=0.5)[0],
        "color": "tab:green",
        "marker": "s",
        "ls": "-.",
    }
}

# Storage containers
rmse_records = {name: [] for name in correction_methods}
sef_records = {name: [] for name in correction_methods}
sef_orig_list = []

psd_first = {"methods": {}}
psd_last = {"methods": {}}

# ==================================================================================================
#                                 Batch Factor Evaluation Loop
# ==================================================================================================

for k, f_val in enumerate(factors):
    current_factors = [1, 1, float(f_val)]

    # 1. Generate signal & spectrogram
    t_k, y_k = get_mixed_OU_signals(
        T, dt, lbda_list, omega_list, sigma_list, current_factors
    )
    f_full, t_full, spectro_full = spectrogram(y_k, fs, nfft_factor=2)

    psd_full_orig = np.mean(spectro_full, axis=1)
    sef_orig_list.append(compute_sef95(f_full, psd_full_orig))

    # 2. Extract >= 20 Hz region
    mask_f20 = f_full >= 20
    f_eval = f_full[mask_f20]
    M_eval = spectro_full[mask_f20, :].copy()
    I_orig_eval = np.log2(M_eval + 1e-11)

    # 3. Segmentation & Mask expansion
    seg_mask_k, psd_k, baseline_k, _, _ = segment_blobs(
        M_eval, f_eval, f_int, lam=1e4, p=0.01
    )
    seg_bool_k = seg_mask_k.astype(bool)
    expanded_mask_k = binary_dilation(seg_bool_k, iterations=2)

    mask_int = (f_eval >= f_int[0]) & (f_eval <= f_int[-1])
    target_baseline_log = np.log2(baseline_k[mask_int] + 1e-11)

    is_first = k == 0
    is_last = k == len(factors) - 1

    if is_first:
        psd_first.update(
            {
                "f_full": f_full,
                "orig": psd_full_orig,
                "f_eval": f_eval,
                "baseline": baseline_k,
            }
        )
    elif is_last:
        psd_last.update(
            {
                "f_full": f_full,
                "orig": psd_full_orig,
                "f_eval": f_eval,
                "baseline": baseline_k,
            }
        )

    # 4. Evaluate each correction algorithm
    for name, method_info in correction_methods.items():
        if np.any(expanded_mask_k):
            I_corr_eval = method_info["fn"](I_orig_eval, expanded_mask_k)
        else:
            I_corr_eval = I_orig_eval.copy()

        # Reconstruct full spectrogram
        spectro_full_corr = spectro_full.copy()
        spectro_full_corr[mask_f20, :] = np.clip(2**I_corr_eval - 1e-11, 0, None)
        psd_full_corr = np.mean(spectro_full_corr, axis=1)

        # Compute RMSE against Whittaker baseline inside f_int
        psd_corr_eval = np.mean(spectro_full_corr[mask_f20, :], axis=1)
        psd_corr_log = np.log2(psd_corr_eval[mask_int] + 1e-11)
        rmse = np.sqrt(np.mean((psd_corr_log - target_baseline_log) ** 2))
        rmse_records[name].append(rmse)

        # Compute SEF95
        sef_records[name].append(compute_sef95(f_full, psd_full_corr))

        # Store PSD profiles for first and last factors
        if is_first:
            psd_first["methods"][name] = psd_full_corr
        elif is_last:
            psd_last["methods"][name] = psd_full_corr


# ==================================================================================================
#                                         Visualizations
# ==================================================================================================

# --- 1. First and Last PSD Comparisons
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

for ax, data_store, factor_label in zip(
    axes, [psd_first, psd_last], [factors[0], factors[-1]]
):
    ax.plot(
        data_store["f_full"],
        np.log2(data_store["orig"] + 1e-11),
        color="red",
        linestyle="-",
        linewidth=1.5,
        label="Original PSD",
    )
    ax.plot(
        data_store["f_eval"],
        np.log2(data_store["baseline"] + 1e-11),
        color="black",
        linestyle=":",
        linewidth=2.5,
        label="Target Baseline",
    )

    for name, method_info in correction_methods.items():
        ax.plot(
            data_store["f_full"],
            np.log2(data_store["methods"][name] + 1e-11),
            label=name,
            color=method_info["color"],
            linestyle=method_info["ls"],
            linewidth=1.3,
        )

    ax.axvspan(f_int[0], f_int[1], color="gray", alpha=0.15, label="Target ROI")
    ax.set_xlim([0, 50])
    ax.set_title(f"PSD Comparison (Bump Factor = {factor_label:.2f})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\log_2(\text{PSD})$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

plt.show()

# --- 2. Multi-Method RMSE vs Bump Factor
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

for name, method_info in correction_methods.items():
    ax.plot(
        factors,
        rmse_records[name],
        label=name,
        color=method_info["color"],
        marker=method_info["marker"],
        linestyle=method_info["ls"],
        linewidth=1.8,
        markersize=6,
    )

ax.set_title(
    r"Reconstruction RMSE vs. Bump Factor within $[25, 35]$ Hz", fontweight="bold"
)
ax.set_xlabel("30 Hz Bump Scaling Factor")
ax.set_ylabel(r"RMSE ($\log_2\text{ PSD vs. Baseline}$)")
ax.set_xticks(factors)
ax.set_xticklabels([f"{f:.2f}" for f in factors])
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")

plt.show()

# --- 3. Multi-Method SEF95 Comparison
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

ax.plot(
    factors,
    sef_orig_list,
    "k--",
    linewidth=2.2,
    marker="x",
    markersize=7,
    label="Original Signal (Uncorrected)",
)

for name, method_info in correction_methods.items():
    ax.plot(
        factors,
        sef_records[name],
        label=name,
        color=method_info["color"],
        marker=method_info["marker"],
        linestyle=method_info["ls"],
        linewidth=1.8,
        markersize=6,
    )

ax.set_title(
    r"Spectral Edge Frequency ($\text{SEF}_{95}$) across Bump Factors",
    fontweight="bold",
)
ax.set_xlabel("30 Hz Bump Scaling Factor")
ax.set_ylabel(r"$\text{SEF}_{95}$ (Hz)")
ax.set_xticks(factors)
ax.set_xticklabels([f"{f:.2f}" for f in factors])
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="best")

plt.show()