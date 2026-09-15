import time
import matplotlib.pyplot as plt
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


def benchmark_and_plot(durations=(30, 60, 120), n_repeats=5):
    dt = 0.001
    fs = 1 / dt

    lbda_list = [1, 2, 1]
    omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
    sigma_list = [3, 2, 2]
    factor_list = [1, 1, 1]
    f_int = [25, 35]

    methods = {
        "Alpha-Blended ECDF": lambda I, m: apply_alpha_blended_ecdf(
            I, m, alpha_cutoff=0.5
        ),
        "Monotonic Quantile Blend": lambda I, m: apply_monotonic_quantile_blend(
            I, m, alpha_blend=0.8
        ),
        "Rank-Preserved Alpha ECDF": lambda I, m: apply_rank_preserved_alpha_ecdf(
            I, m, alpha_cutoff=0.5
        ),
    }

    # Structure to hold timings: {method_name: {"means": [], "stds": []}}
    results = {m_name: {"means": [], "stds": []} for m_name in methods}
    durations_array = np.array(durations)

    for T in durations:
        print(f"Generating and benchmarking signal for T = {T}s...")
        t, y = get_mixed_OU_signals(
            T, dt, lbda_list, omega_list, sigma_list, factor_list
        )
        f_spectro, t_spectro, spectro = spectrogram(y, fs, nfft_factor=2)

        mask_f = f_spectro >= 20
        f_M = f_spectro[mask_f]
        M = spectro[mask_f, :]
        I_orig = np.log2(M + 1e-11)

        seg_mask, _, _, _, _ = segment_blobs(M, f_M, f_int)
        expanded_mask = binary_dilation(seg_mask.astype(bool), iterations=2)

        for name, func in methods.items():
            # Warm-up run
            func(I_orig, expanded_mask)

            timings = []
            for _ in range(n_repeats):
                t_start = time.perf_counter()
                func(I_orig, expanded_mask)
                t_end = time.perf_counter()
                timings.append((t_end - t_start) * 1000.0)  # ms

            results[name]["means"].append(np.mean(timings))
            results[name]["stds"].append(np.std(timings))

    # --- Print Tabular Output ---
    method_names = list(methods.keys())
    col_width_method = max(len(m) for m in method_names) + 2
    col_width_dur = 22

    header = f"{'Method':<{col_width_method}}" + "".join(
        [f"{f'T = {T}s (mean ± std ms)':<{col_width_dur}}" for T in durations]
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    for name in method_names:
        row = f"{name:<{col_width_method}}"
        for idx in range(len(durations)):
            mean_val = results[name]["means"][idx]
            std_val = results[name]["stds"][idx]
            val_str = f"{mean_val:6.2f} ± {std_val:5.2f} ms"
            row += f"{val_str:<{col_width_dur}}"
        print(row)

    print(separator + "\n")

    # --- Plot Scaling Curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^"]

    for (name, data), marker in zip(results.items(), markers):
        means = np.array(data["means"])
        stds = np.array(data["stds"])
        ax.errorbar(
            durations_array,
            means,
            yerr=stds,
            marker=marker,
            capsize=4,
            capthick=1.2,
            linewidth=1.8,
            markersize=6,
            label=name,
        )

    ax.set_title("Execution Time Trend vs. Signal Duration", fontsize=14, pad=12)
    ax.set_xlabel("Signal Duration (s)", fontsize=12)
    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_xticks(durations_array)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    benchmark_and_plot(durations=[30, 60, 120, 240, 480], n_repeats=25)