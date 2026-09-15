import time
import matplotlib.pyplot as plt
import numpy as np

from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram


def benchmark_segmentation_modes(durations=(30, 60, 120, 240, 480), n_repeats=25):
    """Benchmarks and compares execution time of segment_blobs methods."""
    dt = 0.001
    fs = 1 / dt

    lbda_list = [1, 2, 1]
    omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
    sigma_list = [3, 2, 2]
    factor_list = [1, 1, 1]
    f_int = [25, 35]

    methods = {
        "Baseline Method": lambda M, f_M: segment_blobs(
            M, f_M, f_int, method="baseline"
        ),
        "Quantile Method": lambda M, f_M: segment_blobs(
            M, f_M, f_int, method="quantile"
        ),
    }

    results = {m: {"means": [], "stds": []} for m in methods}
    durations_array = np.array(durations)

    for T in durations:
        print(f"Generating signal and benchmarking for T = {T}s...")
        t, y = get_mixed_OU_signals(
            T, dt, lbda_list, omega_list, sigma_list, factor_list
        )
        f_spectro, _, spectro = spectrogram(y, fs, nfft_factor=2)

        mask_f = f_spectro >= 20
        f_M = f_spectro[mask_f]
        M = spectro[mask_f, :]

        for name, func in methods.items():
            # Warm-up call
            func(M, f_M)

            # Timing loop
            timings = []
            for _ in range(n_repeats):
                t_start = time.perf_counter()
                func(M, f_M)
                t_end = time.perf_counter()
                timings.append((t_end - t_start) * 1000.0)  # ms

            results[name]["means"].append(np.mean(timings))
            results[name]["stds"].append(np.std(timings))

    # --- Tabular Summary ---
    col_width_method = max(len(m) for m in methods) + 2
    col_width_dur = 22

    header = f"{'Method':<{col_width_method}}" + "".join(
        [f"{f'T = {T}s (mean ± std ms)':<{col_width_dur}}" for T in durations]
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for name in methods:
        row = f"{name:<{col_width_method}}"
        for idx in range(len(durations)):
            mean_val = results[name]["means"][idx]
            std_val = results[name]["stds"][idx]
            val_str = f"{mean_val:6.2f} ± {std_val:5.2f} ms"
            row += f"{val_str:<{col_width_dur}}"
        print(row)

    print(sep + "\n")

    # --- Plot Comparison ---
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    styles = {
        "Baseline Method": {"marker": "o", "color": "tab:blue"},
        "Quantile Method": {"marker": "s", "color": "tab:orange"},
    }

    for name, data in results.items():
        means = np.array(data["means"])
        stds = np.array(data["stds"])
        ax.errorbar(
            durations_array,
            means,
            yerr=stds,
            label=name,
            capsize=4,
            capthick=1.2,
            linewidth=1.8,
            markersize=6,
            **styles[name],
        )

    ax.set_title("Segmentation Time: Baseline vs. Quantile Method", fontsize=13)
    ax.set_xlabel("Signal Duration $T$ (s)", fontsize=11)
    ax.set_ylabel("Execution Time (ms)", fontsize=11)
    ax.set_xticks(durations_array)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)

    plt.show()


if __name__ == "__main__":
    benchmark_segmentation_modes(durations=[30, 60, 120, 240, 480], n_repeats=25)