import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------------------------
loaded = np.load(
    "spectrogram_correction_evaluation_dataset_factor.npz", allow_pickle=True
)

metadata = loaded["metadata"].item()
original = loaded["original"].item()
methods = loaded["methods"].item()

factor_x_values = metadata["factor_x_values"]
factor_x_per_run = metadata["factor_x_per_run"]
f_full = metadata["f_full"]
f_int = metadata["f_int"]

method_styles = {
    "Alpha-Blended": {"color": "tab:blue", "marker": "o", "ls": "-"},
    "Monotonic Quantile": {"color": "tab:orange", "marker": "^", "ls": "--"},
    "Rank-Preserved Alpha": {"color": "tab:green", "marker": "s", "ls": "-."},
}

# -------------------------------------------------------------------------
# 2. Compute Mean and Std across Realizations per Factor
# -------------------------------------------------------------------------
orig_rmse_mean = []
orig_rmse_std = []
orig_sef_mean = []
orig_sef_std = []

for fac in factor_x_values:
    mask = factor_x_per_run == fac
    orig_rmse_mean.append(np.mean(original["orig_vs_baseline_rmse"][mask]))
    orig_rmse_std.append(np.std(original["orig_vs_baseline_rmse"][mask]))
    orig_sef_mean.append(np.mean(original["sef95"][mask]))
    orig_sef_std.append(np.std(original["sef95"][mask]))

method_stats = {}
for name in methods:
    method_stats[name] = {
        "rmse_mean": [],
        "rmse_std": [],
        "sef_mean": [],
        "sef_std": [],
    }
    for fac in factor_x_values:
        mask = factor_x_per_run == fac
        method_stats[name]["rmse_mean"].append(
            np.mean(methods[name]["rmse_vs_baseline"][mask])
        )
        method_stats[name]["rmse_std"].append(
            np.std(methods[name]["rmse_vs_baseline"][mask])
        )
        method_stats[name]["sef_mean"].append(
            np.mean(methods[name]["sef95"][mask])
        )
        method_stats[name]["sef_std"].append(
            np.std(methods[name]["sef95"][mask])
        )

# -------------------------------------------------------------------------
# 3. Figure 1: RMSE vs. Bump Scaling Factor
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

ax.errorbar(
    factor_x_values,
    orig_rmse_mean,
    yerr=orig_rmse_std,
    label="Original (Uncorrected vs. Baseline)",
    color="black",
    linestyle=":",
    marker="x",
    capsize=3,
    linewidth=1.8,
)

for name, style in method_styles.items():
    ax.errorbar(
        factor_x_values,
        method_stats[name]["rmse_mean"],
        yerr=method_stats[name]["rmse_std"],
        label=name,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["ls"],
        capsize=3,
        linewidth=1.6,
        markersize=5,
    )

ax.set_title(
    r"Reconstruction RMSE vs. 30 Hz Bump Factor within $[25, 35]$ Hz",
    fontsize=12,
)
ax.set_xlabel("30 Hz Bump Factor", fontsize=11)
ax.set_ylabel(r"RMSE ($\log_2\text{ PSD vs. Baseline}$)", fontsize=11)
ax.set_xticks(factor_x_values)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9, loc="best")
plt.show()

# -------------------------------------------------------------------------
# 4. Figure 2: SEF95 vs. Bump Scaling Factor
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

ax.errorbar(
    factor_x_values,
    orig_sef_mean,
    yerr=orig_sef_std,
    label="Original (Uncorrected)",
    color="black",
    linestyle=":",
    marker="x",
    capsize=3,
    linewidth=1.8,
)

for name, style in method_styles.items():
    ax.errorbar(
        factor_x_values,
        method_stats[name]["sef_mean"],
        yerr=method_stats[name]["sef_std"],
        label=name,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["ls"],
        capsize=3,
        linewidth=1.6,
        markersize=5,
    )

ax.set_title(
    r"Spectral Edge Frequency ($\text{SEF}_{95}$) vs. 30 Hz Bump Factor",
    fontsize=12,
)
ax.set_xlabel("30 Hz Bump Factor", fontsize=11)
ax.set_ylabel(r"$\text{SEF}_{95}$ (Hz)", fontsize=11)
ax.set_xticks(factor_x_values)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9, loc="best")
plt.show()

# -------------------------------------------------------------------------
# 5. Figure 3: Sample Spectral Profiles (Lowest vs. Highest Factor)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

eval_factors = [factor_x_values[0], factor_x_values[-1]]

for ax, fac in zip(axes, eval_factors):
    indices = np.where(factor_x_per_run == fac)[0]
    sample_idx = indices[0]

    ax.plot(
        f_full,
        np.log2(original["psd"][sample_idx] + 1e-11),
        label="Original PSD",
        color="red",
        linewidth=1.5,
    )

    for name, style in method_styles.items():
        ax.plot(
            f_full,
            np.log2(methods[name]["psd_corrected"][sample_idx] + 1e-11),
            label=name,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.2,
        )

    ax.axvspan(f_int[0], f_int[1], color="gray", alpha=0.15, label="Target ROI")
    ax.set_xlim([0, 50])
    ax.set_title(f"Sample Log-PSD (Bump Factor = {fac:.1f})", fontsize=11)
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel(r"$\log_2(\text{PSD})$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")

plt.show()