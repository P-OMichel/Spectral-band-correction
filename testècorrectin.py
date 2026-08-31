import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.ndimage import distance_transform_edt


import numpy as np
import matplotlib.pyplot as plt
from Functions.generate_OU import get_mixed_OU_signals
from Functions.time_frequency import spectrogram
from Functions.segment_spectrogram import segment_blobs
from scipy.ndimage import binary_dilation

# ====================== Simulate EEG Signal with ketamine signature =================
# --- Parameters
T = 20 # desired signal duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2*np.pi*1, 2*np.pi*10, 2*np.pi*30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 1]

# --- Generate EEG data
t, y = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

# --- compute spectrogram
f_spectro, t_spectro, spectro = spectrogram(y, fs, nfft_factor=2)
M = spectro.copy()
mask_f = f_spectro >= 20
f_M = f_spectro[mask_f]
M = M[mask_f, :]


# =====================================================================
# 1. WHITTAKER ASYMMETRIC LEAST SQUARES (Baseline Target Profile)
# =====================================================================
def whittaker_als_baseline(y, lam=1e4, p=0.01, max_iter=15):
    """
    Computes a smooth baseline profile across the frequency spectrum.
    Bypasses high-intensity transient peaks to uncover the true background floor.
    """
    L = len(y)
    D = np.zeros((L-2, L))
    for i in range(L-2):
        D[i, i], D[i, i+1], D[i, i+2] = 1, -2, 1
    penalty = lam * np.dot(D.T, D)
    w = np.ones(L)
    for _ in range(max_iter):
        W = np.diag(w)
        z = solve(W + penalty, w * y)
        w = np.where(y > z, p, 1 - p)
    return z

# =====================================================================
# 2. SETUP DATA AND GLOBAL MASKING
# =====================================================================
# Assumes your 2D RAW LINEAR matrix 'M' is loaded.
freq_bins, time_bins = M.shape
f_M = np.arange(freq_bins)

original_psd = M.mean(axis=1)
reference_psd = whittaker_als_baseline(original_psd, lam=1e4, p=0.01)

# Set Threshold T (using 50th percentile as requested)
T = np.percentile(M, 50)
mask = M > T

# Isolate the data inside the active mask
M_blob_orig = M[mask]

# --- Spatial Blending Weights Map ---
distance_map = distance_transform_edt(mask)
transition_width = 0.0 
weights = 1.0 - np.exp(-(distance_map / transition_width) ** 2)
weights[~mask] = 0.0
W_blob = weights[mask]

# =====================================================================
# 3. CURVATURE-MATCHING SPECIFICATION OF TARGET CDF (f1)
# =====================================================================
# Sort the unselected data to analyze its trailing edge curvature
M_below_T = np.sort(M[M <= T])
prob_at_T = len(M_below_T) / M.size

# --- ESTIMATE LOCAL CURVATURE SLOPE ---
# Look at the upper 10% region of the unselected data right before threshold T
idx_near_T = int(len(M_below_T) * 0.90)
x_ref = M_below_T[idx_near_T]
p_ref = (idx_near_T / len(M_below_T)) * prob_at_T

# Compute the local linear derivative to match trailing slope direction
local_slope = (prob_at_T - p_ref) / (T - x_ref + 1e-12)

# Define absolute target window boundaries
min_x_above = T
max_x_above = T * 3.0

# Generate target intensity tracking coordinate steps
x_grid_above = np.linspace(min_x_above, max_x_above, 2000)

# --- CURVATURE FIT EXTRAPOLATION ---
# Construct an exponential decay asymptotic curve that launches with the 
# exact empirical slope found at T, but smoothly bends to terminate at 1.0 at 3T.
headroom = 1.0 - prob_at_T
# Calculate the matching decay rate parameter k to fuse the boundaries smoothly
k = local_slope / (headroom + 1e-12)

# Calculate the smooth continuation profile
raw_decay_curve = 1.0 - np.exp(-k * (x_grid_above - min_x_above))
# Normalize the tail explicitly so it spans exactly from prob_at_T up to 1.0 at 3T
f1_probabilities_above = prob_at_T + headroom * (raw_decay_curve / (raw_decay_curve[-1] + 1e-12))

# Combine below and above targets to seal the unified continuous f1 map
f1_x_below = M_below_T
f1_y_below = np.linspace(0.0, prob_at_T, len(M_below_T))

f1_x_curve = np.concatenate([f1_x_below, x_grid_above])
f1_y_curve = np.concatenate([f1_y_below, f1_probabilities_above])

# --- QUANTILE MATCHING OPERATION ---
sorted_all_M = np.sort(M.flatten())
ranks_in_f0 = np.searchsorted(sorted_all_M, M_blob_orig) / len(sorted_all_M)

# Execute numerical interpolation mapping through the custom curvature curve
M_quantile_matched_blob = np.interp(ranks_in_f0, f1_y_curve, f1_x_curve)

# =====================================================================
# 4. RECONSTRUCTION WITH ABSOLUTE BOUNDARY SAFEGUARD
# =====================================================================
M_final = M.copy()

# 1. Smoothly blend the mapped quantile values at the spatial boundaries
M_final[mask] = (1.0 - W_blob) * M[mask] + W_blob * M_quantile_matched_blob

# 2. Hard limit constraint check following alpha-blend step execution
M_final[mask] = np.clip(M_final[mask], None, max_x_above)

M_difference = M - M_final

# =====================================================================
# 5. VISUALIZATION (SPECTROGRAMS, PSDS, AND CONTINUOUS VIEW COHORTS)
# =====================================================================
fig, axes = plt.subplots(6, 1, figsize=(12, 28))

# Panel 1: Original Spectrogram
im0 = axes[0].pcolormesh(M, shading='auto', cmap='jet')
axes[0].contour(mask)
axes[0].set_title(f"Original Linear Spectrogram Matrix M (True Max: {M.max():.2e})")
axes[0].set_ylabel("Frequency Bins")
fig.colorbar(im0, ax=axes[0], label="Linear Power")

# Panel 2: Quantile-Matched Spectrogram
im1 = axes[1].pcolormesh(M_final, shading='auto', cmap='jet', vmin=M.min(), vmax=M.max())
axes[1].set_title(f"Quantile-Matched Spectrogram (Guaranteed Max Ceiling: {M_final[mask].max():.2e})")
axes[1].set_ylabel("Frequency Bins")
fig.colorbar(im1, ax=axes[1], label="Linear Power")

# Panel 3: 2D Spectrogram Difference Map
im2 = axes[2].pcolormesh(M_difference, shading='auto', cmap='jet')
axes[2].set_title("2D Spectrogram Difference Map (Energy Removed Via Quantile Specification)")
axes[2].set_ylabel("Frequency Bins")
fig.colorbar(im2, ax=axes[2], label="Amplitude Reduced")

# Panel 4: Linear PSD Curve Comparison
axes[3].plot(f_M, original_psd, label='PSD Original Spectrogram', color='red', alpha=0.6)
axes[3].plot(f_M, reference_psd, label='Whittaker Target Baseline', color='black', linestyle='--')
axes[3].plot(f_M, np.mean(M_final, axis=1), label='PSD Cleaned Spectrogram', color='green', linewidth=2)
axes[3].set_title("PSD Profile Verification Analysis")
axes[3].set_ylabel("Mean Linear Intensity")
axes[3].legend()
axes[3].grid(True)

# Panel 5: GLOBAL INTENSITY CDF VISUALIZATION
sorted_M_orig = np.sort(M.flatten())
f0_y = np.linspace(0.0, 1.0, len(sorted_M_orig))

sorted_M_final = np.sort(M_final.flatten())
f_final_y = np.linspace(0.0, 1.0, len(sorted_M_final))

axes[4].plot(sorted_M_orig, f0_y, label='Original Global CDF (f0)', color='red', alpha=0.5, linewidth=3)
axes[4].plot(f1_x_curve, f1_y_curve, label='Curvature-Matched Target CDF (f1)', color='blue', linestyle='--', linewidth=2)
axes[4].plot(sorted_M_final, f_final_y, label='Actual Resulting Global CDF', color='green', linewidth=2)
axes[4].axvline(T, color='purple', linestyle=':', label=f'Threshold T ({T:.2e})')
axes[4].axvline(max_x_above, color='cyan', linestyle=':', label=f'Hard Ceiling Upper Limit 3T ({max_x_above:.2e})')
axes[4].set_title("Global Intensity Cumulative Distribution Function (CDF) Specification")
axes[4].set_ylabel("Probability")
axes[4].set_xscale('log')
axes[4].legend()
axes[4].grid(True, which="both", ls="-")

# Panel 6: SUB-POPULATION COHORT CDF (Selected vs. Unselected Pixels)
unselected_pixels = np.sort(M[~mask])
selected_pixels_orig = np.sort(M_blob_orig)
selected_pixels_final = np.sort(M_final[mask])

y_unsel = np.linspace(0.0, 1.0, len(unselected_pixels))
y_sel_orig = np.linspace(0.0, 1.0, len(selected_pixels_orig))
y_sel_final = np.linspace(0.0, 1.0, len(selected_pixels_final))

axes[5].plot(unselected_pixels, y_unsel, label=f'Unselected Pixels (M <= T)', color='gray', linestyle='-', linewidth=2)
axes[5].plot(selected_pixels_orig, y_sel_orig, label=f'Selected Mask Pixels (Original Tail)', color='darkred', alpha=0.6, linewidth=2.5)
axes[5].plot(selected_pixels_final, y_sel_final, label=f'Selected Mask Pixels (Transformed Curvature Tail)', color='darkgreen', linewidth=2.5)

axes[5].axvline(T, color='purple', linestyle=':', label=f'Threshold T ({T:.2e})')
axes[5].axvline(max_x_above, color='cyan', linestyle=':', label=f'Max 3T Boundary ({max_x_above:.2e})')
axes[5].set_title("Isolated Sub-Population Empirical CDF Comparison")
axes[5].set_xlabel("Pixel Amplitude Value (Log Scale)")
axes[5].set_ylabel("Internal Cohort Probability")
axes[5].set_xscale('log')
axes[5].legend()
axes[5].grid(True, which="both", ls="-")

plt.tight_layout()
plt.show()