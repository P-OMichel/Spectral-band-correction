'''
File to test the funcitons of segmentation and correction of ketamine spectral signature.
'''

import numpy as np
import matplotlib.pyplot as plt
from Functions.generate_OU import get_mixed_OU_signals
from Functions.time_frequency import spectrogram
from Functions.segment_spectrogram import segment_blobs
from scipy.ndimage import binary_dilation, label

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

# --- Display
fig, axes = plt.subplots(4, constrained_layout = True)
axes[0].plot(t, y)
axes[0].set_title('Simulated EEG signal')
axes[1].pcolormesh(t_spectro, f_spectro, np.log2(spectro + 1e-11), shading = 'nearest', cmap = 'jet')
axes[1].set_title('Spectrogram')
axes[2].plot(f_spectro, np.log2(np.median(spectro, axis = 1)))
axes[2].set_title('PSD')
axes[1].sharex(axes[0])
axes[3].pcolormesh(t_spectro, f_M, np.log2(M + 1e-11), shading = 'nearest', cmap = 'jet')
axes[3].sharex(axes[0])
axes[1].set_title('Spectrogram above 20 Hz')

plt.show()


# ====================== Segmentation of ketamine like signature =================
# --- segment 
f_int = [25, 35]
method = 'quantile'
seg_mask, psd, baseline, T_low, T_high = segment_blobs(M, f_M, f_int)
print(f'T_low: {T_low}')

fig, axes = plt.subplots(4, constrained_layout = True)
axes[0].pcolormesh(t_spectro, f_M, np.log2(M + 1e-11), shading = 'nearest', cmap = 'jet')
axes[0].set_title('Spectrogram above 20 Hz')
axes[1].pcolormesh(t_spectro, f_M, seg_mask, shading = 'nearest', cmap = 'jet')
axes[1].axhline(f_int[0], color = 'white', linestyle = '-')
axes[1].axhline(f_int[-1], color = 'white', linestyle = '-')
axes[1].set_title('Segmented blobs')
axes[2].pcolormesh(t_spectro, f_M, np.log2(M + 1e-11), shading = 'nearest', cmap = 'jet')
axes[2].contour(t_spectro, f_M, seg_mask, colors = 'white', linewidths=1)
axes[3].plot(f_M, psd)
axes[3].plot(f_M, baseline)
axes[1].sharex(axes[0])
axes[2].sharex(axes[0])

plt.show()

# ================== Correct spectrogram (Strategy 1: Sigmoidal Shift) ===============

# Flatten the data and split by mask boolean state
spectro_log = np.log2(M + 1e-11)
inside = np.sort(spectro_log[seg_mask.astype(bool)])
outside = np.sort(spectro_log[~seg_mask.astype(bool)])

# Compute ECDF y-values
y_inside = np.linspace(0, 1, len(inside))
y_outside = np.linspace(0, 1, len(outside))

# Plot ECDFs
plt.figure()
plt.plot(inside, y_inside, label='Inside Mask')
plt.plot(outside, y_outside, label='Outside Mask')
plt.xlabel('Log Spectrogram Intensity')
plt.ylabel('ECDF')
plt.legend()
plt.grid(True)
plt.show()

# ================================#
# --- Get neighboring set & Strategy 1
# ================================#

# --- 1. Prepare Data & Masks ---
spectro_log = np.log2(M + 1e-11)
seg_bool = seg_mask.astype(bool)

# Expand mask by 2 iterations
expanded_mask = binary_dilation(seg_bool, iterations=2)
border_mask = expanded_mask & ~seg_bool

# Unpack values
inside_raw = spectro_log[seg_bool]
local_outside_vals = np.sort(spectro_log[border_mask])

# Sorted inside values & indices for rank preservation
sort_indices = np.argsort(inside_raw)
inside_vals = inside_raw[sort_indices]

# ECDF y-values (quantiles)
y_inside = np.linspace(0, 1, len(inside_vals))
y_local_outside = np.linspace(0, 1, len(local_outside_vals))

# --- 2. Construct Smooth Sigmoidal Transformation ---
max_outside_val = local_outside_vals[-1]
new_inside_vals = inside_vals.copy()

mask_above = inside_vals > max_outside_val

if np.any(mask_above):
    x_above = inside_vals[mask_above]
    
    # Measure elevation above the max outside threshold
    delta_x = x_above - max_outside_val
    
    # Parameters for Sigmoid Shape
    L = np.max(delta_x) * 0.5   # Maximum shift (plateau level, e.g., 50% attenuation)
    k = 1.2                     # Growth steepness (controls how smoothly the S-curve bends)
    x0 = np.median(delta_x)     # Inflection point centered halfway through peak range
    
    # Logistic Sigmoid S-Curve mapping: S(x) = L / (1 + exp(-k * (x - x0)))
    # Subtracted from original values to gently compress high peaks
    sigmoid_shift = L / (1.0 + np.exp(-k * (delta_x - x0)))
    
    # Shift values and anchor smoothly to max_outside_val at delta_x = 0
    # Subtracting sigmoid_shift(0) guarantees exact C0 junction at threshold
    shift_anchored = sigmoid_shift - (L / (1.0 + np.exp(k * x0)))
    
    transformed_above = x_above - shift_anchored
    
    # Enforce strict rank monotonicity
    new_inside_vals[mask_above] = np.maximum.accumulate(transformed_above)

# --- 3. Apply Remapped Values Back to Spectrogram ---
corrected_spectro_log = spectro_log.copy()

remapped_inside = np.empty_like(new_inside_vals)
remapped_inside[sort_indices] = new_inside_vals
corrected_spectro_log[seg_bool] = remapped_inside

# --- Figure 1: Diagnostic Subplots & ECDF ---
fig1, axes1 = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)

# 1. Original Spectrogram
pcm0 = axes1[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet')
axes1[0].set_title('Spectrogram (log2 intensity)')
axes1[0].set_ylabel('Frequency (Hz)')
fig1.colorbar(pcm0, ax=axes1[0], label='Log Intensity')

# 2. Original Segmented Mask
axes1[1].pcolormesh(t_spectro, f_M, seg_bool, shading='nearest', cmap='binary')
axes1[1].set_title('Original Segmented Mask')
axes1[1].set_ylabel('Frequency (Hz)')

# 3. Expanded Mask & Border Overlay
mask_display = np.zeros_like(seg_bool, dtype=int)
mask_display[seg_bool] = 1
mask_display[border_mask] = 2

axes1[2].pcolormesh(t_spectro, f_M, mask_display, shading='nearest', cmap='Blues')
axes1[2].set_title('Expanded Mask (Dark Blue) & Border Ring (Light Blue)')
axes1[2].set_ylabel('Frequency (Hz)')

# 4. ECDFs
axes1[3].plot(local_outside_vals, y_local_outside, label='Outside (within 2 bins)', color='black')
axes1[3].plot(inside_vals, y_inside, label='Original Inside Mask', color='red', linestyle='--')
axes1[3].plot(new_inside_vals, y_inside, label='Corrected Inside Mask (Sigmoidal Shift)', color='green')
axes1[3].axvline(max_outside_val, color='gray', linestyle=':', label='Max Outside Threshold')
axes1[3].set_xlabel('Log Spectrogram Intensity')
axes1[3].set_ylabel('ECDF')
axes1[3].set_title('ECDF Matching & Transformation (Sigmoidal)')
axes1[3].legend()
axes1[3].grid(True)
log2_T_low_max = np.log2(np.max(T_low) + 1e-11)
axes1[3].axvline(log2_T_low_max, color='orange', linestyle='--', label='T_low Threshold (Max)')
axes1[3].legend()

plt.show()

# --- Figure 2: Original vs. Corrected Spectrogram (Sigmoidal) ---
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True)

# Determine shared color limits for fair comparison
vmin = min(spectro_log.min(), corrected_spectro_log.min())
vmax = max(spectro_log.max(), corrected_spectro_log.max())

# Original Spectrogram with Overlay
pcm_orig = axes2[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes2[0].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes2[0].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes2[0].set_title('Original Spectrogram (with Mask Contour)')
axes2[0].set_ylabel('Frequency (Hz)')
fig2.colorbar(pcm_orig, ax=axes2[0], label='Log Intensity')

# Corrected Spectrogram with Overlay
pcm_corr = axes2[1].pcolormesh(t_spectro, f_M, corrected_spectro_log, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes2[1].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes2[1].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes2[1].set_title('Corrected Spectrogram (Sigmoidal Peak Attenuation)')
axes2[1].set_xlabel('Time (s)')
axes2[1].set_ylabel('Frequency (Hz)')
fig2.colorbar(pcm_corr, ax=axes2[1], label='Log Intensity')

plt.show()

# ================== Correct spectrogram (Strategy 2: Monotonic Floor) ===============

# --- 1. Initialize Arrays ---
corrected_monotonic_log = spectro_log.copy()

# Label distinct connected components (blobs)
labeled_mask, num_blobs = label(seg_bool, structure=np.ones((3, 3)))

# --- 2. Minimal ECDF Projection Blob-by-Blob ---
for blob_id in range(1, num_blobs + 1):
    blob_idx = (labeled_mask == blob_id)
    
    # Extract 2-bin local border ring specifically for THIS blob
    blob_expanded = binary_dilation(blob_idx, iterations=2)
    blob_border = blob_expanded & ~blob_idx
    
    if not np.any(blob_border):
        continue

    # A. Frequency-Specific Floor Mapping
    freq_indices, _ = np.where(blob_idx)
    unique_freqs = np.unique(freq_indices)
    
    f_floor_map = {}
    for f_idx in unique_freqs:
        border_pixels_at_f = spectro_log[f_idx, :][blob_border[f_idx, :]]
        if len(border_pixels_at_f) > 0:
            f_floor_map[f_idx] = np.max(border_pixels_at_f)
        else:
            # Fallback to general border max if border at f is empty
            f_floor_map[f_idx] = np.max(spectro_log[blob_border])

    # B. Extract Pixels & Map Frequency Floors
    blob_coords = np.argwhere(blob_idx)  # Array of [f, t]
    raw_vals = spectro_log[blob_idx]
    
    pixel_floors = np.array([f_floor_map[coord[0]] for coord in blob_coords])

    # C. Sort Pixels by Rank Ordering
    sort_idx = np.argsort(raw_vals)
    sorted_floors = pixel_floors[sort_idx]

    # D. Compute Minimal Rank-Preserving Monotonic Lower Bound
    min_rank_vals = np.maximum.accumulate(sorted_floors)

    # E. Remap Back to Pixel Rank Positions
    remapped_blob = np.empty_like(min_rank_vals)
    remapped_blob[sort_idx] = min_rank_vals
    corrected_monotonic_log[blob_idx] = remapped_blob

# Extract sorted array for strategy 2 ECDF plotting
monotonic_inside_vals = np.sort(corrected_monotonic_log[seg_bool])

# --- Figure 3: Diagnostic Subplots & ECDF (Strategy 2) ---
fig3, axes3 = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)

# 1. Original Spectrogram
axes3[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet')
axes3[0].set_title('Spectrogram (log2 intensity)')
axes3[0].set_ylabel('Frequency (Hz)')

# 2. Original Segmented Mask
axes3[1].pcolormesh(t_spectro, f_M, seg_bool, shading='nearest', cmap='binary')
axes3[1].set_title('Original Segmented Mask')
axes3[1].set_ylabel('Frequency (Hz)')

# 3. Expanded Mask & Border Overlay
axes3[2].pcolormesh(t_spectro, f_M, mask_display, shading='nearest', cmap='Blues')
axes3[2].set_title('Expanded Mask (Dark Blue) & Border Ring (Light Blue)')
axes3[2].set_ylabel('Frequency (Hz)')

# 4. ECDFs Comparison
axes3[3].plot(local_outside_vals, y_local_outside, label='Outside (within 2 bins)', color='black')
axes3[3].plot(inside_vals, y_inside, label='Original Inside Mask', color='red', linestyle='--')
axes3[3].plot(monotonic_inside_vals, y_inside, label='Corrected Inside Mask (Monotonic Floor)', color='purple')
axes3[3].axvline(max_outside_val, color='gray', linestyle=':', label='Max Outside Threshold')
axes3[3].axvline(log2_T_low_max, color='orange', linestyle='--', label='T_low Threshold (Max)')
axes3[3].set_xlabel('Log Spectrogram Intensity')
axes3[3].set_ylabel('ECDF')
axes3[3].set_title('ECDF Matching & Transformation (Monotonic Floor Projection)')
axes3[3].legend()
axes3[3].grid(True)

plt.show()

# --- Figure 4: Original vs. Corrected Spectrogram (Monotonic Floor Strategy) ---
fig4, axes4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True)

vmin_m = min(spectro_log.min(), corrected_monotonic_log.min())
vmax_m = max(spectro_log.max(), corrected_monotonic_log.max())

# Original Spectrogram
pcm_orig4 = axes4[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet', vmin=vmin_m, vmax=vmax_m)
axes4[0].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes4[0].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes4[0].set_title('Original Spectrogram (with Mask Contour)')
axes4[0].set_ylabel('Frequency (Hz)')
fig4.colorbar(pcm_orig4, ax=axes4[0], label='Log Intensity')

# Monotonic Floor Corrected Spectrogram
pcm_corr4 = axes4[1].pcolormesh(t_spectro, f_M, corrected_monotonic_log, shading='nearest', cmap='jet', vmin=vmin_m, vmax=vmax_m)
axes4[1].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes4[1].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes4[1].set_title('Corrected Spectrogram (Monotonic Floor Lower-Bound Projection)')
axes4[1].set_xlabel('Time (s)')
axes4[1].set_ylabel('Frequency (Hz)')
fig4.colorbar(pcm_corr4, ax=axes4[1], label='Log Intensity')

plt.show()