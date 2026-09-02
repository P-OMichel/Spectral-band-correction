'''
File to test the funcitons of segmentation and correction of ketamine spectral signature.
'''

import numpy as np
import matplotlib.pyplot as plt
from Functions.generate_OU import get_mixed_OU_signals
from Functions.time_frequency import spectrogram
from Functions.segment_spectrogram import segment_blobs
from scipy.ndimage import binary_dilation, label
from sklearn.isotonic import IsotonicRegression

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
    sigmoid_shift = L / (1.0 + np.exp(-k * (delta_x - x0)))
    
    # Shift values and anchor smoothly to max_outside_val at delta_x = 0
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

pcm0 = axes1[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet')
axes1[0].set_title('Spectrogram (log2 intensity)')
axes1[0].set_ylabel('Frequency (Hz)')
fig1.colorbar(pcm0, ax=axes1[0], label='Log Intensity')

axes1[1].pcolormesh(t_spectro, f_M, seg_bool, shading='nearest', cmap='binary')
axes1[1].set_title('Original Segmented Mask')
axes1[1].set_ylabel('Frequency (Hz)')

mask_display = np.zeros_like(seg_bool, dtype=int)
mask_display[seg_bool] = 1
mask_display[border_mask] = 2

axes1[2].pcolormesh(t_spectro, f_M, mask_display, shading='nearest', cmap='Blues')
axes1[2].set_title('Expanded Mask (Dark Blue) & Border Ring (Light Blue)')
axes1[2].set_ylabel('Frequency (Hz)')

axes1[3].plot(local_outside_vals, y_local_outside, label='Outside (within 2 bins)', color='black')
axes1[3].plot(inside_vals, y_inside, label='Original Inside Mask', color='red', linestyle='--')
axes1[3].plot(new_inside_vals, y_inside, label='Corrected Inside Mask (Sigmoidal Shift)', color='green')
axes1[3].axvline(max_outside_val, color='gray', linestyle='--', label='T_correction')
axes1[3].set_xlabel('Log Spectrogram Intensity')
axes1[3].set_ylabel('ECDF')
axes1[3].set_title('ECDF Matching & Transformation (Sigmoidal)')
axes1[3].legend()
axes1[3].grid(True)
log2_T_low_max = np.log2(np.max(T_low) + 1e-11)
axes1[3].axvline(log2_T_low_max, color='orange', linestyle='--', label='T_segmentation')
axes1[3].legend()

plt.show()


# --- Figure 2: Original vs. Corrected Spectrogram (Sigmoidal) ---
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)

vmin = min(spectro_log.min(), corrected_spectro_log.min())
vmax = max(spectro_log.max(), corrected_spectro_log.max())

pcm_orig = axes2[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes2[0].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes2[0].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes2[0].set_title('Original Spectrogram (with Mask Contour)')
axes2[0].set_ylabel('Frequency (Hz)')
fig2.colorbar(pcm_orig, ax=axes2[0], label='Log Intensity')

pcm_corr = axes2[1].pcolormesh(t_spectro, f_M, corrected_spectro_log, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes2[1].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes2[1].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes2[1].set_title('Corrected Spectrogram (Sigmoidal Peak Attenuation)')
axes2[1].set_xlabel('Time (s)')
axes2[1].set_ylabel('Frequency (Hz)')
fig2.colorbar(pcm_corr, ax=axes2[1], label='Log Intensity')

axes2[2].plot(local_outside_vals, y_local_outside, label='Outside (within 2 bins)', color='black')
axes2[2].plot(inside_vals, y_inside, label='Original Inside Mask', color='red', linestyle='--')
axes2[2].plot(new_inside_vals, y_inside, label='Corrected Inside Mask (Sigmoidal Shift)', color='green')
axes2[2].axvline(max_outside_val, color='gray', linestyle='--', label='T_correction')
axes2[2].set_xlabel('Log Spectrogram Intensity')
axes2[2].set_ylabel('ECDF')
axes2[2].set_title('ECDF Matching & Transformation (Sigmoidal)')
axes2[2].legend()
axes2[2].grid(True)
log2_T_low_max = np.log2(np.max(T_low) + 1e-11)
axes2[2].axvline(log2_T_low_max, color='orange', linestyle='--', label='T_segmentation')
axes2[2].legend()

plt.show()

plt.show()

'''
# ================== Correct spectrogram (Strategy 2: Time-Local Monotonic Floor) ===============

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

    # A. Extract Exact Maximum Local Border Floor in 2D Space (Time + Frequency)
    blob_coords = np.argwhere(blob_idx)
    raw_vals = spectro_log[blob_idx]
    
    pixel_floors = []
    for f_i, t_i in blob_coords:
        # 3x3 time-frequency neighborhood in the border
        f_min, f_max = max(0, f_i - 2), min(spectro_log.shape[0], f_i + 3)
        t_min, t_max = max(0, t_i - 2), min(spectro_log.shape[1], t_i + 3)
        
        local_border_window = blob_border[f_min:f_max, t_min:t_max]
        local_spectro_window = spectro_log[f_min:f_max, t_min:t_max]
        
        border_vals_in_win = local_spectro_window[local_border_window]
        
        if len(border_vals_in_win) > 0:
            # Use exact max to respect the boundary constraint
            pixel_floors.append(np.max(border_vals_in_win))
        else:
            border_at_f = spectro_log[f_i, :][blob_border[f_i, :]]
            pixel_floors.append(np.max(border_at_f) if len(border_at_f) > 0 else np.max(spectro_log[blob_border]))

    pixel_floors = np.array(pixel_floors)

    # B. Enforce Constraint: Inside Pixel >= Local Neighbor Floor
    # (Attenuates peaks towards the local border floor, but never below it)
    sort_idx = np.argsort(raw_vals)
    sorted_raw = raw_vals[sort_idx]
    sorted_floors = pixel_floors[sort_idx]

    # C. Compute Minimal Monotonic Sequence
    # Guarantees rank monotonicity AND that inside values >= local border max
    min_rank_vals = np.maximum.accumulate(sorted_floors)

    # Cap at raw values to ensure attenuation only (no accidental amplification)
    min_rank_vals = np.minimum(min_rank_vals, sorted_raw)
    
    # Final monotonic sweep
    min_rank_vals = np.maximum.accumulate(min_rank_vals)

    # D. Remap Back to Spatial Positions
    remapped_blob = np.empty_like(min_rank_vals)
    remapped_blob[sort_idx] = min_rank_vals
    corrected_monotonic_log[blob_idx] = remapped_blob

# Extract sorted array for strategy 2 ECDF plotting
monotonic_inside_vals = np.sort(corrected_monotonic_log[seg_bool])

# --- Figure 3: Diagnostic Subplots & ECDF (Strategy 2) ---
fig3, axes3 = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)

axes3[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet')
axes3[0].set_title('Spectrogram (log2 intensity)')
axes3[0].set_ylabel('Frequency (Hz)')

axes3[1].pcolormesh(t_spectro, f_M, seg_bool, shading='nearest', cmap='binary')
axes3[1].set_title('Original Segmented Mask')
axes3[1].set_ylabel('Frequency (Hz)')

axes3[2].pcolormesh(t_spectro, f_M, mask_display, shading='nearest', cmap='Blues')
axes3[2].set_title('Expanded Mask (Dark Blue) & Border Ring (Light Blue)')
axes3[2].set_ylabel('Frequency (Hz)')

axes3[3].plot(local_outside_vals, y_local_outside, label='Outside (within 2 bins)', color='black')
axes3[3].plot(inside_vals, y_inside, label='Original Inside Mask', color='red', linestyle='--')
axes3[3].plot(monotonic_inside_vals, y_inside, label='Corrected Inside Mask (Time-Local Monotonic Floor)', color='purple')
axes3[3].axvline(max_outside_val, color='gray', linestyle=':', label='Max Outside Threshold')
axes3[3].axvline(log2_T_low_max, color='orange', linestyle='--', label='T_low Threshold (Max)')
axes3[3].set_xlabel('Log Spectrogram Intensity')
axes3[3].set_ylabel('ECDF')
axes3[3].set_title('ECDF Matching & Transformation (Isotonic Projection)')
axes3[3].legend()
axes3[3].grid(True)

plt.show()

# --- Figure 4: Original vs. Corrected Spectrogram (Monotonic Floor Strategy) ---
fig4, axes4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True)

vmin_m = min(spectro_log.min(), corrected_monotonic_log.min())
vmax_m = max(spectro_log.max(), corrected_monotonic_log.max())

pcm_orig4 = axes4[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet', vmin=vmin_m, vmax=vmax_m)
axes4[0].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes4[0].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes4[0].set_title('Original Spectrogram (with Mask Contour)')
axes4[0].set_ylabel('Frequency (Hz)')
fig4.colorbar(pcm_orig4, ax=axes4[0], label='Log Intensity')

pcm_corr4 = axes4[1].pcolormesh(t_spectro, f_M, corrected_monotonic_log, shading='nearest', cmap='jet', vmin=vmin_m, vmax=vmax_m)
axes4[1].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes4[1].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes4[1].set_title('Corrected Spectrogram (Isotonic Monotonic Floor Projection)')
axes4[1].set_xlabel('Time (s)')
axes4[1].set_ylabel('Frequency (Hz)')
fig4.colorbar(pcm_corr4, ax=axes4[1], label='Log Intensity')

plt.show()
'''

# ================== Correct spectrogram (Strategy 2: Single Global Sigmoid Fit) ===============

corrected_monotonic_log = spectro_log.copy()
corrected_sigmoid_log = spectro_log.copy()

# 1. Global extraction of inside and local floor arrays
seg_bool = seg_mask.astype(bool)
raw_inside = spectro_log[seg_bool]
sort_idx = np.argsort(raw_inside)
sorted_raw = raw_inside[sort_idx]

# Extract local border floor per pixel globally
expanded_mask = binary_dilation(seg_bool, iterations=2)
border_mask = expanded_mask & ~seg_bool

blob_coords = np.argwhere(seg_bool)
pixel_floors = []
for f_i, t_i in blob_coords:
    f_min, f_max = max(0, f_i - 2), min(spectro_log.shape[0], f_i + 3)
    t_min, t_max = max(0, t_i - 2), min(spectro_log.shape[1], t_i + 3)
    local_win = border_mask[f_min:f_max, t_min:t_max]
    spectro_win = spectro_log[f_min:f_max, t_min:t_max]
    border_vals = spectro_win[local_win]
    if len(border_vals) > 0:
        pixel_floors.append(np.max(border_vals))
    else:
        pixel_floors.append(np.max(spectro_log[border_mask]))

pixel_floors = np.array(pixel_floors)[sort_idx]

# Linear Monotonic Floor Sequence (Purple Curve)
min_rank_vals = np.maximum.accumulate(pixel_floors)
min_rank_vals = np.minimum(min_rank_vals, sorted_raw)
min_rank_vals = np.maximum.accumulate(min_rank_vals)

# 2. Locate the "Extreme Switch Point" (knee with maximum argument / x-value)
# Find where min_rank_vals achieves its maximum vertical shift towards log2_T_low_max
knee_idx = np.argmax(min_rank_vals) 
x_knee = min_rank_vals[knee_idx]
y_knee = (knee_idx + 1) / len(min_rank_vals)

# Define exact anchors for the single global sigmoid:
# Point 1: The extreme switch knee point (x_knee, y_knee)
# Point 2: Upper plateau anchor slightly to the right
x1, y1 = x_knee, np.clip(y_knee, 0.05, 0.95)
x2, y2 = x_knee + 0.50, 0.99  # Delta x = +0.50 log2 units creates the upper right plateau

# Solve parameters k and x0 using logit transform: logit(y) = ln(y / (1 - y))
logit_y1 = np.log(y1 / (1.0 - y1))
logit_y2 = np.log(y2 / (1.0 - y2))

k_sig = (logit_y2 - logit_y1) / (x2 - x1)
x0_sig = x1 - (logit_y1 / k_sig)

# 3. Generate smooth intensity values directly from ECDF ranks y in (0, 1)
N = len(sorted_raw)
y_ranks = np.linspace(0.001, 0.999, N)

# Invert sigmoid: x = x0 + logit(y) / k
sigmoid_vals = x0_sig + (np.log(y_ranks / (1.0 - y_ranks)) / k_sig)

# Enforce Monotonicity & Floor (Ensures curve stays to the right of purple baseline)
final_sigmoid_vals = np.maximum(sigmoid_vals, min_rank_vals)
final_sigmoid_vals = np.minimum(final_sigmoid_vals, sorted_raw)
final_sigmoid_vals = np.maximum.accumulate(final_sigmoid_vals)

# Remap back to 2D matrix
remapped_mono = np.empty_like(min_rank_vals)
remapped_mono[sort_idx] = min_rank_vals
corrected_monotonic_log[seg_bool] = remapped_mono

remapped_sig = np.empty_like(final_sigmoid_vals)
remapped_sig[sort_idx] = final_sigmoid_vals
corrected_sigmoid_log[seg_bool] = remapped_sig

# Extract sorted arrays for ECDF plotting
monotonic_inside_vals = np.sort(corrected_monotonic_log[seg_bool])
sigmoid_inside_vals = np.sort(corrected_sigmoid_log[seg_bool])

# --- Figure 3: Diagnostic Subplots & ECDF Comparison ---
fig3, axes3 = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)

axes3[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet')
axes3[0].set_title('Spectrogram (log2 intensity)')
axes3[0].set_ylabel('Frequency (Hz)')

axes3[1].pcolormesh(t_spectro, f_M, seg_bool, shading='nearest', cmap='binary')
axes3[1].set_title('Original Segmented Mask')
axes3[1].set_ylabel('Frequency (Hz)')

axes3[2].pcolormesh(t_spectro, f_M, mask_display, shading='nearest', cmap='Blues')
axes3[2].set_title('Expanded Mask (Dark Blue) & Border Ring (Light Blue)')
axes3[2].set_ylabel('Frequency (Hz)')

# Plot ECDFs
axes3[3].plot(local_outside_vals, y_local_outside, label='Outside (within 2 bins)', color='black')
axes3[3].plot(inside_vals, y_inside, label='Original Inside Mask', color='red', linestyle='--')
axes3[3].plot(monotonic_inside_vals, y_inside, label='Corrected Inside (Linear Monotonic Floor)', color='purple')
axes3[3].plot(sigmoid_inside_vals, y_inside, label='Corrected Inside (True Sigmoid ECDF)', color='deepskyblue', linewidth=2.0)
axes3[3].axvline(max_outside_val, color='gray', linestyle=':', label='Max Outside Threshold')
axes3[3].axvline(log2_T_low_max, color='orange', linestyle='--', label='T_low Threshold (Max)')
axes3[3].set_xlabel('Log Spectrogram Intensity')
axes3[3].set_ylabel('ECDF')
axes3[3].set_title('ECDF Matching & Reshaping Comparison')
axes3[3].legend()
axes3[3].grid(True)

plt.show()

# --- Figure 4: Spectrogram Comparison ---
fig4, axes4 = plt.subplots(4, 1, figsize=(10, 9), constrained_layout=True)

vmin_m = min(spectro_log.min(), corrected_monotonic_log.min(), corrected_sigmoid_log.min())
vmax_m = max(spectro_log.max(), corrected_monotonic_log.max(), corrected_sigmoid_log.max())

# 1. Original
pcm_orig4 = axes4[0].pcolormesh(t_spectro, f_M, spectro_log, shading='nearest', cmap='jet', vmin=vmin_m, vmax=vmax_m)
axes4[0].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes4[0].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes4[0].set_title('Original Spectrogram')
axes4[0].set_ylabel('Frequency (Hz)')
fig4.colorbar(pcm_orig4, ax=axes4[0], label='Log Intensity')

# 2. Linear Monotonic Floor
pcm_corr4 = axes4[1].pcolormesh(t_spectro, f_M, corrected_monotonic_log, shading='nearest', cmap='jet')#, vmin=vmin_m, vmax=vmax_m)
axes4[1].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes4[1].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes4[1].set_title('Corrected Spectrogram (Linear Monotonic Floor Projection)')
axes4[1].set_ylabel('Frequency (Hz)')
fig4.colorbar(pcm_corr4, ax=axes4[1], label='Log Intensity')

# 3. Smooth Sigmoid Projection
pcm_sig4 = axes4[2].pcolormesh(t_spectro, f_M, corrected_sigmoid_log, shading='nearest', cmap='jet')#, vmin=vmin_m, vmax=vmax_m)
axes4[2].contour(t_spectro, f_M, seg_bool, colors='white', linewidths=1.2, linestyles='--')
axes4[2].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=1.2, linestyles='--')
axes4[2].set_title('Corrected Spectrogram (True Sigmoid ECDF Projection)')
axes4[2].set_xlabel('Time (s)')
axes4[2].set_ylabel('Frequency (Hz)')
fig4.colorbar(pcm_sig4, ax=axes4[2], label='Log Intensity')

# Plot ECDFs
axes4[3].plot(local_outside_vals, y_local_outside, label='Outside (within 2 bins)', color='black')
axes4[3].plot(inside_vals, y_inside, label='Original Inside Mask', color='red', linestyle='--')
axes4[3].plot(monotonic_inside_vals, y_inside, label='Corrected Inside (Linear Monotonic Floor)', color='purple')
axes4[3].plot(sigmoid_inside_vals, y_inside, label='Corrected Inside (Sigmoid ECDF)', color='deepskyblue', linewidth=2.0)
axes4[3].axvline(max_outside_val, color='gray', linestyle=':', label='Max Outside Threshold')
axes3[3].axvline(log2_T_low_max, color='orange', linestyle='--', label='T_low Threshold (Max)')
axes4[3].set_xlabel('Log Spectrogram Intensity')
axes4[3].set_ylabel('ECDF')
axes4[3].set_title('ECDF Matching & Reshaping Comparison')
axes4[3].legend()
axes4[3].grid(True)

plt.show()