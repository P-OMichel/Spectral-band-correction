'''
File to test the funcitons of segmentation and correction of ketamine spectral signature.
'''

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

# --- Display
fig, axes = plt.subplots(4,  constrained_layout = True)
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
seg_mask, psd, baseline = segment_blobs(M, f_M, f_int)

fig, axes = plt.subplots(4,  constrained_layout = True)
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

# ================== Correct spectrogram ===============

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


spectro_log = np.log2(M + 1e-11)
seg_bool = seg_mask.astype(bool)

# 1. Expand mask by 2 iterations to catch pixels within 2 bins
expanded_mask = binary_dilation(seg_bool, iterations=2)

# 2. Local border pixels = expanded mask MINUS original mask (only uncontaminated neighbors)
border_mask = expanded_mask & ~seg_bool

# 3. Extract and sort log intensity values
inside_vals = np.sort(spectro_log[seg_bool])
local_outside_vals = np.sort(spectro_log[border_mask])

# 4. Compute ECDF y-values
y_inside = np.linspace(0, 1, len(inside_vals))
y_local_outside = np.linspace(0, 1, len(local_outside_vals))

# 5. Plot ECDFs
plt.figure(figsize=(7, 4.5))
plt.plot(inside_vals, y_inside, label='Inside Mask', color='red')
plt.plot(local_outside_vals, y_local_outside, label='Outside (within 2 bins)', color='black')

plt.xlabel('Log Spectrogram Intensity')
plt.ylabel('ECDF')
plt.title('ECDF: Inside Mask vs. Local Outside Neighborhood')
plt.legend()
plt.grid(True)
plt.show()

from scipy.ndimage import label, binary_dilation, distance_transform_edt

def correct_blobs_local_knee(M, seg_mask, iterations=3, gamma=0.6, alpha=0.8):
    """
    Reduces intensity of segmented blobs in a log-spectrogram.
    - Preserves internal pixel ordering (monotonic transformation)
    - Prevents pixels from dropping below local boundary floor
    - Blends smoothly into the background
    """
    spectro_log = np.log2(M + 1e-11)
    corrected_log = spectro_log.copy()
    
    # Label distinct blobs
    labeled_mask, num_features = label(seg_mask)
    
    for blob_id in range(1, num_features + 1):
        blob_idx = (labeled_mask == blob_id)
        
        # 1. Define local border
        expanded = binary_dilation(blob_idx, iterations=iterations)
        border_idx = expanded & ~blob_idx
        
        if not np.any(border_idx):
            continue
            
        # 2. Local Floor (max intensity of surrounding border)
        I_floor = np.max(spectro_log[border_idx])
        
        # 3. Apply Soft-Knee Monotonic Compression
        I_blob = spectro_log[blob_idx]
        delta = np.maximum(0, I_blob - I_floor)
        
        # Monotonic mapping: strictly preserves pixel rank ordering
        I_compressed = I_floor + alpha * (delta ** gamma)
        
        # 4. Spatial Distance Weighting (smooth transition at boundaries)
        dist_inside = distance_transform_edt(blob_idx)
        max_dist = np.max(dist_inside)
        if max_dist > 0:
            weights = dist_inside[blob_idx] / max_dist
        else:
            weights = 1.0
            
        # Blend original and compressed based on distance to edge
        corrected_log[blob_idx] = (1 - weights) * I_blob + weights * I_compressed

    return corrected_log

# Execute Correction
corrected_log = correct_blobs_local_knee(M, seg_mask, iterations=3, gamma=0.6, alpha=0.8)

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
axes[0].pcolormesh(t_spectro, f_M, np.log2(M + 1e-11), shading='nearest', cmap='jet', vmin = -10, vmax = 4)
axes[0].set_title('Original Spectrogram')

axes[1].pcolormesh(t_spectro, f_M, seg_mask, shading='nearest', cmap='gray')
axes[1].set_title('Segmentation Mask')

axes[2].pcolormesh(t_spectro, f_M, corrected_log, shading='nearest', cmap='jet', vmin = -10, vmax = 4)
axes[2].set_title('Corrected Spectrogram (Rank Preserved & Peak Attenuated)')

plt.tight_layout()
plt.show()