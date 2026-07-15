import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from scipy.ndimage import label
from skimage.restoration import inpaint
from sklearn.decomposition import NMF

# =============================================================================
# 1. SETUP: GENERATE DUMMY EEG SIGNAL & DETECT BLOB
# =============================================================================
np.random.seed(42)
fs = 256  
t = np.arange(0, 10, 1/fs)  

# Background EEG
background_signal = np.random.normal(0, 1.0, len(t))

# Spindle/Burst Blob (12 Hz from 4 to 6 seconds)
spindle_time = (t >= 4) & (t <= 6)
envelope = np.exp(-((t[spindle_time] - 5) ** 2) / (2 * 0.4**2))
burst = 15.0 * envelope * np.sin(2 * np.pi * 12 * t[spindle_time])
eeg_signal = background_signal.copy()
eeg_signal[spindle_time] += burst

# Compute STFT
nperseg = 64
frequencies, times, Zxx = stft(eeg_signal, fs=fs, nperseg=nperseg, noverlap=nperseg-8)
magnitude = np.abs(Zxx)
phase = np.angle(Zxx)

# Isolate the highest intensity blob mask
threshold = 3.0 * np.median(magnitude)
high_intensity_mask = magnitude > threshold
labeled_mask, num_features = label(high_intensity_mask)

max_intensity_val = 0
highest_blob_label = 1
for i in range(1, num_features + 1):
    blob_peak = np.max(magnitude[labeled_mask == i])
    if blob_peak > max_intensity_val:
        max_intensity_val = blob_peak
        highest_blob_label = i

blob_mask = (labeled_mask == highest_blob_label)

# =============================================================================
# METHOD 1: BIHARMONIC INPAINTING (Structural Texture & Intensity)
# =============================================================================
mag_method1 = inpaint.inpaint_biharmonic(magnitude, blob_mask)

# =============================================================================
# METHOD 2: NON-NEGATIVE MATRIX FACTORIZATION (NMF) INPAINTING
# =============================================================================
# Find time columns that do NOT contain any part of the blob
clean_cols = ~np.any(blob_mask, axis=0)

# Transpose matrices so time steps = samples, and frequency bins = features
magnitude_T = magnitude.T
clean_magnitude_T = magnitude_T[clean_cols, :]

# Train NMF on the clean background texture to learn frequency components
nmf = NMF(n_components=3, init='random', random_state=42, max_iter=1000)
# H matrix here represents the learned spectral bases (n_components, n_frequencies)
W_train = nmf.fit_transform(clean_magnitude_T) 

# Find the activations (W_full) for ALL time steps using the fixed background components
W_full = nmf.transform(magnitude_T)

# Reconstruct the full magnitude matrix from the background components
mag_nmf_full_T = np.dot(W_full, nmf.components_)
mag_nmf_full = mag_nmf_full_T.T  # Transpose back to original (frequencies, time)

# Replace ONLY the blob region with the NMF background estimate
mag_method2 = magnitude.copy()
mag_method2[blob_mask] = mag_nmf_full[blob_mask]

# =============================================================================
# METHOD 3: ITERATIVE LAPLACIAN OPTIMIZATION (Total Variation / Smoothing Proxy)
# =============================================================================
mag_method3 = magnitude.copy()
# Initialize the blob region with the average of its boundary values
mag_method3[blob_mask] = np.median(magnitude[~blob_mask])

# Iterative relaxation (solving Laplace equation to smooth intensity gradients)
for _ in range(200):
    smoothed = (
        np.roll(mag_method3, 1, axis=0) + np.roll(mag_method3, -1, axis=0) +
        np.roll(mag_method3, 1, axis=1) + np.roll(mag_method3, -1, axis=1)
    ) / 4.0
    # Keep background fixed, update only the blob mask
    mag_method3[blob_mask] = smoothed[blob_mask]

# =============================================================================
# RECONSTRUCT TIME-DOMAIN SIGNALS
# =============================================================================
signals_clean = []
for mag_matrix in [mag_method1, mag_method2, mag_method3]:
    Zxx_corr = mag_matrix * np.exp(1j * phase)
    _, x_rec = istft(Zxx_corr, fs=fs, nperseg=nperseg, noverlap=nperseg-8)
    signals_clean.append(x_rec[:len(t)])

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex='col', sharey='row')

# Plot titles and configurations
titles = [
    ("Original (With Blob)", magnitude, eeg_signal, 'crimson'),
    ("Method 1: Biharmonic Inpainting", mag_method1, signals_clean[0], 'teal'),
    ("Method 2: NMF Factorization", mag_method2, signals_clean[1], 'darkorange'),
    ("Method 3: Laplacian Optimization", mag_method3, signals_clean[2], 'purple')
]

for idx, (title, mag_data, sig_data, sig_color) in enumerate(titles):
    # Left Column: Spectrograms
    mesh = axes[idx, 0].pcolormesh(times, frequencies, mag_data, shading='gouraud', cmap='jet', vmax=5)
    axes[idx, 0].set_title(title)
    axes[idx, 0].set_ylabel("Freq (Hz)")
    axes[idx, 0].set_ylim(0, 30)
    fig.colorbar(mesh, ax=axes[idx, 0], label="Mag")
    
    # Right Column: Time Series
    axes[idx, 1].plot(t, sig_data, color=sig_color, alpha=0.7)
    axes[idx, 1].set_title(f"{title} - Time Domain")
    axes[idx, 1].grid(True)
    if idx == 3:
        axes[idx, 0].set_xlabel("Time (seconds)")
        axes[idx, 1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.show()