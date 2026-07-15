import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def generate_multi_ou_psd(fs, duration, peaks_config, continuous_noise_scale=0.5):
    """
    Generates a mixed signal from multiple independent 2D OU processes and returns
    the empirical PSD alongside structural ground-truth labels for all peaks.
    
    Parameters:
    - fs (int): Sampling frequency
    - duration (float): Signal length in seconds
    - peaks_config (list of dicts): Explicit parameters for each peak, e.g.:
         [{'theta': 4.0, 'f_center': 10.0, 'sigma': 1.0}, ...]
    - continuous_noise_scale (float): Scale of standard white/pink noise added to the mix.
    
    Returns:
    - f (ndarray): Frequency vector
    - psd (ndarray): Combined empirical Power Spectral Density
    - peak_labels (list of dicts): The exact FWHM boundaries mapped for each individual peak.
    """
    n_samples = int(fs * duration)
    dt = 1.0 / fs
    
    # Initialize the master mixed signal array
    mixed_signal = np.zeros(n_samples)
    
    peak_labels = []
    
    # 1. Dynamically generate and stack each independent 2D OU process
    for config in peaks_config:
        theta = config['theta']
        f_center = config['f_center']
        sigma = config['sigma']
        
        omega = 2 * np.pi * f_center
        X = np.zeros(n_samples)
        Y = np.zeros(n_samples)
        
        dW1 = np.random.normal(0, np.sqrt(dt), n_samples)
        dW2 = np.random.normal(0, np.sqrt(dt), n_samples)
        
        for t in range(1, n_samples):
            X[t] = X[t-1] + (-theta * X[t-1] - omega * Y[t-1]) * dt + sigma * dW1[t]
            Y[t] = Y[t-1] + (-theta * Y[t-1] + omega * X[t-1]) * dt + sigma * dW2[t]
            
        # Add this specific oscillation to our global recording trace
        mixed_signal += X
        
        # Calculate analytical limits for this specific peak
        fwhm = theta / np.pi
        peak_labels.append({
            'center_freq': f_center,
            'fwhm_width': fwhm,
            'left_bound': f_center - (fwhm / 2),
            'right_bound': f_center + (fwhm / 2)
        })
        
    # 2. Add an underlying background noise component (Simulating general EEG ambient state)
    background_noise = np.random.normal(0, continuous_noise_scale, n_samples)
    mixed_signal += background_noise
    
    # 3. Compute empirical PSD of the combined multi-peak signal
    f, psd = signal.welch(mixed_signal, fs=fs, nperseg=int(fs * 2))
    
    return f, psd, peak_labels

# --- Configuration for Multiple Peak Generation ---
fs = 250
duration = 60.0

# Define a dictionary for each bump you want to inject into the recording
# Let's create a Theta bump (around 5 Hz) and a sharp Alpha bump (around 11 Hz)
my_peaks = [
    {'f_center': 5.2,  'theta': 3.0, 'sigma': 1.0},  # Broad Theta peak
    {'f_center': 11.5, 'theta': 1.5, 'sigma': 1.2}   # Narrow, tall Alpha peak
]

# Generate the data
f, psd, peak_labels = generate_multi_ou_psd(fs, duration, my_peaks, continuous_noise_scale=0.3)

# --- Create Ground Truth Mask for Semantic Segmentation Training ---
# Start with an array of zeros matching the length of our frequency axis
ground_truth_mask = np.zeros_like(f)

for label in peak_labels:
    # Set indices to 1 if they fall inside ANY of the valid analytical FWHM bands
    in_range = (f >= label['left_bound']) & (f <= label['right_bound'])
    ground_truth_mask[in_range] = 1.0


# --- 4. Plotting Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})

# Top Plot: The combined multi-peak PSD
ax1.plot(f, psd, color='black', alpha=0.8, label='Mixed Empirical PSD')

colors = ['orange', 'purple', 'cyan']
for i, label in enumerate(peak_labels):
    c = colors[i % len(colors)]
    # Vertical line at peak center
    ax1.axvline(label['center_freq'], color=c, linestyle='--', alpha=0.7)
    # Shade the active segment area
    mask_indices = (f >= label['left_bound']) & (f <= label['right_bound'])
    ax1.fill_between(f, psd, where=mask_indices, color=c, alpha=0.25, 
                     label=f"Peak {i+1} ({label['center_freq']} Hz)")

ax1.set_title("Multi-Peak Synthetic EEG PSD Generation")
ax1.set_ylabel("Power Spectral Density")
ax1.set_xlim(0, 30)
ax1.legend()
ax1.grid(True, alpha=0.2)

# Bottom Plot: The resulting supervised 1D segmentation binary target mask
ax2.fill_between(f, ground_truth_mask, color='green', alpha=0.5, step='mid')
ax2.plot(f, ground_truth_mask, color='green', drawstyle='steps-mid')
ax2.set_title("Binary Target Mask Vector (Y_train Label for Deep Learning)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Class Label")
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()