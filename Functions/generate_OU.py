import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def get_OU_signal(T, dt, lbda, omega, sigma):
    """
    Generates a 2D Ornstein-Uhlenbeck signal using the unconditionally stable 
    Implicit Backward Euler scheme.
    """
    N = int(T / dt)          # Number of time steps
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    y = np.zeros(N)

    x[0] = 0.0
    y[0] = 0.0    

    A = np.array([[-lbda,  omega],
                  [-omega, -lbda]])
    I = np.eye(2)
    implicit_operator = np.linalg.inv(I - A * dt)

    for i in range(1, N):
        dW_x = np.random.normal(0, np.sqrt(dt))
        dW_y = np.random.normal(0, np.sqrt(dt))
        
        current_state_with_noise = np.array([
            x[i-1] + np.sqrt(2) * sigma * dW_x,
            y[i-1] + np.sqrt(2) * sigma * dW_y
        ])
        
        next_state = implicit_operator @ current_state_with_noise
        
        x[i] = next_state[0]
        y[i] = next_state[1]

    return t, x


def get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list):
    """
    Generates mixed OU signals and extracts ground-truth peak segmentation limits
    empirically from the computed PSD to account for Welch window smearing.
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    fs = 1.0 / dt

    mixed_OU = np.zeros(N)

    # 1. Generate and sum the OU signals
    for i in range(len(lbda_list)):
        x = get_OU_signal(T, dt, lbda_list[i], omega_list[i], sigma_list[i])[-1]
        mixed_OU += x * factor_list[i]

    # 2. Compute the empirical PSD of the mixture
    f, psd = signal.welch(mixed_OU, fs=fs, nperseg=int(fs * 2))

    peak_limits = []

    # 3. Empirically find the boundaries for each peak on the calculated PSD
    for i in range(len(omega_list)):
        f_target = omega_list[i] / (2 * np.pi)
        
        # Find the index in the frequency array 'f' closest to our theoretical target
        target_idx = np.argmin(np.abs(f - f_target))
        
        # Find the local peak maximum around the target frequency (within a +/- 2 Hz buffer)
        search_radius = int(2.0 / (f[1] - f[0])) 
        start_search = max(0, target_idx - search_radius)
        end_search = min(len(f), target_idx + search_radius)
        
        local_peak_idx = start_search + np.argmax(psd[start_search:end_search])
        peak_max_power = psd[local_peak_idx]
        half_max_power = peak_max_power / 2.0
        
        # Scan left from the peak to find where it drops below half-maximum
        left_idx = local_peak_idx
        while left_idx > 0 and psd[left_idx] > half_max_power:
            left_idx -= 1
            
        # Scan right from the peak to find where it drops below half-maximum
        right_idx = local_peak_idx
        while right_idx < len(f) - 1 and psd[right_idx] > half_max_power:
            right_idx += 1
            
        peak_limits.append({
            'peak_index': i,
            'center_freq': f[local_peak_idx],
            'left_bound': f[left_idx],
            'right_bound': f[right_idx],
            'empirical_fwhm': f[right_idx] - f[left_idx]
        })

    return t, mixed_OU, f, psd, peak_limits


# --- Example Execution & Visualization ---
if __name__ == "__main__":
    T = 60.0
    dt = 1.0 / 250.0  # 250 Hz sampling rate

    # Injecting two distinct peaks: a Theta bump (6Hz) and an Alpha bump (11Hz)
    lbda_list  = [2.5, 1.2]       
    omega_list = [2 * np.pi * 1, 2 * np.pi * 11] 
    sigma_list = [1.0, 1.2]
    factor_list = [1.0, 1.0]

    # Run Generation
    t, mixed_OU, f, psd, peak_limits = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

    # Automatically construct the supervised 1D label mask for training
    ground_truth_mask = np.zeros_like(f)
    for peak in peak_limits:
        in_peak_range = (f >= peak['left_bound']) & (f <= peak['right_bound'])
        ground_truth_mask[in_peak_range] = 1.0

    # Plotting to verify the visual fit
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.semilogy(f, psd, color='black', label='Empirical Mixed PSD')
    colors = ['orange', 'purple']
    for i, p in enumerate(peak_limits):
        ax1.axvspan(p['left_bound'], p['right_bound'], color=colors[i], alpha=0.2, 
                    label=f"Peak {i} Empirical Bound ({p['left_bound']:.2f}-{p['right_bound']:.2f} Hz)")
        ax1.axvline(p['center_freq'], color=colors[i], linestyle='--')
        
    ax1.set_xlim(0, 25)
    ax1.set_ylabel('Power')
    ax1.set_title('PSD with Smear-Corrected Empirical FWHM Bounds')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.fill_between(f, ground_truth_mask, color='green', alpha=0.4, step='mid')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Label Mask (Y_train)')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()