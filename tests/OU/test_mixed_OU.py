'''
File to generate a simulated EEG signal using sum of OU processes
'''

import numpy as np
import matplotlib.pyplot as plt
from Functions.generate_OU import get_mixed_OU_signals
from Functions.time_frequency import spectrogram

# --- Parameters
T = 20 # desired signal duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 1, 1]
omega_list = [2*np.pi*1, 2*np.pi*10, 2*np.pi*30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 1]


# --- Generate EEG data
t, y = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

# --- compute spectrogram
f_spectro, t_spectro, spectro = spectrogram(y, fs)

# --- Display
fig, axes = plt.subplots(3,  constrained_layout = True)
axes[0].plot(t, y)
axes[1].pcolormesh(t_spectro, f_spectro, np.log2(spectro + 1e-11), shading = 'nearest', cmap = 'jet')
axes[2].plot(f_spectro, np.log2(np.sum(spectro, axis = 1)))
axes[1].sharex(axes[0])

plt.show()