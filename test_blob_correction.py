import numpy as np
import matplotlib.pyplot as plt
from Functions.generate_OU import get_OU_signal
from Functions.time_frequency import spectrogram

dt = 0.005
fs = 1 / dt
y = np.load('signal.npy')
t = np.arange(len(y)) / fs

f_spectro, t_spectro, spectro = spectrogram(y, fs)

fig, axes = plt.subplots(2, sharex = True, constrained_layout = True)
axes[0].plot(t, y)
axes[1].pcolormesh(t_spectro, f_spectro, np.log2(spectro + 1e-11), shading = 'nearest', cmap = 'jet')

plt.show()
