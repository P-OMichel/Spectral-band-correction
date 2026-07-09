import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from Functions.time_frequency import spectrogram

#--- load data 
data = sc.io.loadmat('recording/rec_20240118_104943.mat')
y = data['record'][:, 0]
fs = 128
t = np.arange(len(y)) / fs
f_spectro, t_spectro, spectro = spectrogram(y, fs, nfft_factor=2)
spectro = np.log2(spectro + 1e-11)
f_mask = f_spectro>=20
f_spectro, spectro = f_spectro[f_mask], spectro[f_mask, :]
mask = spectro >= np.quantile(spectro, 0.5)

# --- 1. Find local peaks to use as markers
# min_distance ensures we don't get a marker for every single pixel noise
coordinates = peak_local_max(spectro, min_distance=5, labels=mask)

# Create a boolean mask of peaks, then label them uniquely
peaks_mask = np.zeros_like(spectro, dtype=bool)
peaks_mask[tuple(coordinates.T)] = True
markers, _ = ndi.label(peaks_mask)

# --- 2. Run Watershed
# We invert 'spectro' (-spectro) so high energy regions act as basins/valleys
labels = watershed(-spectro, markers, mask=mask)

# --- 3. Visualization
fig, axes = plt.subplots(3, sharex = True, constrained_layout = True)
axes[0].pcolormesh(t_spectro, f_spectro, spectro, shading='auto', cmap='jet', vmin = -4, vmax = 8)
axes[0].set_title('Original Spectrogram')
axes[0].set_ylabel('Frequency (Hz)')
axes[1].pcolormesh(t_spectro, f_spectro, mask, shading='auto', cmap='gray')
axes[1].scatter(t_spectro[coordinates[:, 1]], f_spectro[coordinates[:, 0]], color='red', s=10, label='Markers')
axes[1].set_title('Threshold Mask & Detected Markers')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].legend()
axes[2].pcolormesh(t_spectro, f_spectro, spectro, shading='auto', cmap='jet', vmin = -4, vmax = 8)
axes[2].pcolormesh(t_spectro, f_spectro, np.ma.masked_where(labels == 0, labels), 
               shading='auto', cmap='jet', alpha=0.5)
axes[2].set_title('Watershed Segmentation Output')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Frequency (Hz)')

plt.show()