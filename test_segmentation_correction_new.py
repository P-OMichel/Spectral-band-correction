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



# ==============================================================================
# --- Correction Methods & Comparison
# ==============================================================================
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from skimage.restoration import inpaint

# Target data working in log2 domain
I_orig = spectro_log.copy()
ny, nx = I_orig.shape

# ------------------------------------------------------------------------------
# Method 0: Standard Quantile/ECDF Matching (For reference / baseline comparison)
# ------------------------------------------------------------------------------
inside_sorted = np.sort(inside_raw)
N_in = len(inside_raw)
N_out = len(local_outside_vals)

# Quantiles [0, 1]
q_in = np.linspace(0, 1, N_in)
q_out = np.linspace(0, 1, N_out)

# Map inside values through local outside distribution
ecdf_map = interp1d(
    inside_sorted,
    np.interp(q_in, q_out, local_outside_vals),
    bounds_error=False,
    fill_value="extrapolate",
)
I_ecdf_direct = I_orig.copy()
I_ecdf_direct[seg_bool] = ecdf_map(inside_raw)


# ------------------------------------------------------------------------------
# Method 1: Local Alpha-Blended ECDF Matching
# ------------------------------------------------------------------------------
# Compute distance transform inward from the boundary
dist_inside = ndi.distance_transform_edt(seg_bool)
d_max = np.max(dist_inside) if np.max(dist_inside) > 0 else 1.0
alpha_weight = np.clip(dist_inside / (d_max * 0.75 + 1e-6), 0.0, 1.0)

I_alpha_ecdf = I_orig.copy()
I_alpha_ecdf[seg_bool] = (1.0 - alpha_weight[seg_bool]) * I_orig[
    seg_bool
] + alpha_weight[seg_bool] * I_ecdf_direct[seg_bool]


# ------------------------------------------------------------------------------
# Method 2: Monotonic Dynamic Range Compression (Soft Knee)
# ------------------------------------------------------------------------------
# Threshold set at 75th percentile of local outside reference ring
T_knee = np.percentile(local_outside_vals, 75)
s_scale = np.std(local_outside_vals) + 1e-6

I_soft_knee = I_orig.copy()
inside_mask_knee = seg_bool & (I_orig > T_knee)
I_soft_knee[inside_mask_knee] = T_knee + s_scale * np.tanh(
    (I_orig[inside_mask_knee] - T_knee) / s_scale
)


# ------------------------------------------------------------------------------
# Method 3: Poisson / Gradient-Domain Blending
# ------------------------------------------------------------------------------
# Laplace operator with Dirichlet boundary conditions on ~seg_bool
# Attenuate interior gradients by factor alpha_grad
alpha_grad = 0.2
I_poisson = I_orig.copy()

if np.any(seg_bool):
  # Map 2D indices to linear unknown IDs
  pixel_indices = -np.ones((ny, nx), dtype=int)
  inside_coords = np.argwhere(seg_bool)
  N_vars = len(inside_coords)

  for idx, (r, c) in enumerate(inside_coords):
    pixel_indices[r, c] = idx

  # Discrete Laplacian of original field
  # Del^2(I) approx I(r+1,c) + I(r-1,c) + I(r,c+1) + I(r,c-1) - 4*I(r,c)
  laplacian_orig = (
      np.roll(I_orig, 1, axis=0)
      + np.roll(I_orig, -1, axis=0)
      + np.roll(I_orig, 1, axis=1)
      + np.roll(I_orig, -1, axis=1)
      - 4.0 * I_orig
  )

  A = lil_matrix((N_vars, N_vars), dtype=float)
  b_vec = np.zeros(N_vars, dtype=float)
  neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  for i, (r, c) in enumerate(inside_coords):
    A[i, i] = -4.0
    b_vec[i] = alpha_grad * laplacian_orig[r, c]

    for dr, dc in neighbors:
      nr, nc = r + dr, c + dc
      if 0 <= nr < ny and 0 <= nc < nx:
        if seg_bool[nr, nc]:
          A[i, pixel_indices[nr, nc]] += 1.0
        else:
          # Dirichlet boundary condition from surrounding background
          b_vec[i] -= I_orig[nr, nc]
      else:
        # Array edge fallback
        b_vec[i] -= I_orig[r, c]

  # Solve sparse linear system
  sol = spsolve(csc_matrix(A), b_vec)
  I_poisson[seg_bool] = sol


# ------------------------------------------------------------------------------
# Method 4: Biharmonic Inpainting
# ------------------------------------------------------------------------------
I_inpaint = inpaint.inpaint_biharmonic(I_orig, seg_bool)


# ==============================================================================
# --- Visual Comparison of Spectrograms
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
vmin = np.percentile(I_orig, 5)
vmax = np.percentile(I_orig, 99.5)

panels = [
    (axes[0, 0], I_orig, 'Original Spectrogram (> 20 Hz)'),
    (axes[0, 1], I_ecdf_direct, 'Direct ECDF (Discontinuity visible at edges)'),
    (axes[1, 0], I_alpha_ecdf, 'Alpha-blended Local ECDF'),
    (axes[1, 1], I_soft_knee, 'Monotonic Soft-Knee Compression'),
    (axes[2, 0], I_poisson, 'Poisson Blending (Attenuated Gradients)'),
    (axes[2, 1], I_inpaint, 'Biharmonic Inpainting (Total Removal)'),
]

for ax, data, title in panels:
  pcm = ax.pcolormesh(
      t_spectro, f_M, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax
  )
  ax.contour(t_spectro, f_M, seg_bool, colors='black', linewidths=1)
  ax.set_title(title, fontsize=10)
  ax.set_ylabel('Freq (Hz)')
  ax.sharex(axes[0, 0])

fig.colorbar(pcm, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.04)
plt.show()


# ==============================================================================
# --- ECDF Comparison (Inside Blob vs Reference vs Corrected Methods)
# ==============================================================================
def get_ecdf(data):
  x_sorted = np.sort(data.ravel())
  y_cdf = np.linspace(0, 1, len(x_sorted))
  return x_sorted, y_cdf


fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

# Compute ECDFs inside the segmented blob area
x_bg, y_bg = get_ecdf(local_outside_vals)
x_in, y_in = get_ecdf(I_orig[seg_bool])
x_ecdf_d, y_ecdf_d = get_ecdf(I_ecdf_direct[seg_bool])
x_alpha, y_alpha = get_ecdf(I_alpha_ecdf[seg_bool])
x_knee, y_knee = get_ecdf(I_soft_knee[seg_bool])
x_poiss, y_poiss = get_ecdf(I_poisson[seg_bool])
x_inp, y_inp = get_ecdf(I_inpaint[seg_bool])

ax.plot(
    x_bg,
    y_bg,
    'k--',
    linewidth=2,
    label='Local Outside Ring (Reference Baseline)',
)
ax.plot(x_in, y_in, 'r-', linewidth=1.5, label='Original Inside Blobs')
ax.plot(x_ecdf_d, y_ecdf_d, ':', linewidth=1.5, label='Direct ECDF')
ax.plot(x_alpha, y_alpha, '-.', linewidth=1.5, label='Alpha-blended ECDF')
ax.plot(x_knee, y_knee, '-', linewidth=1.5, label='Soft-Knee Compression')
ax.plot(x_poiss, y_poiss, '-', linewidth=1.5, label='Poisson Blending')
ax.plot(x_inp, y_inp, '-', linewidth=1.5, label='Biharmonic Inpainting')

ax.set_xlabel('Log2 Power Intensity')
ax.set_ylabel('Cumulative Probability')
ax.set_title('ECDF Comparison Inside Segmented Blobs vs. Reference')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower right', frameon=True)

plt.show()


# ==============================================================================
# --- Correction Methods Applied to EXTENDED MASK
# ==============================================================================
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix, lil_matrix
from skimage.restoration import inpaint

# Target mask for modification: expanded_mask
target_mask = expanded_mask.copy()

# New reference boundary: pixels adjacent to the expanded mask
outer_ring = binary_dilation(target_mask, iterations=2) & ~target_mask
ext_outside_vals = np.sort(spectro_log[outer_ring])
ext_inside_raw = spectro_log[target_mask]

# ------------------------------------------------------------------------------
# Method 0: Direct ECDF on Extended Mask
# ------------------------------------------------------------------------------
N_in_ext = len(ext_inside_raw)
N_out_ext = len(ext_outside_vals)
q_in_ext = np.linspace(0, 1, N_in_ext)
q_out_ext = np.linspace(0, 1, N_out_ext)

ecdf_map_ext = interp1d(
    np.sort(ext_inside_raw),
    np.interp(q_in_ext, q_out_ext, ext_outside_vals),
    bounds_error=False,
    fill_value="extrapolate",
)

I_ecdf_ext = I_orig.copy()
I_ecdf_ext[target_mask] = ecdf_map_ext(ext_inside_raw)

# ------------------------------------------------------------------------------
# Method 1: Alpha-blended Local ECDF on Extended Mask
# ------------------------------------------------------------------------------
dist_ext = ndi.distance_transform_edt(target_mask)
d_max_ext = np.max(dist_ext) if np.max(dist_ext) > 0 else 1.0
# Smooth transition: 0 at boundary to 1 deep inside
alpha_weight_ext = np.clip(dist_ext / (d_max_ext * 0.8 + 1e-6), 0.0, 1.0)

I_alpha_ext = I_orig.copy()
I_alpha_ext[target_mask] = (1.0 - alpha_weight_ext[target_mask]) * I_orig[
    target_mask
] + alpha_weight_ext[target_mask] * I_ecdf_ext[target_mask]

# ------------------------------------------------------------------------------
# Method 2: Soft-Knee Compression on Extended Mask (Preserves Hill Curvature)
# ------------------------------------------------------------------------------
# Compression threshold using median and scale of the outer ring
T_knee_ext = np.percentile(ext_outside_vals, 80)
s_scale_ext = np.std(ext_outside_vals) * 1.2 + 1e-6

I_knee_ext = I_orig.copy()
inside_comp_mask = target_mask & (I_orig > T_knee_ext)
I_knee_ext[inside_comp_mask] = T_knee_ext + s_scale_ext * np.tanh(
    (I_orig[inside_comp_mask] - T_knee_ext) / s_scale_ext
)

# ------------------------------------------------------------------------------
# Method 3: Poisson Blending on Extended Mask (alpha_grad controls hill height)
# ------------------------------------------------------------------------------
# alpha_grad = 0.5 retains 50% of original Laplacian gradients (maintains hill profile)
alpha_grad_ext = 0.5
I_poisson_ext = I_orig.copy()

if np.any(target_mask):
  pixel_indices_ext = -np.ones((ny, nx), dtype=int)
  inside_coords_ext = np.argwhere(target_mask)
  N_vars_ext = len(inside_coords_ext)

  for idx, (r, c) in enumerate(inside_coords_ext):
    pixel_indices_ext[r, c] = idx

  laplacian_orig = (
      np.roll(I_orig, 1, axis=0)
      + np.roll(I_orig, -1, axis=0)
      + np.roll(I_orig, 1, axis=1)
      + np.roll(I_orig, -1, axis=1)
      - 4.0 * I_orig
  )

  A_ext = lil_matrix((N_vars_ext, N_vars_ext), dtype=float)
  b_vec_ext = np.zeros(N_vars_ext, dtype=float)
  neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

  for i, (r, c) in enumerate(inside_coords_ext):
    A_ext[i, i] = -4.0
    b_vec_ext[i] = alpha_grad_ext * laplacian_orig[r, c]

    for dr, dc in neighbors:
      nr, nc = r + dr, c + dc
      if 0 <= nr < ny and 0 <= nc < nx:
        if target_mask[nr, nc]:
          A_ext[i, pixel_indices_ext[nr, nc]] += 1.0
        else:
          b_vec_ext[i] -= I_orig[nr, nc]
      else:
        b_vec_ext[i] -= I_orig[r, c]

  sol_ext = spsolve(csc_matrix(A_ext), b_vec_ext)
  I_poisson_ext[target_mask] = sol_ext

# ------------------------------------------------------------------------------
# Method 4: Biharmonic Inpainting on Extended Mask
# ------------------------------------------------------------------------------
I_inpaint_ext = inpaint.inpaint_biharmonic(I_orig, target_mask)


# ==============================================================================
# --- Visual Comparison with Extended Mask
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)

panels_ext = [
    (axes[0, 0], I_orig, 'Original Spectrogram (> 20 Hz)'),
    (axes[0, 1], I_ecdf_ext, 'Direct ECDF (Extended Mask)'),
    (axes[1, 0], I_alpha_ext, 'Alpha-blended Local ECDF (Extended Mask)'),
    (axes[1, 1], I_knee_ext, 'Soft-Knee Compression (Extended Mask)'),
    (
        axes[2, 0],
        I_poisson_ext,
        'Poisson Blending (Extended Mask, alpha=0.5)',
    ),
    (axes[2, 1], I_inpaint_ext, 'Biharmonic Inpainting (Extended Mask)'),
]

for ax, data, title in panels_ext:
  pcm = ax.pcolormesh(
      t_spectro, f_M, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax
  )
  # Show original blob boundary in black, extended boundary in dashed white
  ax.contour(t_spectro, f_M, seg_bool, colors='black', linewidths=0.7)
  ax.contour(
      t_spectro,
      f_M,
      target_mask,
      colors='white',
      linewidths=0.6,
      linestyles='--',
  )
  ax.set_title(title, fontsize=10)
  ax.set_ylabel('Freq (Hz)')
  ax.sharex(axes[0, 0])

fig.colorbar(pcm, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.04)
plt.show()


# ==============================================================================
# --- ECDF Comparison on Original Blob Region (seg_bool)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

x_bg_ext, y_bg_ext = get_ecdf(ext_outside_vals)
x_in, y_in = get_ecdf(I_orig[seg_bool])
x_ecdf_e, y_ecdf_e = get_ecdf(I_ecdf_ext[seg_bool])
x_alpha_e, y_alpha_e = get_ecdf(I_alpha_ext[seg_bool])
x_knee_e, y_knee_e = get_ecdf(I_knee_ext[seg_bool])
x_poiss_e, y_poiss_e = get_ecdf(I_poisson_ext[seg_bool])
x_inp_e, y_inp_e = get_ecdf(I_inpaint_ext[seg_bool])

ax.plot(
    x_bg_ext,
    y_bg_ext,
    'k--',
    linewidth=2,
    label='Outer Reference Ring (Baseline)',
)
ax.plot(x_in, y_in, 'r-', linewidth=1.5, label='Original Inside Blobs')
ax.plot(x_ecdf_e, y_ecdf_e, ':', linewidth=1.5, label='Direct ECDF (Extended)')
ax.plot(
    x_alpha_e,
    y_alpha_e,
    '-.',
    linewidth=1.5,
    label='Alpha-blended ECDF (Extended)',
)
ax.plot(x_knee_e, y_knee_e, '-', linewidth=1.5, label='Soft-Knee (Extended)')
ax.plot(
    x_poiss_e,
    y_poiss_e,
    '-',
    linewidth=1.5,
    label='Poisson (Extended, alpha=0.5)',
)
ax.plot(
    x_inp_e,
    y_inp_e,
    '-',
    linewidth=1.5,
    label='Biharmonic Inpaint (Extended)',
)

ax.set_xlabel('Log2 Power Intensity')
ax.set_ylabel('Cumulative Probability')
ax.set_title(
    'ECDF Comparison Inside Original Blobs (Methods Evaluated on Extended Mask)'
)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower right', frameon=True)

plt.show()

# ==============================================================================
# --- Combined Hybrid Methods
# ==============================================================================

# Compute Laplacian of the Alpha-blended ECDF field
laplacian_alpha_ecdf = (
    np.roll(I_alpha_ext, 1, axis=0)
    + np.roll(I_alpha_ext, -1, axis=0)
    + np.roll(I_alpha_ext, 1, axis=1)
    + np.roll(I_alpha_ext, -1, axis=1)
    - 4.0 * I_alpha_ext
)

# ------------------------------------------------------------------------------
# Hybrid 1: Poisson Guided by Alpha-ECDF (Strategy 1)
# ------------------------------------------------------------------------------
I_poisson_ecdf_guided = I_orig.copy()
b_vec_h1 = np.zeros(N_vars_ext, dtype=float)

for i, (r, c) in enumerate(inside_coords_ext):
  b_vec_h1[i] = laplacian_alpha_ecdf[r, c]
  for dr, dc in neighbors:
    nr, nc = r + dr, c + dc
    if 0 <= nr < ny and 0 <= nc < nx:
      if not target_mask[nr, nc]:
        b_vec_h1[i] -= I_orig[nr, nc]
    else:
      b_vec_h1[i] -= I_orig[r, c]

sol_h1 = spsolve(csc_matrix(A_ext), b_vec_h1)
I_poisson_ecdf_guided[target_mask] = sol_h1

# ------------------------------------------------------------------------------
# Hybrid 2: Distance-Adaptive Gradient Poisson (Strategy 2)
# Transitions alpha from 1.0 at border to 0.4 at blob centers
# ------------------------------------------------------------------------------
alpha_spatial = 1.0 - 0.6 * alpha_weight_ext  # 1.0 at edge, 0.4 inside

I_poisson_adaptive = I_orig.copy()
b_vec_h2 = np.zeros(N_vars_ext, dtype=float)

for i, (r, c) in enumerate(inside_coords_ext):
  b_vec_h2[i] = alpha_spatial[r, c] * laplacian_orig[r, c]
  for dr, dc in neighbors:
    nr, nc = r + dr, c + dc
    if 0 <= nr < ny and 0 <= nc < nx:
      if not target_mask[nr, nc]:
        b_vec_h2[i] -= I_orig[nr, nc]
    else:
      b_vec_h2[i] -= I_orig[r, c]

sol_h2 = spsolve(csc_matrix(A_ext), b_vec_h2)
I_poisson_adaptive[target_mask] = sol_h2

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)

panels_hybrid = [
    (axes[0, 0], I_alpha_ext, 'Alpha-blended Local ECDF'),
    (axes[0, 1], I_poisson_ext, 'Standard Poisson (alpha=0.5)'),
    (
        axes[1, 0],
        I_poisson_ecdf_guided,
        'Hybrid 1: ECDF-Guided Poisson (Smooth Edges + Preserved Hills)',
    ),
    (
        axes[1, 1],
        I_poisson_adaptive,
        'Hybrid 2: Distance-Adaptive Poisson (Variable Gradient Attenuation)',
    ),
]

for ax, data, title in panels_hybrid:
  pcm = ax.pcolormesh(
      t_spectro, f_M, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax
  )
  ax.contour(t_spectro, f_M, seg_bool, colors='black', linewidths=0.7)
  ax.contour(
      t_spectro,
      f_M,
      target_mask,
      colors='white',
      linewidths=0.6,
      linestyles='--',
  )
  ax.set_title(title, fontsize=10)
  ax.set_ylabel('Freq (Hz)')
  ax.sharex(axes[0, 0])

fig.colorbar(pcm, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.06)
plt.show()


# ==============================================================================
# --- Local Inversion Analysis & Ordering Feature Maps
# ==============================================================================
from scipy.stats import kendalltau


def compute_inversion_map(I_before, I_after, eval_mask):
  """Computes local edge inversions on 4-connected neighbors within/across eval_mask.

  Returns:
      inversion_count_map (2D array): Number of inverted edges per pixel.
      inversion_rate (float): Fraction of inverted edges over active edges.
  """
  ny, nx = I_before.shape
  inv_count = np.zeros((ny, nx), dtype=float)
  total_edges = 0
  inverted_edges = 0

  # Horizontal neighbor pairs (r, c) <-> (r, c + 1)
  mask_h = eval_mask[:, :-1] | eval_mask[:, 1:]
  diff_orig_h = I_before[:, 1:] - I_before[:, :-1]
  diff_corr_h = I_after[:, 1:] - I_after[:, :-1]

  # Inversion occurs when the non-zero sign flips
  inv_h = (
      (diff_orig_h * diff_corr_h < -1e-9)
      & (np.abs(diff_orig_h) > 1e-6)
      & mask_h
  )

  inv_count[:, :-1] += inv_h
  inv_count[:, 1:] += inv_h
  inverted_edges += np.sum(inv_h)
  total_edges += np.sum(mask_h & (np.abs(diff_orig_h) > 1e-6))

  # Vertical neighbor pairs (r, c) <-> (r + 1, c)
  mask_v = eval_mask[:-1, :] | eval_mask[1:, :]
  diff_orig_v = I_before[1:, :] - I_before[:-1, :]
  diff_corr_v = I_after[1:, :] - I_after[:-1, :]

  inv_v = (
      (diff_orig_v * diff_corr_v < -1e-9)
      & (np.abs(diff_orig_v) > 1e-6)
      & mask_v
  )

  inv_count[:-1, :] += inv_v
  inv_count[1:, :] += inv_v
  inverted_edges += np.sum(inv_v)
  total_edges += np.sum(mask_v & (np.abs(diff_orig_v) > 1e-6))

  inv_rate = (inverted_edges / total_edges) if total_edges > 0 else 0.0
  return inv_count, inv_rate


# Dictionary of methods to test
methods = {
    'Direct ECDF': I_ecdf_ext,
    'Alpha ECDF': I_alpha_ext,
    'Soft Knee': I_knee_ext,
    'Poisson (alpha=0.5)': I_poisson_ext,
    'ECDF-Guided Poisson': I_poisson_ecdf_guided,
    'Distance-Adaptive Poisson': I_poisson_adaptive,
}

# ------------------------------------------------------------------------------
# Visualization: Local Inversion Heatmaps
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
axes = axes.ravel()

# Downsample for faster Kendall tau calculation if mask is very large
eval_indices = np.where(target_mask)
step = max(1, len(eval_indices[0]) // 4000)
orig_sample = I_orig[eval_indices][::step]

for idx, (name, I_corr) in enumerate(methods.items()):
  ax = axes[idx]
  inv_map, inv_rate = compute_inversion_map(I_orig, I_corr, target_mask)

  corr_sample = I_corr[eval_indices][::step]
  tau, _ = kendalltau(orig_sample, corr_sample)

  # Mask unanalyzed regions for display
  inv_map_display = np.ma.masked_where(~target_mask, inv_map)

  pcm = ax.pcolormesh(
      t_spectro,
      f_M,
      inv_map_display,
      shading='nearest',
      cmap='magma',
      vmin=0,
      vmax=4,
  )
  ax.contour(t_spectro, f_M, seg_bool, colors='cyan', linewidths=0.6)
  ax.contour(
      t_spectro,
      f_M,
      target_mask,
      colors='white',
      linewidths=0.5,
      linestyles='--',
  )

  ax.set_title(
      f'{name}\nInverted Edges: {inv_rate*100:.2f}% | Kendall $\\tau$: {tau:.3f}',
      fontsize=9,
  )
  ax.set_ylabel('Freq (Hz)')
  ax.sharex(axes[0])

cbar = fig.colorbar(
    pcm, ax=axes.tolist(), orientation='horizontal', pad=0.05, shrink=0.6
)
cbar.set_label(
    'Inverted Neighbor Count per Pixel (0 = fully preserved, 4 = complete local'
    ' inversion)'
)
plt.show()


# ==============================================================================
# --- Inversion-Free Strong Attenuation Methods
# ==============================================================================

# ------------------------------------------------------------------------------
# Method A: Monotonic Boundary-Anchored ECDF
# ------------------------------------------------------------------------------
# Reference anchor: minimum value along the extended mask edge
T_anchor = np.min(spectro_log[outer_ring])

# ECDF mapping strictly shifted so f(T_anchor) == T_anchor
raw_mapped = ecdf_map_ext(ext_inside_raw)
anchor_mapped = ecdf_map_ext(np.array([T_anchor]))[0]

# Rescale positive excursions from the anchor
gamma_scale = 0.85  # controls strength of reduction
mono_ecdf_vals = ext_inside_raw.copy()
above_anchor = ext_inside_raw > T_anchor
mono_ecdf_vals[above_anchor] = T_anchor + gamma_scale * np.maximum(
    0.0, raw_mapped[above_anchor] - anchor_mapped
)

I_mono_ecdf = I_orig.copy()
I_mono_ecdf[target_mask] = mono_ecdf_vals


# ------------------------------------------------------------------------------
# Method B: Strong C1-Smooth Logarithmic Compression
# ------------------------------------------------------------------------------
# Set threshold at 50th percentile (median) of outer background
T_log = np.median(ext_outside_vals)
beta_param = (
    np.std(ext_outside_vals) * 0.8
)  # smaller beta = stronger reduction

I_strong_log = I_orig.copy()
compress_mask = target_mask & (I_orig > T_log)
I_strong_log[compress_mask] = T_log + beta_param * np.log(
    1.0 + (I_orig[compress_mask] - T_log) / beta_param
)


# ------------------------------------------------------------------------------
# Inversion & Visual Verification
# ------------------------------------------------------------------------------
comp_methods = {
    'Alpha ECDF (Baseline)': I_alpha_ext,
    'Monotonic Anchored ECDF': I_mono_ecdf,
    'Strong Log-Compressor': I_strong_log,
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

for ax, (name, I_corr) in zip(axes, comp_methods.items()):
  inv_map, inv_rate = compute_inversion_map(I_orig, I_corr, target_mask)
  corr_sample = I_corr[eval_indices][::step]
  tau, _ = kendalltau(orig_sample, corr_sample)

  inv_display = np.ma.masked_where(~target_mask, inv_map)
  pcm = ax.pcolormesh(
      t_spectro, f_M, inv_display, shading='nearest', cmap='magma', vmin=0, vmax=4
  )
  ax.contour(t_spectro, f_M, seg_bool, colors='cyan', linewidths=0.6)
  ax.contour(
      t_spectro,
      f_M,
      target_mask,
      colors='white',
      linewidths=0.5,
      linestyles='--',
  )

  ax.set_title(
      f'{name}\nInversions: {inv_rate*100:.2f}% | Kendall $\\tau$: {tau:.3f}',
      fontsize=10,
  )
  ax.set_ylabel('Freq (Hz)')
  ax.sharex(axes[0])

cbar = fig.colorbar(pcm, ax=axes.tolist(), orientation='horizontal', pad=0.08)
cbar.set_label('Inverted Neighbor Count per Pixel')
plt.show()


# ==============================================================================
# --- Visual Comparison: Corrected Spectrograms vs. Original
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)

panels_compare = [
    (axes[0, 0], I_orig, "Original Spectrogram (> 20 Hz)"),
    (axes[0, 1], I_alpha_ext, "Alpha-blended Local ECDF (Reference Baseline)"),
    (
        axes[1, 0],
        I_mono_ecdf,
        "Monotonic Anchored ECDF (Inversion-Free & Strong Attenuation)",
    ),
    (
        axes[1, 1],
        I_strong_log,
        f"Strong C1-Smooth Log-Compressor (Beta={beta_param:.2f}, Strict Tau=1.0)",
    ),
]

for ax, data, title in panels_compare:
  pcm = ax.pcolormesh(
      t_spectro, f_M, data, shading="nearest", cmap="jet", vmin=vmin, vmax=vmax
  )
  # Cyan: Core detected blob contour; White dashed: Extended mask border
  ax.contour(t_spectro, f_M, seg_bool, colors="cyan", linewidths=0.7)
  ax.contour(
      t_spectro,
      f_M,
      target_mask,
      colors="white",
      linewidths=0.6,
      linestyles="--",
  )
  ax.set_title(title, fontsize=10)
  ax.set_ylabel("Freq (Hz)")
  ax.sharex(axes[0, 0])

axes[1, 0].set_xlabel("Time (s)")
axes[1, 1].set_xlabel("Time (s)")

fig.colorbar(
    pcm,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    pad=0.06,
    shrink=0.7,
    label="Log2 Power Intensity",
)
plt.show()