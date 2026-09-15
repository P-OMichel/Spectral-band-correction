"""Segmentation and ECDF knee Correction of Spectral Signatures.
"""

from Functions.generate_OU import get_mixed_OU_signals
from Functions.segment_spectrogram import segment_blobs
from Functions.time_frequency import spectrogram
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from Functions.correct_spectrogram import apply_spatially_conditioned_quantile_blend, apply_monotonic_quantile_blend, apply_alpha_blended_ecdf, apply_rank_preserved_alpha_ecdf
from Functions.utils import get_ecdf, compute_sef95
import matplotlib.patches as patches
import scipy.ndimage as ndi

def get_ecdf_mapped_matrices(I_in, mask, dilation_iter=2):
    """
    Computes the full target mapped matrix T (alpha=1.0) alongside the original.
    
    Returns:
        I_orig : 2D ndarray, exact copy of input
        T      : 2D ndarray, target matrix with mask mapped to boundary ECDF
    """
    I_orig = I_in.copy()
    T = I_in.copy()

    if not np.any(mask):
        return I_orig, T

    # Outside reference ring
    dilated = binary_dilation(mask, iterations=dilation_iter)
    border_ring = dilated & ~mask

    outside_vals = np.sort(I_in[border_ring])
    inside_vals = I_in[mask]

    # Map outside ECDF to inside rank order
    sort_order = np.argsort(inside_vals)
    q_in = np.linspace(0, 1, len(inside_vals))
    q_out = np.linspace(0, 1, len(outside_vals))
    target_sorted = np.interp(q_in, q_out, outside_vals)

    # Invert sorting to reconstruct full target inside region
    target_inside = np.empty_like(inside_vals)
    target_inside[sort_order] = target_sorted
    T[mask] = target_inside

    return T


def plot_spatial_distance_analysis(I_in, mask, vmin, vmax, alpha_cutoff=0.8, zoom_pad=10):
    if not np.any(mask):
        raise ValueError("Mask contains no positive pixels.")

    # 1. Distance transform computation
    dist = ndi.distance_transform_edt(mask)
    inside_vals = I_in[mask]
    sort_order = np.argsort(inside_vals)

    d_max = np.max(dist) if np.max(dist) > 0 else 1.0
    alpha_spatial = np.clip(dist / (d_max * alpha_cutoff + 1e-6), 0.0, 1.0)
    inside_dist_weights = alpha_spatial[mask]

    sorted_spatial_weights = inside_dist_weights[sort_order]
    alpha_quantile = np.maximum.accumulate(sorted_spatial_weights)

    # 2. Key spatial reference points
    max_idx = np.unravel_index(np.argmax(dist), dist.shape)  # (row, col)

    # Find closest boundary pixel (~mask) to max distance point
    outside_coords = np.argwhere(~mask)
    sq_dists_max = np.sum((outside_coords - np.array(max_idx)) ** 2, axis=1)
    closest_outside_max_pt = outside_coords[np.argmin(sq_dists_max)]  # (r_out, c_out)

    # Sample pixel for the zoom panel (e.g., median distance inside mask)
    inside_coords = np.argwhere(mask)
    inside_dists = dist[mask]
    sample_idx = np.argsort(inside_dists)[len(inside_dists) // 2]
    sample_pt = inside_coords[sample_idx]
    d_sample = dist[sample_pt[0], sample_pt[1]]

    closest_outside_sample_pt = outside_coords[np.argmin(np.sum((outside_coords - sample_pt) ** 2, axis=1))]

    # Bounding indices for zoom window
    r_min = max(0, sample_pt[0] - zoom_pad)
    r_max = min(I_in.shape[0], sample_pt[0] + zoom_pad + 1)
    c_min = max(0, sample_pt[1] - zoom_pad)
    c_max = min(I_in.shape[1], sample_pt[1] + zoom_pad + 1)

    # 3. Layout setup
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.2, 1.3], wspace=0.3)

    ax_zoom = fig.add_subplot(gs[0])
    ax_full = fig.add_subplot(gs[1])
    ax_weights = fig.add_subplot(gs[2])

    # -------------------------------------------------------------
    # Panel 1: Zoomed View (shading='nearest')
    # -------------------------------------------------------------
    x_zoom_centers = np.arange(c_min, c_max)
    y_zoom_centers = np.arange(r_min, r_max)
    X_zoom, Y_zoom = np.meshgrid(x_zoom_centers, y_zoom_centers)
    C_zoom = I_in[r_min:r_max, c_min:c_max]

    ax_zoom.pcolormesh(X_zoom, Y_zoom, C_zoom,shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    ax_zoom.contour(x_zoom_centers, y_zoom_centers,mask[r_min:r_max, c_min:c_max],levels=[0.5], colors='black', linewidths=1.2)

    # Distance vector
    ax_zoom.scatter(sample_pt[1], sample_pt[0], color='black', s=45, zorder=5, label='Pixel $p$')
    ax_zoom.scatter(closest_outside_sample_pt[1], closest_outside_sample_pt[0], color='red', s=45, zorder=5, label='Nearest border')
    ax_zoom.annotate("",xy=(closest_outside_sample_pt[1], closest_outside_sample_pt[0]),xytext=(sample_pt[1], sample_pt[0]),arrowprops=dict(arrowstyle="->", color='black', lw=1.6))
    ax_zoom.text((sample_pt[1] + closest_outside_sample_pt[1]) / 2.0 + 0.6,(sample_pt[0] + closest_outside_sample_pt[0]) / 2.0,f"$d = {d_sample:.1f}$",color='black', fontsize=10, weight='bold')
    ax_zoom.set_xlim(c_min - 0.5, c_max - 0.5)
    ax_zoom.set_ylim(r_min - 0.5, r_max - 0.5)
    ax_zoom.set_title(r"Zoomed Region ($d_{min}$)", fontsize=18)
    ax_zoom.legend(loc='upper right', fontsize=14)

    # -------------------------------------------------------------
    # Panel 2: Full Spectrogram View (shading='nearest')
    # -------------------------------------------------------------
    H, W = I_in.shape
    x_full_centers = np.arange(W)
    y_full_centers = np.arange(H)
    X_full, Y_full = np.meshgrid(x_full_centers, y_full_centers)

    ax_full.pcolormesh(
        X_full, Y_full, I_in,
        shading='nearest', cmap='jet', vmin=vmin, vmax=vmax
    )
    ax_full.contour(x_full_centers, y_full_centers, mask, levels=[0.5], colors='black', linewidths=1.5)

    # Max distance marker (dot instead of star) with distance arrow to nearest border
    ax_full.scatter(max_idx[1], max_idx[0],color='black', s=70, linewidth=0.8,label=rf"Max Distance ($d_{{{{\max}}}} = {d_max:.1f}$)", zorder=5)
    ax_full.scatter(closest_outside_max_pt[1], closest_outside_max_pt[0], color='red', s=45, zorder=5)
    ax_full.annotate("",xy=(closest_outside_max_pt[1], closest_outside_max_pt[0]),xytext=(max_idx[1], max_idx[0]),arrowprops=dict(arrowstyle="->", color='black', lw=1.6))

    # Bounding box of the zoom with dotted black style and larger linewidth
    zoom_rect = patches.Rectangle((c_min - 0.5, r_min - 0.5), c_max - c_min, r_max - r_min,linewidth=2.5, edgecolor='black', facecolor='none', linestyle=':', label='Zoom Window')
    ax_full.add_patch(zoom_rect)
    ax_full.set_xlim(-0.5, W - 0.5)
    ax_full.set_ylim(-0.5, H - 0.5)
    ax_full.set_title("Original Spectrogram", fontsize=18)
    ax_full.legend(loc='upper right', fontsize=14)

    # -------------------------------------------------------------
    # Panel 3: Spatial Weights vs. Quantile Rank
    # -------------------------------------------------------------
    rank_axis = np.arange(len(sorted_spatial_weights))
    ax_weights.plot(rank_axis, sorted_spatial_weights, color='gray', alpha=0.45, lw=0.8, label=r"$\alpha_{spatial}$ (Rank-ordered)")
    #ax_weights.plot(rank_axis, alpha_quantile, color='blue', lw=2.2, label=r"$\alpha_{quantile}$ (Monotonic Accum.)")
    ax_weights.set_xlabel("Sorted Pixel Index (Quantile Rank)")
    ax_weights.set_ylabel(r"Weight $\alpha$")
    ax_weights.set_ylim(-0.05, 1.05)
    ax_weights.grid(True, linestyle=':', alpha=0.6)
    ax_weights.set_title(r"Pixel intensity ordered $\alpha$ distances", fontsize=18)
    ax_weights.legend(loc='lower right', fontsize=14)

    return fig

# --- generate synthetic signal
T = 20  # Duration (s)
dt = 0.001
fs = 1 / dt

lbda_list = [1, 2, 1]
omega_list = [2 * np.pi * 1, 2 * np.pi * 10, 2 * np.pi * 30]
sigma_list = [3, 2, 2]
factor_list = [1, 1, 1]

t, y = get_mixed_OU_signals(T, dt, lbda_list, omega_list, sigma_list, factor_list)

f_spectro, t_spectro, spectro = spectrogram(y, fs, nfft_factor=2)

# Focus strictly on frequencies >= 20 Hz
mask_f = f_spectro >= 20
f_M = f_spectro[mask_f]
M = spectro[mask_f, :]
spectro_log = np.log2(M + 1e-11)
I_orig = spectro_log.copy()

# --- Segmentation of the pixels to correct
f_int = [25, 35]
seg_mask, psd, baseline, T_low, T_high = segment_blobs(M, f_M, f_int)
seg_bool = seg_mask.astype(bool)
# Extended mask via morphological dilation (2 iterations)
expanded_mask = binary_dilation(seg_bool, iterations=2)
ref_mask = binary_dilation(expanded_mask, iterations=2)
border_ring = ref_mask & ~expanded_mask





# --- Display Segmentation + M_in + M_in-->out

x_in, y_in = get_ecdf(I_orig[expanded_mask])
x_ref, y_ref = get_ecdf(I_orig[border_ring])

shade_opacity = 0.65  # 0.0 = fully clear, 1.0 = fully opaque
overlay = np.zeros((*expanded_mask.shape, 4), dtype=float)
overlay[~expanded_mask] = [0.0, 0.0, 0.0, shade_opacity]

I_in_out = get_ecdf_mapped_matrices(I_orig, expanded_mask)

fig, axes = plt.subplots(1, 4, constrained_layout = True)
vmin = np.percentile(I_orig, 5)
vmax = np.percentile(I_orig, 99.5)
axes[0].pcolormesh(t_spectro, f_M, I_orig, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes[0].contour(t_spectro, f_M, expanded_mask, colors='black', linewidths=2)
axes[0].contour(t_spectro, f_M, ref_mask, linestyle = '--', colors='black', linewidths=2)
axes[0].set_title(r'Original Spectrogram, $M$', fontsize=18)
axes[0].set_ylabel('Frequency (Hz)', fontsize = 14)
axes[1].plot(x_in, y_in, label = 'Inside mask')
axes[1].plot(x_ref, y_ref, label = 'Border ring')
axes[1].set_title('ECDF', fontsize=18)
axes[1].legend()
axes[2].pcolormesh(t_spectro, f_M, I_orig, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes[2].pcolormesh(t_spectro, f_M, overlay, shading='nearest', cmap='jet')
axes[2].contour(t_spectro, f_M, expanded_mask, levels=[0.5], colors='black', linewidths=1.5)
axes[2].set_title(r'$M_{in}$', fontsize=18)
axes[3].pcolormesh(t_spectro, f_M, I_in_out, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
axes[3].pcolormesh(t_spectro, f_M, overlay, shading='nearest', cmap='jet')
axes[3].contour(t_spectro, f_M, expanded_mask, levels=[0.5], colors='black', linewidths=1.5)
axes[3].set_title(r'$M^{in \rightarrow out}$', fontsize=18)

plt.show()




# --- Distance computation


fig = plot_spatial_distance_analysis(I_orig, expanded_mask, vmin, vmax)

plt.show()





# --- Correction
# 1. Run all 4 correction methods
I_a_blend, ref_vals_a_blend = apply_alpha_blended_ecdf(I_orig, expanded_mask, alpha_cutoff=0.5)
I_m_q_blend, ref_vals_m_q_blend = apply_monotonic_quantile_blend(I_orig, expanded_mask, alpha_blend=0.8)
I_a_r_blend, ref_vals_a_r_blend = apply_rank_preserved_alpha_ecdf(I_orig, expanded_mask, alpha_cutoff=0.5)
I_s_c_q_blend, ref_vals_s_c_q_blend = apply_spatially_conditioned_quantile_blend(I_orig, expanded_mask, alpha_cutoff=0.5)

# 2. Compute ECDFs for inside mask
x_orig, y_orig = get_ecdf(I_orig[seg_bool])
x_a, y_a = get_ecdf(I_a_blend[seg_bool])
x_mq, y_mq = get_ecdf(I_m_q_blend[seg_bool])
x_ar, y_ar = get_ecdf(I_a_r_blend[seg_bool])
x_scq, y_scq = get_ecdf(I_s_c_q_blend[seg_bool])

# Reference distribution (boundary ring)
x_ref, y_ref = get_ecdf(ref_vals_a_blend)

# 3. Setup Grid Layout: 2x2 Spectrograms on left, 1 tall ECDF plot on right
fig = plt.figure(figsize=(18, 9), constrained_layout=True)
gs_main = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0])
gs_spec = gs_main[0].subgridspec(2, 2, hspace=0.15, wspace=0.15)

spectro_configs = [
    (0, 0, I_a_blend, 'Alpha-Blended ECDF'),(0, 1, I_m_q_blend, 'Monotonic Quantile Blend'),
    (1, 0, I_a_r_blend, 'Rank-Preserved Alpha ECDF'),(1, 1, I_s_c_q_blend, 'Spatially Conditioned Quantile Blend'),
]

spec_axes = []
for r, c, data, title in spectro_configs:
    ax = fig.add_subplot(gs_spec[r, c])
    spec_axes.append(ax)
    pcm = ax.pcolormesh(t_spectro, f_M, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    ax.contour(t_spectro, f_M, expanded_mask, levels=[0.5], colors='black', linewidths=1.5)
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    if c == 0:
        ax.set_ylabel('Frequency (Hz)', fontsize=14)
    else:
        ax.set_yticklabels([])
    if r == 1:
        ax.set_xlabel('Time (s)', fontsize=14)
    else:
        ax.set_xticklabels([])

# fig.colorbar(pcm, ax=spec_axes, orientation='horizontal', fraction=0.04, pad=0.08, label=r'$\log_2(\text{Power})$')

# 4. Right Panel: Combined ECDF Curves
ax_ecdf = fig.add_subplot(gs_main[1])

ax_ecdf.plot(x_ref, y_ref, 'k--', linewidth=2.0, label='Reference Ring')
ax_ecdf.plot(x_orig, y_orig, color='gray', linewidth=2.0, label='Original Inside Blobs')
ax_ecdf.plot(x_a, y_a, linewidth=2.2, label='Alpha-Blended')
ax_ecdf.plot(x_mq, y_mq, linewidth=2.2, label='Monotonic Quantile')
ax_ecdf.plot(x_ar, y_ar, linewidth=2.2, label='Rank-Preserved Alpha')
ax_ecdf.plot(x_scq, y_scq, linewidth=2.2, label='Spatially Cond. Quantile')

ax_ecdf.set_title('ECDF Comparison Inside Mask', fontsize=18)
ax_ecdf.set_xlabel(r'Intensity ($\log_2\text{ Power}$)', fontsize=12)
ax_ecdf.set_ylabel('Empirical Cumulative Probability', fontsize=12)
ax_ecdf.grid(True, linestyle='--', alpha=0.5)
ax_ecdf.legend(loc='upper left', frameon=True, fontsize=14)

plt.show()