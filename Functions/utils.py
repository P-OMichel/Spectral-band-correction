import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

def get_ecdf(vals):
  x = np.sort(vals.ravel())
  y = np.linspace(0, 1, len(x))
  return x, y

def compute_sef95(freqs, psd_profile):
  """Computes Spectral Edge Frequency (95%) from a 1D PSD profile."""
  cum_power = np.cumsum(psd_profile)
  total_power = cum_power[-1]
  if total_power <= 0:
    return freqs[0]
  idx = np.searchsorted(cum_power, 0.95 * total_power)
  idx = min(idx, len(freqs) - 1)
  return freqs[idx]

def plot_rank_inversion_heatmap(I_orig, I_corr, mask=None, figsize=(10, 8), cmap="magma"):
    """
    Plots a heatmap showing where relative pixel rank ordering shifted between 
    the original and corrected matrices, with an optional mask contour overlay.

    Parameters:
        I_orig  : 2D ndarray, original spectrogram
        I_corr  : 2D ndarray, corrected spectrogram
        mask    : 2D bool ndarray, region of interest / blob mask
        figsize : tuple, figure dimensions (width, height)
        cmap    : str, colormap for the rank displacement heatmap
    """
    if mask is not None and np.any(mask):
        eval_region = mask
    else:
        eval_region = np.ones_like(I_orig, dtype=bool)

    orig_vals = I_orig[eval_region]
    corr_vals = I_corr[eval_region]
    N = len(orig_vals)

    # Compute normalized rank (0.0 to 1.0) for both distributions
    # 'average' handles identical value ties gracefully
    rank_orig = (rankdata(orig_vals, method='average') - 1.0) / max(N - 1, 1)
    rank_corr = (rankdata(corr_vals, method='average') - 1.0) / max(N - 1, 1)

    # Calculate absolute displacement in percentile rank
    rank_discrepancy = np.abs(rank_orig - rank_corr)

    # Reconstruct 2D spatial error map
    error_map = np.zeros_like(I_orig, dtype=float)
    error_map[eval_region] = rank_discrepancy

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(error_map, origin='lower', cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Percentile Rank Shift |Δ Rank|", fontsize=11)

    if mask is not None and np.any(mask):
        # Draw target blob boundary contour in bright green
        ax.contour(mask.astype(float), levels=[0.5], colors='lime', linewidths=1.8)

    max_shift = np.max(rank_discrepancy) if len(rank_discrepancy) > 0 else 0.0
    mean_shift = np.mean(rank_discrepancy) if len(rank_discrepancy) > 0 else 0.0

    ax.set_title(
        f"Rank Discrepancy Heatmap\n(Max Shift: {max_shift * 100:.2f}%, Mean Shift: {mean_shift * 100:.2f}%)",
        fontsize=12,
        fontweight='bold'
    )
    ax.set_xlabel("Time Bins", fontsize=10)
    ax.set_ylabel("Frequency Bins", fontsize=10)

    plt.tight_layout()
    plt.show()

    return error_map


def plot_global_rank_inversion_heatmap(I_orig, I_corr, mask=None, figsize=(10, 8), cmap="magma"):
    """
    Plots a heatmap showing where relative pixel rank ordering shifted across the
    entire matrix, with an optional mask contour overlay for localization.

    Parameters:
        I_orig  : 2D ndarray, original spectrogram
        I_corr  : 2D ndarray, corrected spectrogram
        mask    : 2D bool ndarray, optional region of interest/blob mask for contour overlay
        figsize : tuple, figure dimensions (width, height)
        cmap    : str, colormap for rank displacement (0.0 to 1.0)
    """
    orig_flat = I_orig.ravel()
    corr_flat = I_corr.ravel()
    total_pixels = len(orig_flat)

    # Compute global normalized rank in [0, 1] for all pixels in the full spectrogram
    rank_orig = (rankdata(orig_flat, method='average') - 1.0) / (total_pixels - 1)
    rank_corr = (rankdata(corr_flat, method='average') - 1.0) / (total_pixels - 1)

    # Global rank displacement across the entire matrix
    global_rank_diff = np.abs(rank_orig - rank_corr)
    error_map = global_rank_diff.reshape(I_orig.shape)

    # Calculate summary metrics
    global_max_shift = np.max(error_map)
    global_mean_shift = np.mean(error_map)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(error_map, origin='lower', cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Global Percentile Rank Shift |Δ Rank|", fontsize=11)

    if mask is not None and np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors='lime', linewidths=1.8)
        mask_shift_mean = np.mean(error_map[mask])
        title_suffix = f"\n(Whole Matrix Mean: {global_mean_shift * 100:.2f}%, Inside Mask Mean: {mask_shift_mean * 100:.2f}%, Max: {global_max_shift * 100:.2f}%)"
    else:
        title_suffix = f"\n(Whole Matrix Mean: {global_mean_shift * 100:.2f}%, Max: {global_max_shift * 100:.2f}%)"

    ax.set_title(f"Global Rank Discrepancy Heatmap{title_suffix}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time Bins", fontsize=10)
    ax.set_ylabel("Frequency Bins", fontsize=10)

    plt.tight_layout()
    plt.show()

    return error_map



from scipy.ndimage import binary_dilation, binary_erosion

def plot_local_boundary_inversion_heatmap(
    I_orig, I_corr, mask, boundary_width=2, figsize=(10, 8), cmap="Reds"
):
    """
    Evaluates and plots local rank preservation specifically between pixels inside 
    the mask and their immediate neighbors outside the mask.

    Parameters:
        I_orig         : 2D ndarray, original spectrogram
        I_corr         : 2D ndarray, corrected spectrogram
        mask           : 2D bool ndarray, target region
        boundary_width : int, thickness of the inner and outer border band to check
        figsize        : tuple, plot dimensions
        cmap           : str, colormap for the inversion rate (0.0 to 1.0)
    """
    # 1. Define inner border (inside mask) and outer border (outside mask)
    inner_eroded = binary_erosion(mask, iterations=boundary_width)
    inner_border = mask & ~inner_eroded

    outer_dilated = binary_dilation(mask, iterations=boundary_width)
    outer_border = outer_dilated & ~mask

    # Active evaluation zone spanning both sides of the boundary
    local_eval_zone = inner_border | outer_border

    # 2. Extract pairs of neighboring pixels across the seam using standard 8-connectivity
    rows, cols = I_orig.shape
    inversion_counts = np.zeros_like(I_orig, dtype=float)
    pair_counts = np.zeros_like(I_orig, dtype=float)

    # Direction offsets for 8-neighborhood
    shifts = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for dr, dc in shifts:
        # Shift mask and borders to align neighbors
        r_src = slice(max(0, -dr), min(rows, rows - dr))
        c_src = slice(max(0, -dc), min(cols, cols - dc))
        r_dst = slice(max(0, dr), min(rows, rows + dr))
        c_dst = slice(max(0, dc), min(cols, cols + dc))

        # Check neighbor pairs where one is inside mask and one is outside
        p_in = mask[r_src, c_src]
        q_out = (~mask)[r_dst, c_dst]

        # Restrict to the specified border width
        active_pairs = p_in & q_out & local_eval_zone[r_src, c_src] & local_eval_zone[r_dst, c_dst]

        if not np.any(active_pairs):
            continue

        # Extract values for both original and corrected spectrograms
        orig_p = I_orig[r_src, c_src]
        orig_q = I_orig[r_dst, c_dst]
        corr_p = I_corr[r_src, c_src]
        corr_q = I_corr[r_dst, c_dst]

        # Identify strictly inverted pairs:
        # Condition: (p > q and p' < q') OR (p < q and p' > q')
        inversion = active_pairs & (
            ((orig_p > orig_q) & (corr_p < corr_q)) |
            ((orig_p < orig_q) & (corr_p > corr_q))
        )

        # Accumulate metrics onto both pixels of the pair
        inversion_counts[r_src, c_src] += inversion
        inversion_counts[r_dst, c_dst] += inversion
        pair_counts[r_src, c_src] += active_pairs
        pair_counts[r_dst, c_dst] += active_pairs

    # 3. Calculate local inversion rate per pixel
    local_inversion_rate = np.zeros_like(I_orig, dtype=float)
    valid = pair_counts > 0
    local_inversion_rate[valid] = inversion_counts[valid] / pair_counts[valid]

    # 4. Plotting
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        local_inversion_rate,
        origin='lower',
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect='auto'
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Local Pairwise Inversion Rate across Seam", fontsize=11)

    # Overlay mask contour
    ax.contour(mask.astype(float), levels=[0.5], colors='deepskyblue', linewidths=1.8)

    total_inversions = np.sum(inversion_counts[valid]) / 2  # Each pair counted twice
    total_pairs = np.sum(pair_counts[valid]) / 2
    avg_rate = (total_inversions / total_pairs * 100) if total_pairs > 0 else 0.0

    ax.set_title(
        f"Local Boundary Rank Inversion Heatmap\n"
        f"(Seam Inversion Rate: {avg_rate:.2f}%, Inverted Pairs: {int(total_inversions)}/{int(total_pairs)})",
        fontsize=12,
        fontweight='bold'
    )
    ax.set_xlabel("Time Bins", fontsize=10)
    ax.set_ylabel("Frequency Bins", fontsize=10)

    plt.tight_layout()
    plt.show()

    return local_inversion_rate