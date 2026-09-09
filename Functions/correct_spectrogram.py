'''
File for function to correct segmented spectrogram of too high intensity bins.
'''
import numpy as np
from scipy.ndimage import binary_dilation
import scipy.ndimage as ndi
from scipy.interpolate import interp1d

def apply_alpha_blended_ecdf(I_in, mask, dilation_iter=2, alpha_cutoff=0.8):
  """Applies local ECDF matching with distance-transform alpha blending.

  Parameters:
      I_in : 2D ndarray, input spectrogram in log scale
      mask : 2D bool ndarray, target region to correct
      dilation_iter : int, border expansion steps to form reference ring
      alpha_cutoff : float, distance scaling factor for transition ring
  """
  # Construct local outside reference ring
  dilated = binary_dilation(mask, iterations=dilation_iter)
  border_ring = dilated & ~mask

  outside_vals = np.sort(I_in[border_ring])
  inside_vals = I_in[mask]

  # Quantile-to-quantile transfer (ECDF mapping)
  N_in = len(inside_vals)
  N_out = len(outside_vals)
  q_in = np.linspace(0, 1, N_in)
  q_out = np.linspace(0, 1, N_out)

  ecdf_map = interp1d(np.sort(inside_vals),np.interp(q_in, q_out, outside_vals),bounds_error=False,fill_value='extrapolate')
  I_ecdf_direct = ecdf_map(inside_vals)

  # Distance transform from mask edge inward
  dist = ndi.distance_transform_edt(mask)
  d_max = np.max(dist) if np.max(dist) > 0 else 1.0
  alpha_weight = np.clip(dist / (d_max * alpha_cutoff + 1e-6), 0.0, 1.0)

  # Alpha blend: 0 at boundary (retaining exterior values), 1 at core
  I_out = I_in.copy()
  I_out[mask] = (1.0 - alpha_weight[mask]) * inside_vals + alpha_weight[mask] * I_ecdf_direct
  return I_out, outside_vals


def apply_monotonic_quantile_blend(I_in, mask, dilation_iter=2, alpha_blend=1.0):
    """
    Applies quantile-level blending to guarantee strict preservation of pixel order.

    Parameters:
        I_in : 2D ndarray, input spectrogram in log scale
        mask : 2D bool ndarray, target region to correct
        dilation_iter : int, border expansion steps to form reference ring
        alpha_blend : float in [0, 1], blending weight:
                      0.0 = original values (identity)
                      1.0 = full boundary ECDF matching
    """
    if not np.any(mask):
        return I_in.copy(), np.array([])

    # Construct local outside reference ring
    dilated = binary_dilation(mask, iterations=dilation_iter)
    border_ring = dilated & ~mask

    outside_vals = np.sort(I_in[border_ring])
    inside_vals = I_in[mask]

    N_in = len(inside_vals)
    N_out = len(outside_vals)

    # 1. Rank inside pixels and get their sorted values
    sort_order = np.argsort(inside_vals)
    inside_sorted = inside_vals[sort_order]

    # 2. Map reference outside distribution onto the inside quantile grid
    q_in = np.linspace(0, 1, N_in)
    q_out = np.linspace(0, 1, N_out)
    target_sorted = np.interp(q_in, q_out, outside_vals)

    # 3. Blend the quantile functions directly (guarantees monotonicity)
    blended_sorted = (1.0 - alpha_blend) * inside_sorted + alpha_blend * target_sorted

    # 4. Invert sorting permutation to restore original spatial positions
    corrected_inside = np.empty_like(inside_vals)
    corrected_inside[sort_order] = blended_sorted

    I_out = I_in.copy()
    I_out[mask] = corrected_inside

    return I_out, outside_vals


def apply_monotonic_quantile_blend_1(I_in, mask, max_expand_iters=25, alpha_blend=1.0):
    """
    Applies quantile-level blending to guarantee strict preservation of pixel order.

    Parameters:
        I_in : 2D ndarray, input spectrogram in log scale
        mask : 2D bool ndarray, target region to correct
        dilation_iter : int, border expansion steps to form reference ring
        alpha_blend : float in [0, 1], blending weight:
                      0.0 = original values (identity)
                      1.0 = full boundary ECDF matching
    """
    if not np.any(mask):
        return I_in.copy(), np.array([])

    # 1. Expand the mask downhill until resting at local minima
    active_mask = expand_to_local_minima(I_in, mask, max_iters=max_expand_iters)

    # 2. Extract reference ring directly from the outer boundary at the valley floor
    dilated_boundary = binary_dilation(active_mask, iterations=1)
    border_ring = dilated_boundary & ~active_mask

    outside_vals = np.sort(I_in[border_ring])
    inside_vals = I_in[mask]

    N_in = len(inside_vals)
    N_out = len(outside_vals)

    # 1. Rank inside pixels and get their sorted values
    sort_order = np.argsort(inside_vals)
    inside_sorted = inside_vals[sort_order]

    # 2. Map reference outside distribution onto the inside quantile grid
    q_in = np.linspace(0, 1, N_in)
    q_out = np.linspace(0, 1, N_out)
    target_sorted = np.interp(q_in, q_out, outside_vals)

    # 3. Blend the quantile functions directly (guarantees monotonicity)
    blended_sorted = (1.0 - alpha_blend) * inside_sorted + alpha_blend * target_sorted

    # 4. Invert sorting permutation to restore original spatial positions
    corrected_inside = np.empty_like(inside_vals)
    corrected_inside[sort_order] = blended_sorted

    I_out = I_in.copy()
    I_out[mask] = corrected_inside

    return I_out, outside_vals




def apply_rank_preserved_alpha_ecdf(I_in, mask, dilation_iter=2, alpha_cutoff=0.8):
    """
    Applies distance-transform alpha-blended ECDF matching while strictly
    preserving the original pixel rank ordering inside the mask.

    Parameters:
        I_in : 2D ndarray, input spectrogram in log scale
        mask : 2D bool ndarray, target region to correct
        dilation_iter : int, border expansion steps to form reference ring
        alpha_cutoff : float, distance scaling factor for transition ring
    """
    if not np.any(mask):
        return I_in.copy(), np.array([])

    # Construct local outside reference ring
    dilated = binary_dilation(mask, iterations=dilation_iter)
    border_ring = dilated & ~mask

    outside_vals = np.sort(I_in[border_ring])
    inside_vals = I_in[mask]

    # Quantile-to-quantile transfer (ECDF mapping)
    N_in = len(inside_vals)
    N_out = len(outside_vals)
    q_in = np.linspace(0, 1, N_in)
    q_out = np.linspace(0, 1, N_out)

    ecdf_map = interp1d(
        np.sort(inside_vals),
        np.interp(q_in, q_out, outside_vals),
        bounds_error=False,
        fill_value='extrapolate'
    )
    I_ecdf_direct = ecdf_map(inside_vals)

    # Distance transform from mask edge inward
    dist = ndi.distance_transform_edt(mask)
    d_max = np.max(dist) if np.max(dist) > 0 else 1.0
    alpha_weight = np.clip(dist / (d_max * alpha_cutoff + 1e-6), 0.0, 1.0)

    # Compute standard spatial alpha-blended values
    blended_vals = (1.0 - alpha_weight[mask]) * inside_vals + alpha_weight[mask] * I_ecdf_direct

    # Rank-preserving step:
    # Sort blended values ascendingly and map them to the original ascending pixel ranks
    orig_rank_indices = np.argsort(inside_vals)
    sorted_blended_vals = np.sort(blended_vals)

    corrected_inside = np.empty_like(inside_vals)
    corrected_inside[orig_rank_indices] = sorted_blended_vals

    I_out = I_in.copy()
    I_out[mask] = corrected_inside

    return I_out, outside_vals



def expand_to_local_minima(I_in, mask, max_iters=25):
    """
    Expands the mask strictly downhill. For every advancing frontier pixel,
    it steps down into unmasked strictly lower neighbors. A path terminates
    when all outward neighbors are >= current pixel (local minimum reached).
    """
    rows, cols = I_in.shape
    expanded_mask = mask.copy()
    current_frontier = set(zip(*np.where(mask)))

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for _ in range(max_iters):
        next_frontier = set()

        for r, c in current_frontier:
            curr_val = I_in[r, c]
            lowest_val = curr_val
            lowest_neighbor = None

            for dr, dc in shifts:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not expanded_mask[nr, nc]:
                        # Check strictly lower unvisited neighbor
                        if I_in[nr, nc] < lowest_val:
                            lowest_val = I_in[nr, nc]
                            lowest_neighbor = (nr, nc)

            if lowest_neighbor is not None:
                next_frontier.add(lowest_neighbor)
                expanded_mask[lowest_neighbor[0], lowest_neighbor[1]] = True

        if not next_frontier:
            break

        current_frontier = next_frontier

    return expanded_mask


def apply_rank_preserved_alpha_ecdf_1(I_in, mask, max_expand_iters=25, alpha_cutoff=0.8):
    """
    Applies distance-transform alpha-blended ECDF matching while strictly
    preserving the original pixel rank ordering inside the mask, using
    local minima descent to determine mask expansion and reference borders.

    Parameters:
        I_in : 2D ndarray, input spectrogram in log scale
        mask : 2D bool ndarray, target region to correct
        max_expand_iters : int, maximum downhill descent steps toward local minima
        alpha_cutoff : float, distance scaling factor for transition ring
    """
    if not np.any(mask):
        return I_in.copy(), np.array([])

    # 1. Expand the mask downhill until resting at local minima
    active_mask = expand_to_local_minima(I_in, mask, max_iters=max_expand_iters)

    # 2. Extract reference ring directly from the outer boundary at the valley floor
    dilated_boundary = binary_dilation(active_mask, iterations=1)
    border_ring = dilated_boundary & ~active_mask

    # Fallback in case mask touches edges and border_ring is empty
    if not np.any(border_ring):
        return I_in.copy(), np.array([])

    outside_vals = np.sort(I_in[border_ring])
    inside_vals = I_in[active_mask]

    # 3. Quantile-to-quantile transfer (ECDF mapping)
    N_in = len(inside_vals)
    N_out = len(outside_vals)
    q_in = np.linspace(0, 1, N_in)
    q_out = np.linspace(0, 1, N_out)

    ecdf_map = interp1d(
        np.sort(inside_vals),
        np.interp(q_in, q_out, outside_vals),
        bounds_error=False,
        fill_value='extrapolate'
    )
    I_ecdf_direct = ecdf_map(inside_vals)

    # 4. Distance transform from the expanded local minima boundary inward
    dist = ndi.distance_transform_edt(active_mask)
    d_max = np.max(dist) if np.max(dist) > 0 else 1.0
    alpha_weight = np.clip(dist / (d_max * alpha_cutoff + 1e-6), 0.0, 1.0)

    # 5. Distance-weighted alpha blend
    blended_vals = (1.0 - alpha_weight[active_mask]) * inside_vals + alpha_weight[active_mask] * I_ecdf_direct

    # 6. Rank-preserving step
    orig_rank_indices = np.argsort(inside_vals)
    sorted_blended_vals = np.sort(blended_vals)

    corrected_inside = np.empty_like(inside_vals)
    corrected_inside[orig_rank_indices] = sorted_blended_vals

    # 7. Write back corrected region
    I_out = I_in.copy()
    I_out[active_mask] = corrected_inside

    return I_out, outside_vals











def apply_bounded_quantile_correction(I_in, mask, dilation_iter=2, L_ratio=0.95, k=3):
    """Corrects segmented regions of high intensity via logistic sigmoidal suppression.

    Parameters:
        I_in : 2D ndarray, input spectrogram in log scale
        mask : 2D bool ndarray, target region to correct
        dilation_iter : int, border expansion steps to sample the neighborhood
        L_ratio : float, fraction of max peak elevation used as plateau level L
        k : float, growth steepness of the logistic sigmoid

    Returns:
        I_out : 2D ndarray, corrected spectrogram
        outside_vals : 1D ndarray, sorted neighborhood border values
    """

    # Step 1: Construct local outside reference ring & extract sets
    dilated = binary_dilation(mask, iterations=dilation_iter)
    border_ring = dilated & ~mask

    outside_vals = np.sort(I_in[border_ring])
    inside_vals = I_in[mask]

    if len(outside_vals) == 0 or len(inside_vals) == 0:
        return I_in.copy(), outside_vals

    # Step 2: Fix threshold to maximum border value
    t_seg = outside_vals[-1]

    # Step 3: Sort inside values for rank preservation
    N_in = len(inside_vals)
    sort_idx = np.argsort(inside_vals)
    inside_sorted = inside_vals[sort_idx]

    # Step 4: Apply anchored logistic sigmoid suppression above t_seg
    corrected_sorted = inside_sorted.copy()
    mask_above = inside_sorted > t_seg

    if np.any(mask_above):
        x_above = inside_sorted[mask_above]
        delta_x = x_above - t_seg

        # Sigmoid parameters
        L = np.max(delta_x) * L_ratio
        x0 = np.median(delta_x)

        # Logistic sigmoid S-curve
        sigmoid_shift = L / (1.0 + np.exp(-k * (delta_x - x0)))

        # Zero-anchoring at delta_x = 0 (t_seg)
        shift_anchored = sigmoid_shift - (L / (1.0 + np.exp(k * x0)))

        # Shift values downward
        transformed_above = x_above - shift_anchored

        # Ensure values stay attenuated and maintain strict monotonicity
        corrected_sorted[mask_above] = np.maximum.accumulate(
            np.minimum(x_above, transformed_above)
        )

    # Step 5: Remap back to original spatial layout
    remapped_inside = np.empty_like(corrected_sorted)
    remapped_inside[sort_idx] = corrected_sorted

    I_out = I_in.copy()
    I_out[mask] = remapped_inside

    return I_out, outside_vals




def apply_monotonic_floor_sigmoid(I_in, mask, dilation_iter=2, delta_x_plateau=0.50):
    """Corrects segmented regions via local boundary floor extraction and

    logit-calibrated inverse sigmoidal quantile reshaping.

    Parameters:
        I_in : 2D ndarray, input spectrogram in log scale
        mask : 2D bool/int ndarray, target region to correct
        dilation_iter : int, border expansion steps to sample the neighborhood
        delta_x_plateau : float, headroom allowance Delta x (in log2 units)

    Returns:
        I_out : 2D ndarray, corrected spectrogram
        outside_vals : 1D ndarray, sorted neighborhood border values
    """
    seg_bool = mask.astype(bool)
    raw_inside = I_in[seg_bool]

    # Step 1: Construct local outside reference ring & extract sets
    expanded_mask = binary_dilation(seg_bool, iterations=dilation_iter)
    border_mask = expanded_mask & ~seg_bool

    outside_vals = np.sort(I_in[border_mask])

    if len(raw_inside) == 0 or len(outside_vals) == 0:
        return I_in.copy(), outside_vals

    # Step 2: Sort inside values for rank preservation
    sort_idx = np.argsort(raw_inside)
    sorted_raw = raw_inside[sort_idx]
    N = len(sorted_raw)

    # Step 3: Extract local border floor per pixel
    global_border_max = outside_vals[-1]
    inside_coords = np.argwhere(seg_bool)

    pixel_floors = []
    H, W = I_in.shape
    for f_i, t_i in inside_coords:
        f_min, f_max = max(0, f_i - 2), min(H, f_i + 3)
        t_min, t_max = max(0, t_i - 2), min(W, t_i + 3)

        local_win = border_mask[f_min:f_max, t_min:t_max]
        if np.any(local_win):
            pixel_floors.append(np.max(I_in[f_min:f_max, t_min:t_max][local_win]))
        else:
            pixel_floors.append(global_border_max)

    # Align floors with sorted inside values
    pixel_floors = np.array(pixel_floors)[sort_idx]

    # Monotonic baseline sequence (min_rank_vals)
    min_rank_vals = np.maximum.accumulate(pixel_floors)
    min_rank_vals = np.minimum(min_rank_vals, sorted_raw)
    min_rank_vals = np.maximum.accumulate(min_rank_vals)

    # Step 4: Locate transition knee and calibration anchors
    knee_idx = np.argmax(min_rank_vals)
    x_knee = min_rank_vals[knee_idx]
    y_knee = (knee_idx + 1) / N

    x1, y1 = x_knee, np.clip(y_knee, 0.05, 0.95)
    x2, y2 = x_knee + delta_x_plateau, 0.99

    # Solve parameters k and x0 using logit transform
    logit_y1 = np.log(y1 / (1.0 - y1))
    logit_y2 = np.log(y2 / (1.0 - y2))

    k_sig = (logit_y2 - logit_y1) / (x2 - x1)
    x0_sig = x1 - (logit_y1 / k_sig)

    # Step 5: Inverse sigmoid evaluation
    y_ranks = np.linspace(0.001, 0.999, N)
    sigmoid_vals = x0_sig + (np.log(y_ranks / (1.0 - y_ranks)) / k_sig)

    # Enforce monotonicity and floor bounds
    final_sigmoid_vals = np.maximum(sigmoid_vals, min_rank_vals)
    final_sigmoid_vals = np.minimum(final_sigmoid_vals, sorted_raw)
    final_sigmoid_vals = np.maximum.accumulate(final_sigmoid_vals)

    # Step 6: Remap back to 2D matrix
    remapped_sig = np.empty_like(final_sigmoid_vals)
    remapped_sig[sort_idx] = final_sigmoid_vals

    I_out = I_in.copy()
    I_out[seg_bool] = remapped_sig

    return I_out, outside_vals


def correct_by_logit_anchored_sigmoid(
    I_in, mask, dilation_iter=2, delta_x_plateau=0.50
):
    """Corrects segmented regions via local boundary floor extraction and

    logit-calibrated inverse sigmoidal quantile reshaping.

    Returns:
        I_out_sig : 2D ndarray, sigmoid-corrected spectrogram
        I_out_mono : 2D ndarray, monotonic baseline floor spectrogram
        outside_vals : 1D ndarray, sorted neighborhood border values
    """
    seg_bool = mask.astype(bool)
    raw_inside = I_in[seg_bool]

    # Step 1: Construct local outside reference ring & extract sets
    expanded_mask = binary_dilation(seg_bool, iterations=dilation_iter)
    border_mask = expanded_mask & ~seg_bool

    outside_vals = np.sort(I_in[border_mask])

    if len(raw_inside) == 0 or len(outside_vals) == 0:
        return I_in.copy(), I_in.copy(), outside_vals

    # Step 2: Sort inside values for rank preservation
    sort_idx = np.argsort(raw_inside)
    sorted_raw = raw_inside[sort_idx]
    N = len(sorted_raw)

    # Step 3: Extract local border floor per pixel
    global_border_max = outside_vals[-1]
    inside_coords = np.argwhere(seg_bool)

    pixel_floors = []
    H, W = I_in.shape
    for f_i, t_i in inside_coords:
        f_min, f_max = max(0, f_i - 2), min(H, f_i + 3)
        t_min, t_max = max(0, t_i - 2), min(W, t_i + 3)

        local_win = border_mask[f_min:f_max, t_min:t_max]
        if np.any(local_win):
            pixel_floors.append(np.max(I_in[f_min:f_max, t_min:t_max][local_win]))
        else:
            pixel_floors.append(global_border_max)

    # Align floors with sorted inside values
    pixel_floors = np.array(pixel_floors)[sort_idx]

    # Monotonic baseline sequence (min_rank_vals)
    min_rank_vals = np.maximum.accumulate(pixel_floors)
    min_rank_vals = np.minimum(min_rank_vals, sorted_raw)
    min_rank_vals = np.maximum.accumulate(min_rank_vals)

    # Step 4: Locate transition knee and calibration anchors
    knee_idx = np.argmax(min_rank_vals)
    x_knee = min_rank_vals[knee_idx]
    y_knee = (knee_idx + 1) / N

    x1, y1 = x_knee, np.clip(y_knee, 0.05, 0.95)
    x2, y2 = x_knee + delta_x_plateau, 0.99

    # Solve parameters k and x0 using logit transform
    logit_y1 = np.log(y1 / (1.0 - y1))
    logit_y2 = np.log(y2 / (1.0 - y2))

    k_sig = (logit_y2 - logit_y1) / (x2 - x1)
    x0_sig = x1 - (logit_y1 / k_sig)

    # Step 5: Inverse sigmoid evaluation
    y_ranks = np.linspace(0.001, 0.999, N)
    sigmoid_vals = x0_sig + (np.log(y_ranks / (1.0 - y_ranks)) / k_sig)

    # Enforce monotonicity and floor bounds
    final_sigmoid_vals = np.maximum(sigmoid_vals, min_rank_vals)
    final_sigmoid_vals = np.minimum(final_sigmoid_vals, sorted_raw)
    final_sigmoid_vals = np.maximum.accumulate(final_sigmoid_vals)

    # Step 6: Remap back to 2D matrices
    # Remap sigmoid correction
    I_out_sig = I_in.copy()
    remapped_sig = np.empty_like(final_sigmoid_vals)
    remapped_sig[sort_idx] = final_sigmoid_vals
    I_out_sig[seg_bool] = remapped_sig

    # Remap monotonic floor baseline
    I_out_mono = I_in.copy()
    remapped_mono = np.empty_like(min_rank_vals)
    remapped_mono[sort_idx] = min_rank_vals
    I_out_mono[seg_bool] = remapped_mono

    return I_out_sig, I_out_mono, outside_vals






def apply_monotonic_floor_ecdf(I_in, mask, dilation_iter=2, alpha_cutoff=0.8):
    """
    Applies distance-weighted ECDF matching combined with a monotonic boundary 
    floor projection to prevent edge inversions while strictly preserving internal ranks.

    Parameters:
        I_in : 2D ndarray, input spectrogram in log scale
        mask : 2D bool ndarray, target region to correct
        dilation_iter : int, border expansion steps to form reference ring
        alpha_cutoff : float, distance scaling factor for transition ring
    """
    if not np.any(mask):
        return I_in.copy(), np.array([])

    rows, cols = I_in.shape

    # 1. Reference ring extraction
    dilated = binary_dilation(mask, iterations=dilation_iter)
    border_ring = dilated & ~mask

    outside_vals = np.sort(I_in[border_ring])
    inside_vals = I_in[mask]
    N_in = len(inside_vals)
    N_out = len(outside_vals)

    # 2. Standard quantile matching
    q_in = np.linspace(0, 1, N_in)
    q_out = np.linspace(0, 1, N_out)
    ecdf_map = interp1d(
        np.sort(inside_vals),
        np.interp(q_in, q_out, outside_vals),
        bounds_error=False,
        fill_value='extrapolate'
    )
    I_ecdf_direct = ecdf_map(inside_vals)

    # 3. Distance transform alpha blending
    import scipy.ndimage as ndi
    dist = ndi.distance_transform_edt(mask)
    d_max = np.max(dist) if np.max(dist) > 0 else 1.0
    alpha_weight = np.clip(dist / (d_max * alpha_cutoff + 1e-6), 0.0, 1.0)
    blended_vals = (1.0 - alpha_weight[mask]) * inside_vals + alpha_weight[mask] * I_ecdf_direct

    # 4. Compute local boundary floor at seam
    # For every pixel inside mask, find adjacent exterior neighbors
    local_floor_map = np.full_like(I_in, -np.inf)

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in shifts:
        r_src = slice(max(0, -dr), min(rows, rows - dr))
        c_src = slice(max(0, -dc), min(cols, cols - dc))
        r_dst = slice(max(0, dr), min(rows, rows + dr))
        c_dst = slice(max(0, dc), min(cols, cols + dc))

        # Pixel p is inside, pixel q is outside
        p_in = mask[r_src, c_src]
        q_out = (~mask)[r_dst, c_dst]
        active = p_in & q_out

        if not np.any(active):
            continue

        val_p = I_in[r_src, c_src]
        val_q = I_in[r_dst, c_dst]

        # Floor is imposed by exterior neighbor if original inside >= exterior
        valid_pairs = active & (val_p >= val_q)
        
        current_sub = local_floor_map[r_src, c_src]
        current_sub[valid_pairs] = np.maximum(current_sub[valid_pairs], val_q[valid_pairs])
        local_floor_map[r_src, c_src] = current_sub

    inside_local_floors = local_floor_map[mask]

    # 5. Monotonic Cumulative Projection
    # Sort original inside pixels to establish the true rank order
    sort_order = np.argsort(inside_vals)
    sorted_blended = np.sort(blended_vals)

    # Align boundary floors to the sorted rank order
    sorted_floors = inside_local_floors[sort_order]

    # Monotonize the floor: a pixel with rank k cannot be lower than the floor of any rank j < k
    monotonic_floor = np.maximum.accumulate(sorted_floors)

    # Project sorted blended values above the monotonic floor
    projected_sorted = np.maximum(sorted_blended, monotonic_floor)

    # Optional upper clamp: ensure no pixel exceeds its original intensity
    sorted_orig = inside_vals[sort_order]
    projected_sorted = np.minimum(projected_sorted, sorted_orig)

    # 6. Reassign to spatial layout
    corrected_inside = np.empty_like(inside_vals)
    corrected_inside[sort_order] = projected_sorted

    I_out = I_in.copy()
    I_out[mask] = corrected_inside

    return I_out, outside_vals



def apply_spatially_conditioned_quantile_blend(I_in, mask, dilation_iter=2, alpha_cutoff=0.8):
    """
    Blends quantile distributions using spatial distance information aggregated 
    over rank levels, guaranteeing 100% internal rank preservation.
    """
    if not np.any(mask):
        return I_in.copy(), np.array([])

    # 1. Reference ring
    dilated = binary_dilation(mask, iterations=dilation_iter)
    border_ring = dilated & ~mask
    outside_vals = np.sort(I_in[border_ring])
    inside_vals = I_in[mask]

    N_in = len(inside_vals)
    N_out = len(outside_vals)

    # 2. Normalized distance transform
    dist = ndi.distance_transform_edt(mask)
    d_max = np.max(dist) if np.max(dist) > 0 else 1.0
    alpha_spatial = np.clip(dist / (d_max * alpha_cutoff + 1e-6), 0.0, 1.0)
    inside_dist_weights = alpha_spatial[mask]

    # 3. Sort inside pixels to establish rank order
    sort_order = np.argsort(inside_vals)
    inside_sorted = inside_vals[sort_order]

    # 4. Map outside distribution to quantile grid
    q_in = np.linspace(0, 1, N_in)
    q_out = np.linspace(0, 1, N_out)
    target_sorted = np.interp(q_in, q_out, outside_vals)

    # 5. Project spatial weights into quantile space
    # Re-order spatial distance weights by pixel rank
    sorted_spatial_weights = inside_dist_weights[sort_order]

    # Enforce non-decreasing alpha along rank axis to guarantee monotonicity
    alpha_quantile = np.maximum.accumulate(sorted_spatial_weights)

    # 6. Monotonic quantile blend
    blended_sorted = (1.0 - alpha_quantile) * inside_sorted + alpha_quantile * target_sorted

    # 7. Write back
    corrected_inside = np.empty_like(inside_vals)
    corrected_inside[sort_order] = blended_sorted

    I_out = I_in.copy()
    I_out[mask] = corrected_inside

    return I_out, outside_vals