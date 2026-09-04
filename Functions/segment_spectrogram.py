'''
File for function to segment high intensity blobs in spectrogram using first a high intenisty bins mask
in a constrained frequency interval and a watershed routine with a second background mask for refinement.
'''

import numpy as np
from scipy.ndimage import label
from skimage.segmentation import watershed
from scipy.linalg import solve

def whittaker_als_baseline(y, lam=1e4, p=0.01, max_iter=15):
    L = len(y)
    D = np.zeros((L-2, L))
    for i in range(L-2):
        D[i, i], D[i, i+1], D[i, i+2] = 1, -2, 1
    penalty = lam * np.dot(D.T, D)
    w = np.ones(L)
    for _ in range(max_iter):
        W = np.diag(w)
        z = solve(W + penalty, w * y)
        w = np.where(y > z, p, 1 - p)
    return z



def segment_blobs(M, f_M, f_int, factor_high=3, factor_low=2, average='mean', lam=1e2, p=0.01, max_iter=15, method = 'baseline'):
    '''
    Inputs:
    - M: spectrogram matrix
    - f_M: frequency vector fro matrix M
    - f_int: frequency interval in which the signature is mostly contained
    - factor_high: multiplicative factor for high bins intensity
    - factor_low: multiplicative factor for background bins intensity
    - average: str --> whether mean or median is used to do the psd projection
    '''
    # --- get PSD
    if average == 'mean':
        psd = np.mean(M, axis = 1) 
    elif average == 'median':
        psd = np.median(M, axis = 1) 
    else:
        print('average mode is incorrect: select mean or median')

    # --- Get baseline of PSD
    # NOTE: should replace parameters with the spectrogram resolution
    psd_baseline = whittaker_als_baseline(psd, lam, p, max_iter)

    # --- Get threshold above which blobs are considered high and under which pixels are considered to be background
    if method == 'baseline':
        T_high = factor_high * psd_baseline  # Using your defined factor
        T_low = factor_low * psd_baseline 
    elif method == 'quantile':
        med  = np.quantile(M, 0.5)
        T_high = factor_high * med
        T_low = factor_low * med
    else: 
        print('method name is incorrect: --> select baseline or quantile')

    # get mask of high intensity bins
    mask = M > (T_high[:, np.newaxis]) 

    # --- refine mask to region within f_int
    freq_mask = (f_M >= f_int[0]) & (f_M <= f_int[-1])
    mask_f_int = mask * freq_mask[:, None]

    # --- Watershed extension
    # A: Label distinct seed regions inside f_int | Structure (3x3 array of 1s) enables 8-connectivity labeling
    structure = np.ones((3, 3), dtype=int)
    markers, num_features = label(mask_f_int, structure=structure)
    # B: Define Topography (Invert M so high intensity peaks = deep valleys)
    topography = -M
    # C: Define a Background Mask / Stop Boundary | Pixels below this noise floor are treated as "background" (label = 0)
    background_mask = M < (T_low[:, np.newaxis]) 
    # D: Run Watershed | Compactness controls boundary smoothness (0 = standard intensity-driven flooding)
    watershed_markers = markers.copy()
    watershed_markers[background_mask] = -1  # Boundary where flooding stops
    labels = watershed(topography, markers=watershed_markers, mask=~background_mask)
    # E: Convert watershed labels into a binary mask (any pixel assigned a peak label > 0)
    mask_watershed = (labels > 0).astype(int)

    return mask_watershed, psd, psd_baseline, T_low, T_high