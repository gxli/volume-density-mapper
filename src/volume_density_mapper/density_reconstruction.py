import numpy as np
import math
from scipy.ndimage import gaussian_filter
import constrained_diffusion as cdd
import warnings

# --- Helper Functions for Padding ---

def _get_pad_info(shape, npad):
    """
    Calculate padding parameters to center data in a square array 
    of size (npad * max_dim).
    """
    ny, nx = shape
    nm = max(ny, nx)
    target_size = int(nm * npad)
    
    # Calculate start indices to center the image
    y_start = (target_size - ny) // 2
    x_start = (target_size - nx) // 2
    
    return target_size, y_start, x_start

def _pad_data(data, npad):
    """
    Pad 2D data to a square size of (npad * max_dim).
    """
    if npad < 1:
        return data, 0, 0
        
    ny, nx = data.shape
    target_size, y_start, x_start = _get_pad_info((ny, nx), npad)
    
    # Create zero-filled padded array
    padded_data = np.zeros((target_size, target_size), dtype=data.dtype)
    
    # Insert data in center
    padded_data[y_start : y_start + ny, x_start : x_start + nx] = data
    
    return padded_data, y_start, x_start

def _unpad_data(data, original_shape, y_start, x_start):
    """
    Crop data back to original dimensions.
    """
    ny, nx = original_shape
    return data[y_start : y_start + ny, x_start : x_start + nx]

def _slice_decomposition(result, sc, n, verbose=False):
    """
    Helper to slice decomposition results if a limit is set.
    """
    if n is not None and isinstance(n, int) and n < len(sc):
        if verbose:
            print(f"Limiting processing to first {n} scales (out of {len(sc)}).")
        return result[:n], sc[:n]
    return result, sc

# --- Main Physics Functions ---

def compute_characteristic_scale(input_map, dx=1, padding=True, npad=2, decomposition_map_n=None, verbose=False):
    """
    Calculate the characteristic scales of the input map using Constrained Diffusion Decomposition.

    Args:
        input_map (ndarray): 2D array, the input map.
        dx (float): Pixel size.
        padding (bool): If True, pad input to square * npad.
        npad (int): Padding factor (result size = npad * max(nx, ny)).
        decomposition_map_n (int or None): If set, only use the first N scales. 
                                           Useful for saving computation.
        verbose (bool): If True, print details.

    Returns:
        width_map (ndarray): 2D array of characteristic scales (same shape as input).
    """
    ny, nx = input_map.shape
    
    # Handle Padding
    if padding:
        data_to_process, y_start, x_start = _pad_data(input_map, npad)
        if verbose:
            print(f"Padding enabled: Input ({ny}x{nx}) -> Padded ({data_to_process.shape[0]}x{data_to_process.shape[1]})")
    else:
        data_to_process = input_map

    # Run Decomposition
    result, residual, sc = cdd.constrained_diffusion_decomposition(
        np.nan_to_num(data_to_process),
        up_sample=False,
        return_scales=True,
        log_scale_base=np.sqrt(2)
    )
    
    # Apply Limit
    result = np.array(result)
    result, sc = _slice_decomposition(result, sc, decomposition_map_n, verbose)
    
    scale_list = np.log2(sc)
    
    if verbose:
        print(f"Using {len(sc)} scales ranging from {sc[0]:.2f} to {sc[-1]:.2f} pixels.")

    # Calculate weighted average scale
    total_weight = np.sum(result, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_log_scale = np.sum(result * scale_list[:, np.newaxis, np.newaxis], axis=0) / total_weight
        width_map = (2**avg_log_scale) * dx

    # Handle NaNs/Zeros
    width_map = np.nan_to_num(width_map, nan=0.0)

    # Un-pad if necessary
    if padding:
        width_map = _unpad_data(width_map, (ny, nx), y_start, x_start)
        # Restore original NaNs from input if they existed
        width_map[np.isnan(input_map)] = np.nan
    else:
        width_map[np.isnan(input_map)] = np.nan
        
    return width_map

def compute_mean_density_width(column_density, dx, padding=True, npad=2, decomposition_map_n=None, verbose=False):
    """
    Calculate the mean volume density from column density.

    Args:
        column_density (ndarray): 2D array (g cm^-2).
        dx (float): Pixel size in cm.
        padding (bool): If True, pad input.
        npad (int): Padding factor.
        decomposition_map_n (int or None): If set, only use the first N scales.
        verbose (bool): If True, print details.

    Returns:
        tuple: (mean_density, width)
    """
    width = compute_characteristic_scale(
        column_density, 
        dx, 
        padding=padding, 
        npad=npad, 
        decomposition_map_n=decomposition_map_n, 
        verbose=verbose
    )
    
    # Convert characteristic FWHM-like width to equivalent linear thickness
    thickness = width / np.sqrt(8 * np.log(2)) * (2 * np.sqrt(np.pi))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density = column_density / thickness
        
    density = np.nan_to_num(density, nan=0.0)
    
    return density, width

def generate_z_profile(nz, scale, scale_fz=1.0):
    """
    Generate a normalized 1D Gaussian profile for the Z-axis.
    """
    profile = np.zeros(nz)
    mid = nz // 2
    profile[mid] = 1.0
    
    sigma = (scale * scale_fz) / np.sqrt(2 * np.log(2))
    
    if sigma > 0:
        profile = gaussian_filter(profile, sigma=sigma, mode='constant', cval=0.0)
    
    total = profile.sum()
    if total > 0:
        profile /= total
    
    return profile

def density_reconstruction_3d(data_in, dx, scale_fz=1.0, padding=True, npad=2, decomposition_map_n=None, verbose=False):
    """
    Reconstruct 3D volume density from 2D column density.

    Args:
        data_in (ndarray): 2D column density map.
        dx (float): Pixel size in cm.
        scale_fz (float): Anisotropy factor (Z-stretch).
        padding (bool): If True, pad input to reduce edge effects.
        npad (int): Padding factor. Result will be padded to size npad*max(nx,ny).
        decomposition_map_n (int or None): If set, only process the first N scales.
                                           Reduces loop iterations and memory usage.
        verbose (bool): Explicitly print decomposition levels.

    Returns:
        ndarray: 3D density reconstruction (g cm^-3) with shape (max(ny,nx), ny, nx).
    """
    if not isinstance(data_in, np.ndarray) or data_in.ndim != 2:
        raise ValueError("Input data_in must be a 2D numpy array")
        
    ny, nx = data_in.shape
    # The physical depth of the object is determined by its largest physical dimension
    nz = max(ny, nx) 
    
    # 1. Handle Padding
    if padding:
        data_to_process, y_start, x_start = _pad_data(data_in, npad)
        if verbose:
            print(f"Padding enabled: Input ({ny}x{nx}) -> Padded ({data_to_process.shape[0]}x{data_to_process.shape[1]})")
    else:
        data_to_process = data_in

    # 2. Perform Decomposition
    decomp, _, sc = cdd.constrained_diffusion_decomposition(
        np.nan_to_num(data_to_process),
        up_sample=False,
        return_scales=True,
        log_scale_base=np.sqrt(2)
    )
    
    # Apply Limit
    decomp = list(decomp) # Ensure it's list-like for slicing
    decomp, sc = _slice_decomposition(decomp, sc, decomposition_map_n, verbose)
    
    scale_list = np.log2(sc)

    if verbose:
        print(f"\n--- Decomposition Levels (Total: {len(sc)}) ---")
        for i, s in enumerate(sc):
            print(f"Level {i+1}: Scale = {s:.2f} pix")
        print("------------------------------------------")

    # 3. Allocate Output Cube (Original dimensions)
    # We only reconstruct the 'valid' region to save memory, even if CDD used padding
    final_cube = np.zeros((nz, ny, nx), dtype=np.float32)

    if verbose:
        print(f"Reconstructing output cube: {nx}x{ny}x{nz}")

    # 4. Iterate over layers
    for i, (padded_layer, log_scale) in enumerate(zip(decomp, scale_list)):
        scale = 2**log_scale
        
        # If padding was used, crop the layer back to original size BEFORE 3D projection
        # This ensures we don't waste memory on the padded void
        if padding:
            valid_layer = _unpad_data(padded_layer, (ny, nx), y_start, x_start)
        else:
            valid_layer = padded_layer

        # Generate Z-profile (scales with anisotropy)
        z_profile = generate_z_profile(nz, scale, scale_fz)
        
        # Calculate volume density for this layer
        # Density = Column / dx * Profile
        density_layer = valid_layer / dx 
        
        # Accumulate
        final_cube += density_layer[np.newaxis, :, :] * z_profile[:, np.newaxis, np.newaxis]

    # 5. Final Checks
    if np.isnan(final_cube).any():
        warnings.warn("NaNs detected in the reconstructed cube. Replaced with zeros.")
        final_cube = np.nan_to_num(final_cube)

    return final_cube