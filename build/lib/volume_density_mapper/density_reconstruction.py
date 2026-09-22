import inspect
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
import constrained_diffusion as cdd


# --- Helper Functions for Padding ---

def _get_pad_info(shape, npad):
    """
    Calculate padding parameters to center data in a square array
    of size (npad * max_dim).
    """
    ny, nx = shape
    nm = max(ny, nx)
    target_size = int(nm * npad)

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

    padded_data = np.zeros((target_size, target_size), dtype=data.dtype)
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


def _cdd_accepts_keyword(name):
    """Return whether the installed CDD entry point accepts a keyword.

    ``None`` means that the signature could not be inspected reliably.
    """
    func = cdd.constrained_diffusion_decomposition
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None

    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    return name in parameters


def _run_cdd(data, use_GPU=False, **kwargs):
    """
    Run constrained-diffusion decomposition with backward-compatible GPU handling.

    CPU mode deliberately does not pass GPU-only keywords so older CDD releases
    keep working.  With current CDD releases, GPU mode enables ``use_gpu=True``
    and ``n_chunk=None`` so CDD can automatically plan VRAM-efficient tiling.
    If GPU support is unavailable or fails at runtime, decomposition is retried
    on CPU.
    """
    func = cdd.constrained_diffusion_decomposition

    if not use_GPU:
        return func(data, **kwargs)

    supports_use_gpu = _cdd_accepts_keyword("use_gpu")
    if supports_use_gpu is False:
        warnings.warn(
            "Installed constrained-diffusion does not support the 'use_gpu' "
            "argument. Falling back to CPU decomposition.",
            RuntimeWarning,
            stacklevel=2,
        )
        return func(data, **kwargs)

    gpu_kwargs = dict(kwargs)
    supports_n_chunk = _cdd_accepts_keyword("n_chunk")
    if supports_n_chunk is not False:
        # In CDD >= 1.2.5/1.2.6, n_chunk=None enables automatic GPU chunk
        # planning based on available accelerator memory.
        gpu_kwargs.setdefault("n_chunk", None)

    try:
        return func(
            data,
            use_gpu=True,
            **gpu_kwargs,
        )
    except TypeError as exc:
        message = str(exc)

        # If signature inspection was inconclusive, retry without the new tiling
        # keyword before giving up on GPU support.
        if "n_chunk" in gpu_kwargs and "n_chunk" in message:
            retry_kwargs = dict(gpu_kwargs)
            retry_kwargs.pop("n_chunk", None)
            try:
                return func(
                    data,
                    use_gpu=True,
                    **retry_kwargs,
                )
            except TypeError as retry_exc:
                if "use_gpu" not in str(retry_exc):
                    raise
            except Exception as retry_exc:
                warnings.warn(
                    f"GPU decomposition failed "
                    f"({type(retry_exc).__name__}: {retry_exc}). "
                    "Falling back to CPU decomposition.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return func(data, **kwargs)

        elif "use_gpu" not in message:
            raise

        warnings.warn(
            "Installed constrained-diffusion does not support the GPU keyword "
            "interface used here. Falling back to CPU decomposition.",
            RuntimeWarning,
            stacklevel=2,
        )
    except Exception as exc:
        # CUDA/MPS/runtime failures inside CDD should not make the package unusable.
        warnings.warn(
            f"GPU decomposition failed ({type(exc).__name__}: {exc}). "
            "Falling back to CPU decomposition.",
            RuntimeWarning,
            stacklevel=2,
        )

    return func(data, **kwargs)


def _to_numpy_array(array_like):
    """Convert NumPy/Torch-like array objects to a NumPy ndarray safely."""
    if isinstance(array_like, np.ndarray):
        return array_like

    value = array_like
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _load_cupy():
    """
    Import CuPy only when GPU reconstruction is actually requested.

    Keeping CuPy optional allows CPU-only users to import and use the package
    without installing a CUDA-specific CuPy build.
    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GPU reconstruction requires CuPy. Install a CuPy build matching "
            "your CUDA version, or call with use_GPU=False."
        ) from exc

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA-capable device is available to CuPy.")
    except Exception as exc:
        raise RuntimeError(f"CuPy/CUDA is not usable: {exc}") from exc

    return cp


def _reconstruct_cpu(
    decomp,
    scale_list,
    ny,
    nx,
    nz,
    dx,
    scale_fz,
    padding,
    y_start,
    x_start,
):
    """Accumulate the 3D reconstruction in CPU memory."""
    final_cube = np.zeros((nz, ny, nx), dtype=np.float32)

    for padded_layer, log_scale in zip(decomp, scale_list):
        scale = 2 ** log_scale

        if padding:
            valid_layer = _unpad_data(
                padded_layer,
                (ny, nx),
                y_start,
                x_start,
            )
        else:
            valid_layer = padded_layer

        # Make the CPU path robust to NumPy or Torch-like CDD outputs.
        valid_layer = _to_numpy_array(valid_layer)

        z_profile = generate_z_profile(nz, scale, scale_fz)
        density_layer = valid_layer / dx

        # Plane-by-plane accumulation avoids allocating a temporary 3D
        # broadcast array, keeping peak memory usage low.
        for z in range(nz):
            final_cube[z, :, :] += density_layer * z_profile[z]

    return final_cube


def _reconstruct_gpu(
    decomp,
    scale_list,
    ny,
    nx,
    nz,
    dx,
    scale_fz,
    padding,
    y_start,
    x_start,
):
    """Accumulate the 3D reconstruction with CuPy and return a NumPy array."""
    cp = _load_cupy()

    final_cube = None
    try:
        final_cube = cp.zeros((nz, ny, nx), dtype=cp.float32)

        for padded_layer, log_scale in zip(decomp, scale_list):
            scale = 2 ** log_scale

            if padding:
                valid_layer = _unpad_data(
                    padded_layer,
                    (ny, nx),
                    y_start,
                    x_start,
                )
            else:
                valid_layer = padded_layer

            # CDD currently returns host arrays for the public NumPy API, but
            # normalize Torch-like outputs defensively before crossing into CuPy.
            valid_layer = _to_numpy_array(valid_layer)

            z_profile = generate_z_profile(nz, scale, scale_fz)
            density_layer_gpu = cp.asarray(valid_layer, dtype=cp.float32) / dx
            z_profile_gpu = cp.asarray(z_profile, dtype=cp.float32)

            # Avoid a full (nz, ny, nx) temporary broadcast allocation.
            for z in range(nz):
                final_cube[z, :, :] += density_layer_gpu * z_profile_gpu[z]

            del density_layer_gpu, z_profile_gpu

        return cp.asnumpy(final_cube)
    finally:
        # Release CuPy's cached allocations once the whole reconstruction ends.
        # Do not flush the pools once per scale; doing so defeats allocator reuse.
        if final_cube is not None:
            del final_cube
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


# --- Main Physics Functions ---

def compute_characteristic_scale(
    input_map,
    dx=1,
    padding=True,
    npad=2,
    decomposition_map_n=None,
    verbose=False,
    use_GPU=False,
):
    """
    Calculate the characteristic scales of the input map using Constrained
    Diffusion Decomposition.

    Args:
        input_map (ndarray): 2D array, the input map.
        dx (float): Pixel size.
        padding (bool): If True, pad input to square * npad.
        npad (int): Padding factor (result size = npad * max(nx, ny)).
        decomposition_map_n (int or None): If set, only use the first N scales.
        verbose (bool): If True, print details.
        use_GPU (bool): If True, request GPU acceleration from CDD when supported.

    Returns:
        width_map (ndarray): 2D array of characteristic scales (same shape as input).
    """
    input_map = np.asarray(input_map)
    if input_map.ndim != 2:
        raise ValueError("input_map must be a 2D array")

    if use_GPU:
        # CDD's GPU path is optimized for float32 and its published GPU examples
        # use float32 inputs. This also halves VRAM use compared with float64.
        input_map = input_map.astype(np.float32, copy=False)

    ny, nx = input_map.shape

    if padding:
        data_to_process, y_start, x_start = _pad_data(input_map, npad)
        if verbose:
            print(
                f"Padding enabled: Input ({ny}x{nx}) -> Padded "
                f"({data_to_process.shape[0]}x{data_to_process.shape[1]})"
            )
    else:
        data_to_process = input_map

    result, _, sc = _run_cdd(
        np.nan_to_num(data_to_process),
        use_GPU=use_GPU,
        up_sample=False,
        return_scales=True,
        log_scale_base=np.sqrt(2),
        mode="log",
    )

    # Normalize CDD outputs in case a backend returns Torch-like arrays.
    result = np.asarray([_to_numpy_array(layer) for layer in result])
    sc = _to_numpy_array(sc)
    result, sc = _slice_decomposition(result, sc, decomposition_map_n, verbose)

    if len(sc) == 0:
        raise ValueError("CDD returned no decomposition scales")

    scale_list = np.log2(sc)

    if verbose:
        print(
            f"Using {len(sc)} scales ranging from {sc[0]:.2f} "
            f"to {sc[-1]:.2f} pixels."
        )

    total_weight = np.sum(result, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_log_scale = (
            np.sum(result * scale_list[:, np.newaxis, np.newaxis], axis=0)
            / total_weight
        )
        width_map = (2 ** avg_log_scale) * dx

    width_map = np.nan_to_num(
        width_map,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if padding:
        width_map = _unpad_data(width_map, (ny, nx), y_start, x_start)

    width_map[np.isnan(input_map)] = np.nan
    return width_map


def compute_mean_density_width(
    column_density,
    dx,
    padding=True,
    npad=2,
    decomposition_map_n=None,
    verbose=False,
    use_GPU=False,
):
    """
    Calculate the mean volume density from column density.

    Args:
        column_density (ndarray): 2D array (g cm^-2).
        dx (float): Pixel size in cm.
        padding (bool): If True, pad input.
        npad (int): Padding factor.
        decomposition_map_n (int or None): If set, only use the first N scales.
        verbose (bool): If True, print details.
        use_GPU (bool): If True, request GPU acceleration from CDD when supported.

    Returns:
        tuple: (mean_density, width)
    """
    column_density = np.asarray(column_density)
    if column_density.ndim != 2:
        raise ValueError("column_density must be a 2D array")

    if use_GPU:
        column_density = column_density.astype(np.float32, copy=False)

    width = compute_characteristic_scale(
        column_density,
        dx,
        padding=padding,
        npad=npad,
        decomposition_map_n=decomposition_map_n,
        verbose=verbose,
        use_GPU=use_GPU,
    )

    thickness = width / np.sqrt(8 * np.log(2)) * (2 * np.sqrt(np.pi))

    with np.errstate(divide="ignore", invalid="ignore"):
        density = column_density / thickness

    density = np.nan_to_num(
        density,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

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
        profile = gaussian_filter(
            profile,
            sigma=sigma,
            mode="constant",
            cval=0.0,
        )

    total = profile.sum()
    if total > 0:
        profile /= total

    return profile


def density_reconstruction_3d(
    data_in,
    dx,
    scale_fz=1.0,
    padding=True,
    npad=2,
    decomposition_map_n=None,
    verbose=False,
    use_GPU=False,
):
    """
    Reconstruct 3D volume density from a 2D column-density map.

    Args:
        data_in (ndarray): 2D column density map.
        dx (float): Pixel size in cm.
        scale_fz (float): Anisotropy factor (Z-stretch).
        padding (bool): If True, pad input to reduce edge effects.
        npad (int): Padding factor used by the CDD step.
        decomposition_map_n (int or None): If set, only process the first N scales.
        verbose (bool): If True, print decomposition details.
        use_GPU (bool): If True, request GPU CDD and CuPy reconstruction. If the
                        GPU reconstruction is unavailable or fails, reconstruction
                        falls back to CPU with a warning.

    Returns:
        ndarray: 3D density reconstruction (g cm^-3) with shape
                 (max(ny, nx), ny, nx).
    """
    if not isinstance(data_in, np.ndarray) or data_in.ndim != 2:
        raise ValueError("Input data_in must be a 2D numpy array")

    if use_GPU:
        data_in = data_in.astype(np.float32, copy=False)

    ny, nx = data_in.shape
    # Keep the public output shape stable regardless of decomposition_map_n.
    nz = max(ny, nx)

    if padding:
        data_to_process, y_start, x_start = _pad_data(data_in, npad)
        if verbose:
            print(
                f"Padding enabled: Input ({ny}x{nx}) -> Padded "
                f"({data_to_process.shape[0]}x{data_to_process.shape[1]})"
            )
    else:
        data_to_process = data_in
        y_start = x_start = 0

    decomp, _, sc = _run_cdd(
        np.nan_to_num(data_to_process),
        use_GPU=use_GPU,
        up_sample=False,
        return_scales=True,
        log_scale_base=np.sqrt(2),
    )

    decomp = list(decomp)
    sc = _to_numpy_array(sc)
    decomp, sc = _slice_decomposition(
        decomp,
        sc,
        decomposition_map_n,
        verbose,
    )

    if len(sc) == 0:
        raise ValueError("CDD returned no decomposition scales")

    scale_list = np.log2(sc)

    if verbose:
        print(f"\n--- Decomposition Levels (Total: {len(sc)}) ---")
        for i, s in enumerate(sc):
            print(f"Level {i + 1}: Scale = {s:.2f} pix")
        print("------------------------------------------")

    if use_GPU:
        try:
            if verbose:
                print(
                    f"Reconstructing output cube: {nx}x{ny}x{nz} (GPU: True)"
                )
            final_cube = _reconstruct_gpu(
                decomp,
                scale_list,
                ny,
                nx,
                nz,
                dx,
                scale_fz,
                padding,
                y_start,
                x_start,
            )
        except Exception as exc:
            warnings.warn(
                f"GPU reconstruction failed ({type(exc).__name__}: {exc}). "
                "Falling back to CPU reconstruction.",
                RuntimeWarning,
                stacklevel=2,
            )
            if verbose:
                print(
                    f"Reconstructing output cube: {nx}x{ny}x{nz} (GPU: False)"
                )
            final_cube = _reconstruct_cpu(
                decomp,
                scale_list,
                ny,
                nx,
                nz,
                dx,
                scale_fz,
                padding,
                y_start,
                x_start,
            )
    else:
        if verbose:
            print(f"Reconstructing output cube: {nx}x{ny}x{nz} (GPU: False)")
        final_cube = _reconstruct_cpu(
            decomp,
            scale_list,
            ny,
            nx,
            nz,
            dx,
            scale_fz,
            padding,
            y_start,
            x_start,
        )

    if not np.isfinite(final_cube).all():
        warnings.warn(
            "Non-finite values detected in the reconstructed cube. "
            "Replaced with zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        final_cube = np.nan_to_num(
            final_cube,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return final_cube
