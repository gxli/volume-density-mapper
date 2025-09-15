from astropy.io import fits
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import constrained_diffusion as cdd

def prepare_cube(data):
    """Prepare a 3D data cube from the input data.

    Args:
        data (ndarray): 2D input data array.

    Returns:
        cube (ndarray): 3D data cube with zeros, with depth equal to max dimension of input.
    """
    ny, nx = data.shape
    nz = max(ny, nx)
    cube = np.zeros((nz, ny, nx))
    return cube

def compute_characterstic_scale(input_map, dx=1):
    '''Calculate the characteristic scales of the input map.
    
    Args:
        input_map: 2D array, the input map
        dx: float, the pixel size, default is 1, the output will be width
        measured in the number of pixels.
    
    Returns:
        width_map: 2D array of characteristic scales in the same unit as dx
    '''
    result, residual = cdd.constrained_diffusion_decomposition(np.nan_to_num(input_map))
    result = np.array(result)
    nz, ny, nx = result.shape

    scale_list = 2 ** np.arange(nz)  # Exponential scale progression
    total_weight = np.sum(result, axis=0)
    total_weight[total_weight == 0] = np.nan  # Avoid division by zero
    
    width_map = np.sum(result * scale_list[:, np.newaxis, np.newaxis], axis=0) / total_weight * dx
    width_map[np.isnan(input_map)] = np.nan
    return width_map

def compute_mean_density_width(column_density, dx):
    '''Calculate the mean density from column density.
    
    Args:
        column_density: 2D array, the column density map (g cm^-2)
        dx: float, the pixel size in cm
        
    Returns:
        mean_density: 2D array, the mean density map (g cm^-3)
        width: 2D array, characteristic scales in cm   
    '''
    width = compute_characterstic_scale(column_density, dx)
    # Convert FWHM to equivalent thickness
    thickness = width / np.sqrt(8*np.log(2)) * (2*np.sqrt(np.pi))
    density =  column_density / thickness
    return density, width

def process_one_channel(data, scale, dx):
    """
    Process a single scale channel to create a 3D density reconstruction.
    
    Args:
        data: 2D array, input data for this scale
        scale: float, characteristic scale for this channel
        dx: float, pixel size
        
    Returns:
        3D density reconstruction for this channel
    """
    thickness = max(data.shape)
    total_mass = data.sum() * dx**2
    
    # Create 3D cube with 2D data repeated
    tcube = np.array([data] * thickness)
    
    # Create Gaussian profile along z-axis
    z_profile = np.zeros(thickness)
    z_profile[thickness // 2] = 1  # Center peak
    z_profile = gaussian_filter(z_profile, scale/np.sqrt(2*np.log(2)))
    
    # Apply profile to cube
    tcube = tcube * z_profile[:, np.newaxis, np.newaxis]
    
    # Normalize to conserve mass
    tcube = tcube / tcube.sum() * total_mass / dx**3
    
    return tcube

def decomposition_to_cube(decomposition, dx):
    """Convert decomposition result into a 3D data cube.
    
    Args:
        decomposition: decomposition result from constrained_diffusion_decomposition
        dx: pixel size
        
    Returns:
        3D density reconstruction cube
    """
    data_3d = None
    
    for i, (channel, scale) in enumerate(zip(decomposition, 2**np.arange(len(decomposition)))):
        tcube = process_one_channel(channel, scale, dx)
        
        if data_3d is None:
            data_3d = tcube
        else:
            data_3d += tcube
            
    return data_3d

def density_reconstruction_3d(data_in, dx):
    '''
    Reconstruct 3D density from 2D column density.
    
    Args:
        data_in: column density (g cm^-2)
        dx: pixel size in cm
        
    Returns:
        3D density reconstruction (g cm^-3)
    '''
    decomp, residual = cdd.constrained_diffusion_decomposition(np.nan_to_num(data_in))
    data_3d = decomposition_to_cube(decomp, dx)
    return data_3d
