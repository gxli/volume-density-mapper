#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# --- 1. Gracefully handle PyVista import ---
try:
    import pyvista as pv
    pyvista_installed = True
except ImportError:
    pyvista_installed = False
    print("---")
    print("Warning: PyVista is not installed. The 3D plot will be skipped.")
    print("To install it, run: pip install pyvista pyqt5")
    print("---")


# --- 2. Ensure the custom module can be found ---
print("Attempting to import `volume_density_mapper` from ../src...")
try:
    current_dir = os.getcwd()
    src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from density_reconstruction import *
    print("...Import successful!")
except (ImportError, ModuleNotFoundError) as e:
    print(f"...Failed to import `volume_density_mapper`: {e}")
    sys.exit("Please ensure 'volume_density_mapper.py' is in the '../src' directory.")
except FileNotFoundError:
    print("...Could not find 'IC348_nh.fits'. Make sure it is in the same directory.")
    sys.exit()


# --- 3. Load Data and Reconstruct to 3D ---
print("Loading and reconstructing 3D data...")
nh = fits.getdata('IC348_nh.fits')
header = fits.getheader('IC348_nh.fits')

mh2 = 1.34 * 3.34e-24
pc = 3.08e18

data_in = nh * mh2
dx_cm = header['CDELT2'] / 180 * np.pi * 270 * pc
data3d = density_reconstruction_3d(data_in, dx_cm)
density, width = compute_mean_density_width(data_in, dx_cm)
print(f"Shape of the reconstructed 3D data: {np.shape(data3d)}")


# --- 4. Generate Original 2D Matplotlib Plots ---
print("Generating 2D diagnostic plots...")

# Plot 1: Surface Density
plt.figure(dpi=100)
plt.imshow(np.log10(data_in), origin='lower')
plt.colorbar(label=r'Log(surface density ($\rm g cm^{-2}$))')
plt.title('2D Surface Density')

# Plot 2: Mean Volume Density
plt.figure(dpi=100)
plt.imshow(np.log10(density), origin='lower')
plt.colorbar(label=r'log(Volume Density ($g\;cm^{-3}$))')
plt.title('Mean Volume Density')

# Plot 3: Characteristic Width
plt.figure(dpi=100)
plt.imshow(np.log10(width), origin='lower', cmap='magma')
plt.colorbar(label=r'log(width (cm))')
plt.title('Characteristic Width')
plt.show()

# --- 5. Render the 3D Data with PyVista (if installed) ---
if pyvista_installed:
    print("Rendering 3D volume plot with PyVista...")

    # Take a central slice for visualization
    center_idx = data3d.shape[0] // 2
    slice_width = 30
    data_slice = data3d[center_idx - slice_width : center_idx + slice_width, :, :]

    # Wrap the NumPy array in a PyVista UniformGrid object
    grid = pv.wrap(data_slice)
    grid['Density'] = data_slice.flatten(order='F')

    # --- Custom Contrast Control ---
    # Define the data range for the opacity map to focus on the structure
    non_zero_values = data_slice[data_slice > 0]
    scalar_limits = np.percentile(non_zero_values, [20, 99.5])

    # Create a custom opacity transfer function for better visibility
    opacity_map = np.linspace(0, 0.7, 5)

    # --- Create the plot ---
    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        cmap='plasma',
        opacity=opacity_map,      # Use our custom map
        clim=scalar_limits,       # Apply map over our chosen data range
        scalar_bar_args={}        # No title on the color bar
    )

    # Display the interactive rendering window (this is a blocking call)
    print("Showing interactive PyVista plot window... Close the window to continue.")
    plotter.show()
else:
    print("Skipping 3D plot because PyVista is not installed.")


# --- 6. Display all generated Matplotlib plots ---
print("Showing Matplotlib plots...")

plt.show()
print("\nScript finished.")