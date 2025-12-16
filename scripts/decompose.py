#!/usr/bin/env python
import argparse
import sys
import os
import time
import numpy as np
from astropy.io import fits

# --- Import Logic: Try Direct, Fallback to ../src ---
try:
    # 1. Try importing directly (assuming installed via pip or in PYTHONPATH)
    from volume_density_mapper.density_reconstruction import (
        density_reconstruction_3d,
        compute_characteristic_scale
    )
except (ImportError, ModuleNotFoundError):
    # 2. If failed, look in ../src relative to this script
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
        
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        from volume_density_mapper.density_reconstruction import (
            density_reconstruction_3d,
            compute_characteristic_scale
        )
    except ImportError as e:
        print("\nCRITICAL ERROR: Could not import backend modules.")
        print(f"Details: {e}")
        print("\nPlease ensure 'volume_density_mapper' is installed OR located in:")
        print("  ../src/volume_density_mapper/")
        sys.exit(1)

def print_example_usage():
    """Prints example commands to help the user."""
    print("\n" + "="*50)
    print("EXAMPLE USAGE:")
    print("="*50)
    print("1. Basic (Automatic output name):")
    print("   python reconstruct_cube.py data/IC348_nh.fits")
    print("\n2. Specify physical pixel size (dx) and verbose mode:")
    print("   python reconstruct_cube.py data/IC348_nh.fits --dx 3.08e18 -v")
    print("\n3. Fast approximation (first 5 scales) with custom output name:")
    print("   python reconstruct_cube.py data/IC348_nh.fits -o my_cube.fits -n 5")
    print("\n4. Save both Density Cube and Characteristic Width Map:")
    print("   python reconstruct_cube.py data/IC348_nh.fits --save-width")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a 3D Density Cube from a 2D Column Density Map (FITS)."
    )

    # Required Argument
    parser.add_argument("input", help="Path to input 2D FITS file (Column Density)")

    # Optional Arguments
    parser.add_argument("-o", "--output", default=None,
                        help="Path to output 3D FITS file. If omitted, defaults to input_filename_3d_density.fits")

    parser.add_argument("--dx", type=float, default=1.0, 
                        help="Pixel size in cm (or physical units). Default: 1.0")
    
    parser.add_argument("--fz", type=float, default=1.0, 
                        help="Anisotropy factor (Z-axis stretch). Default: 1.0")
    
    parser.add_argument("-n", "--nscales", type=int, default=None,
                        help="Limit decomposition to the first N scales (saves RAM/Time).")
    
    parser.add_argument("--no-padding", action="store_true", 
                        help="Disable padding (not recommended, increases edge effects).")
    
    parser.add_argument("--pad-factor", type=int, default=2, 
                        help="Padding factor relative to max dimension. Default: 2")
    
    parser.add_argument("--save-width", action="store_true",
                        help="Also save the characteristic width map.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed progress.")

    args = parser.parse_args()

    # --- 1. Validate Input and Determine Output Name ---
    if not os.path.exists(args.input):
        print(f"\nERROR: Input file not found: '{args.input}'")
        print_example_usage()
        sys.exit(1)

    # Auto-generate output name if not provided
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_3d_density{ext}"
        if args.verbose:
            print(f"No output filename provided. Using default: {args.output}")

    # --- 2. Load Data ---
    if args.verbose:
        print(f"Loading input: {args.input}")
    
    try:
        data, header = fits.getdata(args.input, header=True)
    except Exception as e:
        print(f"\nERROR: Failed to read FITS file '{args.input}'.")
        print(f"Details: {e}")
        print_example_usage()
        sys.exit(1)

    # Cleanup dimensions
    data = data.squeeze()
    if data.ndim != 2:
        print(f"\nERROR: Input data must be 2D. Found {data.ndim} dimensions.")
        print("This tool requires a 2D column density map.")
        print_example_usage()
        sys.exit(1)

    # --- 3. Run Reconstruction ---
    start_time = time.time()
    if args.verbose:
        print(f"Starting reconstruction...")
        print(f"  dx: {args.dx}")
        print(f"  Padding: {'Disabled' if args.no_padding else f'Enabled (factor={args.pad_factor})'}")
        print(f"  Scales limit: {args.nscales if args.nscales else 'All'}")

    try:
        cube_3d = density_reconstruction_3d(
            data_in=data,
            dx=args.dx,
            scale_fz=args.fz,
            padding=(not args.no_padding),
            npad=args.pad_factor,
            decomposition_map_n=args.nscales,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\nERROR during reconstruction: {e}")
        if "constrained_diffusion" in str(e):
            print("Hint: This looks like an issue with the Constrained Diffusion Decomposition library.")
        print_example_usage()
        sys.exit(1)

    elapsed = time.time() - start_time
    if args.verbose:
        print(f"Reconstruction complete in {elapsed:.2f} seconds.")
        print(f"Output shape: {cube_3d.shape}")

    # --- 4. Save Output ---
    header['NAXIS'] = 3
    header['NAXIS3'] = cube_3d.shape[0]
    header['HISTORY'] = "3D Density reconstructed from Column Density"
    header['HISTORY'] = f"Params: dx={args.dx}, fz={args.fz}, padding={not args.no_padding}"
    
    if args.verbose:
        print(f"Saving 3D Cube to: {args.output}")
    
    try:
        fits.writeto(args.output, cube_3d, header, overwrite=True)
    except Exception as e:
        print(f"\nERROR: Could not write output file '{args.output}'.")
        print(f"Details: {e}")
        sys.exit(1)

    # --- 5. Optional: Save Width Map ---
    if args.save_width:
        base, ext = os.path.splitext(args.output)
        if "_3d_density" in base:
            width_out = base.replace("_3d_density", "_characteristic_width") + ext
        else:
            width_out = base + "_width" + ext

        if args.verbose:
            print(f"Calculating and saving width map to: {width_out}")
        
        try:
            width_map = compute_characteristic_scale(
                data, 
                dx=args.dx,
                padding=(not args.no_padding),
                npad=args.pad_factor,
                decomposition_map_n=args.nscales
            )
            header_w = header.copy()
            header_w['NAXIS'] = 2
            if 'NAXIS3' in header_w: del header_w['NAXIS3']
            
            fits.writeto(width_out, width_map, header_w, overwrite=True)
        except Exception as e:
            print(f"Warning: Failed to save width map. Details: {e}")

    if args.verbose:
        print("Done.")

if __name__ == "__main__":
    main()