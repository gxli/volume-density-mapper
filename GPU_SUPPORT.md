# Experimental GPU support

This development branch keeps the existing CPU API and adds optional GPU
acceleration to constrained-diffusion decomposition and 3-D reconstruction.

## Installation

CPU-only installation remains unchanged:

```bash
pip install -e .
```

For NVIDIA/CUDA systems, install the GPU extra:

```bash
pip install -e ".[gpu]"
```

The `gpu` extra currently targets CUDA 12.x through CuPy. If your CUDA
installation requires another CuPy wheel, install the matching CuPy package
manually instead.

## Usage

```python
from volume_density_mapper import density_reconstruction_3d

rho = density_reconstruction_3d(
    column_density,
    dx,
    use_GPU=True,
    verbose=True,
)
```

`use_GPU=True` requests the GPU backend in constrained-diffusion decomposition.
On current CDD releases the wrapper also requests automatic GPU chunk planning.
The reconstruction stage uses CuPy on CUDA. If CDD GPU execution or CuPy
reconstruction is unavailable, the implementation warns and falls back to CPU.

The public output shape remains `(max(ny, nx), ny, nx)`.

## Validation before merging to main

Before merging this branch, run on a real CUDA host:

1. CPU versus GPU numerical-consistency tests on representative maps.
2. Runtime and peak VRAM measurements.
3. At least one large padded reconstruction.
4. CPU-only installation/import tests in an environment without CuPy.

The existing physical choice `up_sample=False` is intentionally preserved so
GPU enablement does not change the reconstruction algorithm.
