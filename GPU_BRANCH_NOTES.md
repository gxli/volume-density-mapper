# gpu-support branch notes

Development version: `0.2.0.dev0`.

This branch is intentionally not a main-branch release. It integrates the
validated GPU-capable `density_reconstruction.py` while preserving CPU behavior.

Key integration choices:
- CuPy is imported lazily; CPU users do not need CuPy.
- The base CDD requirement remains compatible with existing CPU installs.
- The `gpu` extra selects a modern CDD GPU extra and CUDA CuPy.
- Current CDD GPU calls use automatic chunk planning when supported.
- GPU failures warn and fall back to CPU.
- `up_sample=False` remains unchanged from the volume-density-mapper algorithm.
- The 3-D output shape remains stable when `decomposition_map_n` is used.
