
import sys
import types
import warnings

import numpy as np
import pytest

import volume_density_mapper.density_reconstruction as dr


def _fake_decomposition(data, **kwargs):
    data = np.asarray(data, dtype=np.float32)
    # Two deterministic layers that sum back to the input.
    result = np.stack((0.6 * data, 0.4 * data), axis=0)
    residual = np.zeros_like(data)
    scales = np.array([1.0, 2.0], dtype=np.float32)
    return result, residual, scales


def test_public_cpu_path(monkeypatch):
    monkeypatch.setattr(
        dr.cdd, "constrained_diffusion_decomposition", _fake_decomposition
    )
    data = np.arange(35, dtype=np.float64).reshape(5, 7) + 1
    width = dr.compute_characteristic_scale(data, dx=2.0, padding=False)
    density, width2 = dr.compute_mean_density_width(data, dx=2.0, padding=False)
    cube = dr.density_reconstruction_3d(data, dx=2.0, padding=False)

    assert width.shape == data.shape
    assert density.shape == data.shape
    assert np.allclose(width, width2)
    assert cube.shape == (7, 5, 7)
    assert cube.dtype == np.float32
    assert np.isfinite(cube).all()


def test_cpu_mode_does_not_require_new_cdd_signature(monkeypatch):
    calls = []

    def old_style_cdd(data, up_sample=False, return_scales=True,
                      log_scale_base=np.sqrt(2), mode=None):
        calls.append(True)
        return _fake_decomposition(data)

    monkeypatch.setattr(
        dr.cdd, "constrained_diffusion_decomposition", old_style_cdd
    )
    out = dr.compute_characteristic_scale(
        np.ones((4, 6)), padding=False, use_GPU=False
    )
    assert out.shape == (4, 6)
    assert calls


def test_gpu_request_falls_back_when_cupy_is_unavailable(monkeypatch):
    seen = []

    def cdd_with_gpu(data, **kwargs):
        seen.append(kwargs.copy())
        return _fake_decomposition(data)

    monkeypatch.setattr(
        dr.cdd, "constrained_diffusion_decomposition", cdd_with_gpu
    )
    monkeypatch.setattr(
        dr, "_load_cupy",
        lambda: (_ for _ in ()).throw(RuntimeError("no CUDA/CuPy"))
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cube = dr.density_reconstruction_3d(
            np.ones((4, 6), dtype=np.float32),
            dx=1.0,
            padding=False,
            use_GPU=True,
        )

    assert cube.shape == (6, 4, 6)
    assert any("Falling back to CPU reconstruction" in str(w.message)
               for w in caught)
    assert any(call.get("use_gpu") is True for call in seen)


def test_gpu_cdd_uses_auto_chunking_when_supported(monkeypatch):
    captured = {}

    def current_cdd(data, use_gpu=False, n_chunk=1, **kwargs):
        captured["use_gpu"] = use_gpu
        captured["n_chunk"] = n_chunk
        return _fake_decomposition(data)

    monkeypatch.setattr(
        dr.cdd, "constrained_diffusion_decomposition", current_cdd
    )
    dr.compute_characteristic_scale(
        np.ones((4, 6), dtype=np.float64),
        padding=False,
        use_GPU=True,
    )
    assert captured["use_gpu"] is True
    assert captured["n_chunk"] is None


def test_decomposition_limit_does_not_change_public_cube_shape(monkeypatch):
    monkeypatch.setattr(
        dr.cdd, "constrained_diffusion_decomposition", _fake_decomposition
    )
    cube = dr.density_reconstruction_3d(
        np.ones((3, 5)),
        dx=1.0,
        padding=False,
        decomposition_map_n=1,
    )
    assert cube.shape == (5, 3, 5)
