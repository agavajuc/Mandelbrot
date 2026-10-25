"""Run with:  python -m pytest tests/ -q   (from the fractal_engine folder)"""
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import FormulaError, compute, generate_source, load_spec  # noqa: E402

HERE = os.path.dirname(__file__)
MANDEL = load_spec(os.path.join(HERE, "..", "fractals", "mandelbrot.toml"))


def reference_mandelbrot(cx, cy, view_width, size, max_iter, radius):
    """Plain, slow, obviously-correct Python. The engine must agree with this."""
    step = view_width / (size - 1)
    out = np.full((size, size), -1)
    for i in range(size):
        for j in range(size):
            c = complex(cx + (j - (size - 1) / 2) * step, cy + (i - (size - 1) / 2) * step)
            z = 0j
            for k in range(max_iter):
                z = z * z + c
                if abs(z) > radius:
                    out[i, j] = k
                    break
    return out


def test_matches_plain_python():
    size, max_iter = 60, 200
    n, _ = compute(MANDEL, size, size, max_iter=max_iter, periodicity=False)
    ref = reference_mandelbrot(*MANDEL.center, MANDEL.width, size, max_iter, MANDEL.radius)
    # smooth count lies in [k, k+1), so its floor is the plain integer count
    engine_int = np.where(n >= 0, np.floor(n), -1)
    assert (engine_int == ref).mean() > 0.999


def test_periodicity_only_skips_work():
    """Periodicity checking may only stop pixels that would never escape anyway."""
    kw = dict(width_px=300, height_px=300, max_iter=2000)
    n_off, _ = compute(MANDEL, periodicity=False, **kw)
    n_on, _ = compute(MANDEL, periodicity=True, **kw)
    assert np.array_equal(n_off < 0, n_on < 0)
    assert np.allclose(n_off[n_off >= 0], n_on[n_on >= 0])


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo hacked')",   # function that is not whitelisted
    "z.real",                                    # attribute access
    "[z for z in range(3)]",                     # comprehension
    "w**2 + c",                                  # unknown variable
    "'text'",                                    # non-numeric constant
    "z**2 +",                                    # syntax error
])
def test_rejects_unsafe_or_invalid_formulas(bad):
    spec = dataclasses.replace(MANDEL, step=bad)
    with pytest.raises(FormulaError):
        generate_source(spec)


def test_reserved_names_rejected():
    spec = dataclasses.replace(MANDEL, setup={"z": "0", "k": "p"})
    with pytest.raises(FormulaError):
        generate_source(spec)
