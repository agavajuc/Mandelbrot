"""
Turns the raw kernel output (n, z) into an RGB image. Kept separate from the formula on purpose:
any colouring works with any fractal, as long as the kernel outputs the same two arrays.
"""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb


def _normalise(values, low=1, high=99.5):
    """Scale to 0..1 using percentiles of this frame (robust to a few extreme pixels)."""
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [low, high])
    return np.clip((values - lo) / max(hi - lo, 1e-12), 0, 1)


def escape_colouring(n, cmap="twilight_shifted"):
    """Diverge mode: colour by (log of) escape time; points that never escaped are black."""
    rgb = np.zeros(n.shape + (3,))
    escaped = n >= 0
    t = _normalise(np.log1p(n[escaped]))
    rgb[escaped] = colormaps[cmap](t)[:, :3]
    return rgb


def basin_colouring(n, z):
    """Converge mode: hue = which point the orbit converged to (its angle), brightness = how fast."""
    rgb = np.zeros(n.shape + (3,))
    done = n >= 0
    hue = (np.angle(z[done]) / (2 * np.pi)) % 1.0
    value = 1.0 - 0.75 * _normalise(np.log1p(n[done]))
    rgb[done] = hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), value], axis=-1))
    return rgb


def colour(spec, n, z):
    if spec.stop_mode == "converge":
        return basin_colouring(n, z)
    return escape_colouring(n, spec.cmap)
