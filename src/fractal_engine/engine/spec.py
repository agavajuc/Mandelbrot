"""
A fractal *spec* is plain data: everything needed to describe one fractal and one view of it.
It is loaded from a small TOML file (see the fractals/ folder) and knows nothing about Numba.
"""
import tomllib
from dataclasses import dataclass


@dataclass
class FractalSpec:
    name: str
    step: str                           # the recurrence, e.g. "z**2 + c"
    setup: dict                         # per-pixel starting values; may use p (the pixel) and earlier names
    stop_mode: str = "diverge"          # "diverge" (|z| -> infinity) or "converge" (z_{n+1} ~ z_n)
    radius: float = 256.0               # bailout radius for diverge mode
    degree: float | None = 2.0          # degree of the step in z, for smooth colouring (0 = plain integer counts)
    tolerance: float = 1e-10            # converge mode: |z_{n+1} - z_n| < tolerance means "converged"
    center: tuple = (0.0, 0.0)
    width: float = 4.0
    max_iter: int = 500
    cmap: str = "twilight_shifted"
    periodicity: bool = True            # Brent periodicity checking (diverge mode only)


def load_spec(path):
    with open(path, "rb") as f:
        d = tomllib.load(f)
    stop = d.get("stop", {})
    view = d.get("view", {})
    colour = d.get("colour", {})
    return FractalSpec(
        name=d["name"],
        step=d["step"],
        setup=dict(d["setup"]),
        stop_mode=stop.get("mode", "diverge"),
        radius=float(stop.get("radius", 256.0)),
        degree=stop.get("degree", 2.0),
        tolerance=float(stop.get("tolerance", 1e-10)),
        periodicity=bool(stop.get("periodicity", True)),
        center=tuple(view.get("center", (0.0, 0.0))),
        width=float(view.get("width", 4.0)),
        max_iter=int(view.get("max_iter", 500)),
        cmap=colour.get("cmap", "twilight_shifted"),
    )


def dump_spec(s: FractalSpec) -> str:
    """Write a spec back as TOML (by hand: the format is simple and tomllib can only read)."""
    q = lambda text: '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'
    lines = [f"name = {q(s.name)}", f"step = {q(s.step)}", "", "[setup]"]
    lines += [f"{k} = {q(v)}" for k, v in s.setup.items()]
    lines += ["", "[stop]", f"mode = {q(s.stop_mode)}", f"radius = {s.radius!r}"]
    if s.degree is not None:
        lines.append(f"degree = {s.degree!r}")
    lines += [f"tolerance = {s.tolerance!r}", f"periodicity = {str(s.periodicity).lower()}"]
    lines += ["", "[view]", f"center = [{s.center[0]!r}, {s.center[1]!r}]",
              f"width = {s.width!r}", f"max_iter = {s.max_iter}"]
    lines += ["", "[colour]", f"cmap = {q(s.cmap)}", ""]
    return "\n".join(lines)
