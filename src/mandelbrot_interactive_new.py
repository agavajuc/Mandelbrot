import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

BAILOUT_SQ = 256.0 ** 2      # big escape radius -> smooth (continuous) colouring is accurate
EPS = np.finfo(np.float64).eps


# --------------------------------------------------------------------------------------
# Numba kernels
# --------------------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def mandelbrot_smooth(cr, ci, max_iter):
    """Smooth escape count for c = cr + i*ci, or -1.0 if c never escaped (treated as 'inside')."""
    zr = 0.0
    zi = 0.0
    for i in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > BAILOUT_SQ:
            # Continuous iteration count: removes the hard colour bands of integer counts
            log_zn = 0.5 * np.log(zr2 + zi2)
            return i + 1.0 - np.log(log_zn / np.log(2.0)) / np.log(2.0)
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr
    return -1.0


@njit(parallel=True, fastmath=True, cache=True)
def mandelbrot_grid(cx, cy, step, width, height, max_iter):
    """Grid centred on (cx, cy) with pixel spacing `step`."""
    out = np.empty((height, width))
    for i in prange(height):
        ci = cy + (i - (height - 1) / 2.0) * step
        for j in range(width):
            cr = cx + (j - (width - 1) / 2.0) * step
            out[i, j] = mandelbrot_smooth(cr, ci, max_iter)
    return out


# --------------------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------------------
class MandelbrotViewer:
    """
    Scroll        zoom in/out around the cursor
    + / -         double / halve the iteration limit (overrides the automatic value)
    a             back to automatic iteration limit
    r             reset view
    """

    START = (-0.5, 0.0, 3.0)     # centre x, centre y, view width

    def __init__(self, width=1000, height=1000, cmap="twilight_shifted", base_iter=100, iter_cap=50_000):
        self.width, self.height = width, height
        self.base_iter, self.iter_cap = base_iter, iter_cap
        self.manual_iter = None              # None -> automatic

        self.cx, self.cy, self.view_w = self.START

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("black")                  # NaN (= inside the set) is drawn black, never confused with a colour
        self.im = self.ax.imshow(np.zeros((height, width)), cmap=cm, origin="lower",
                                 interpolation="nearest")
        self.ax.set_xlabel("Re(c)")
        self.ax.set_ylabel("Im(c)")

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()

    # ----- geometry -----------------------------------------------------------------
    @property
    def step(self):
        return self.view_w / (self.width - 1)

    def extent(self):
        hw = self.step * (self.width - 1) / 2
        hh = self.step * (self.height - 1) / 2
        return [self.cx - hw, self.cx + hw, self.cy - hh, self.cy + hh]

    # ----- iteration limit ----------------------------------------------------------
    def auto_iter(self):
        # Escape counts near the boundary grow as you zoom; how fast depends on WHERE you are
        # (slowly at filament tips, quickly in 'valleys' and near minibrots). This grows a bit
        # faster than log(zoom) so valleys stay resolved; press + if you still see black blobs.
        zoom = max(self.START[2] / self.view_w, 1.0)
        return int(self.base_iter + 50 * np.log10(zoom) ** 1.5)

    def current_iter(self):
        it = self.manual_iter if self.manual_iter is not None else self.auto_iter()
        return int(np.clip(it, 50, self.iter_cap))

    # ----- drawing ------------------------------------------------------------------
    def render(self):
        max_iter = self.current_iter()
        t0 = time.perf_counter()
        data = mandelbrot_grid(self.cx, self.cy, self.step, self.width, self.height, max_iter)
        dt = time.perf_counter() - t0

        data[data < 0] = np.nan                       # inside -> NaN -> black
        escaped = data[np.isfinite(data)]
        img = np.log1p(np.maximum(data, 0))           # log compresses the long tail of counts

        self.im.set_data(img)
        self.im.set_extent(self.extent())
        if escaped.size:
            # THE key fix: rescale colours to this frame. Otherwise the limits from the first
            # frame (1..100) stay forever and every count above 100 gets the same colour.
            lo, hi = np.percentile(np.log1p(escaped), [1, 99.5])
            self.im.set_clim(lo, max(hi, lo + 1e-9))

        inside = 100 * (1 - escaped.size / data.size)
        # Float64 limit: neighbouring pixels must differ by more than the spacing of
        # representable doubles around |c|, or they collapse onto the same number.
        ulp = EPS * max(abs(self.cx), abs(self.cy), 1e-300)
        precision = self.step / ulp
        warn = "   ⚠ float64 limit — pixels are merging" if precision < 8 else ""
        mode = "manual" if self.manual_iter is not None else "auto"
        self.ax.set_title(
            f"width {self.view_w:.2e}  |  zoom ×{self.START[2] / self.view_w:.1e}  |  "
            f"max_iter {max_iter} ({mode})\n"
            f"{inside:.1f}% black  |  {dt * 1000:.0f} ms{warn}",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

    # ----- events -------------------------------------------------------------------
    def on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        f = 0.5 if event.button == "up" else 2.0
        # keep the point under the cursor fixed
        self.cx = event.xdata + (self.cx - event.xdata) * f
        self.cy = event.ydata + (self.cy - event.ydata) * f
        self.view_w = min(self.view_w * f, 8.0)
        self.render()

    def on_key(self, event):
        if event.key in ("+", "="):
            self.manual_iter = min(self.current_iter() * 2, self.iter_cap)
        elif event.key == "-":
            self.manual_iter = max(self.current_iter() // 2, 50)
        elif event.key == "a":
            self.manual_iter = None
        elif event.key == "r":
            self.cx, self.cy, self.view_w = self.START
            self.manual_iter = None
        else:
            return
        self.render()


if __name__ == "__main__":
    viewer = MandelbrotViewer(width=1000, height=1000)
    plt.show()
