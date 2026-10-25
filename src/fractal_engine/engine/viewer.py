"""
Minimal interactive viewer on top of the engine.

    scroll    zoom around the cursor
    + / -     double / halve max_iter
    r         reset to the view in the spec file
    s         save the current view as a new spec file (render it big later with `fractal.py render`)
"""
import dataclasses
import time

import matplotlib.pyplot as plt
import numpy as np

from .colouring import colour
from .compiler import compute
from .spec import dump_spec


class Viewer:
    def __init__(self, spec, size=800):
        self.spec = spec
        self.size = size
        self.reset()

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(np.zeros((size, size, 3)), origin="lower", interpolation="nearest")
        self.ax.set_xlabel("Re")
        self.ax.set_ylabel("Im")
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    def reset(self):
        self.center = tuple(self.spec.center)
        self.width = self.spec.width
        self.max_iter = self.spec.max_iter

    def _compute(self):
        return compute(self.spec, self.size, self.size, self.center, self.width, self.max_iter)

    def redraw(self):
        t0 = time.perf_counter()
        n, z = self._compute()
        ms = (time.perf_counter() - t0) * 1000
        half = self.width / 2
        cx, cy = self.center
        self.im.set_data(colour(self.spec, n, z))
        self.im.set_extent([cx - half, cx + half, cy - half, cy + half])
        self.ax.set_title(f"{self.spec.name}\nwidth {self.width:.2e} | max_iter {self.max_iter} | "
                          f"{(n < 0).mean() * 100:.0f}% black | {ms:.0f} ms", fontsize=10)
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        f = 0.5 if event.button == "up" else 2.0
        cx, cy = self.center
        # keep the point under the cursor where it is
        self.center = (event.xdata + (cx - event.xdata) * f, event.ydata + (cy - event.ydata) * f)
        self.width *= f
        self.redraw()

    def on_key(self, event):
        if event.key in ("+", "="):
            self.max_iter *= 2
        elif event.key == "-":
            self.max_iter = max(self.max_iter // 2, 10)
        elif event.key == "r":
            self.reset()
        elif event.key == "s":
            self.save()
            return
        else:
            return
        self.redraw()

    def save(self):
        view = dataclasses.replace(self.spec, center=self.center, width=self.width, max_iter=self.max_iter)
        path = f"{self.spec.name.split()[0].lower()}_view_{time.strftime('%H%M%S')}.toml"
        with open(path, "w") as f:
            f.write(dump_spec(view))
        print(f"saved view to {path}")


def explore(spec, size=800):
    viewer = Viewer(spec, size)
    plt.show()
    return viewer
