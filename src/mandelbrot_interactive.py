import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

# Keep our lightning-fast Numba engine exactly as it was
@jit(nopython=True, fastmath=True)
def mandelbrot(c, max_iter):
    z = 0.0j
    for i in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0:
            return i
        z = z ** 2 + c
    return max_iter

@jit(nopython=True, parallel=True, fastmath=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    mset = np.zeros((height, width))
    for i in prange(height):
        for j in range(width):
            mset[i, j] = mandelbrot(complex(x[j], y[i]), max_iter)
    return mset

class MandelbrotViewer:
    def __init__(self, width=1000, height=1000):
        # 1. Setup initial view bounds and fixed pixel resolution
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.5, 1.5
        self.width = width
        self.height = height
        
        # 2. Setup the Matplotlib Figure and Axis
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Calculate initial frame
        self.max_iter = 100
        self.img_data = mandelbrot_set(self.xmin, self.xmax, self.ymin, self.ymax, 
                                       self.width, self.height, self.max_iter)
        
        # Render the initial image object
        self.im = self.ax.imshow(self.img_data, extent=[self.xmin, self.xmax, self.ymin, self.ymax], 
                                 cmap='twilight_shifted', origin='lower')
        
        # self.ax.set_title("Use Mouse Scroll Wheel to Zoom In/Out")
        self.ax.set_xlabel("Re(c)")
        self.ax.set_ylabel("Im(c)")

        # 3. CRITICAL: Bind the scroll event to our class function
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        # If the scroll happens outside the plot area, ignore it
        if event.xdata is None or event.ydata is None:
            return

        # Determine zoom factor based on scroll direction
        # event.button can be 'up' (zoom in) or 'down' (zoom out)
        zoom_factor = 0.5 if event.button == 'up' else 2.0

        # Get the coordinate position where the mouse cursor is hovering
        mouse_x = event.xdata
        mouse_y = event.ydata

        # Calculate the new boundary limits centered relative to the mouse cursor
        self.xmin = mouse_x - (mouse_x - self.xmin) * zoom_factor
        self.xmax = mouse_x + (self.xmax - mouse_x) * zoom_factor
        self.ymin = mouse_y - (mouse_y - self.ymin) * zoom_factor
        self.ymax = mouse_y + (self.ymax - mouse_y) * zoom_factor

        # Solve the boundary problem: dynamically scale max_iter based on zoom level
        current_width = self.xmax - self.xmin
        # self.max_iter = int(100 + 120 * np.log10(3.0 / current_width))
        # self.max_iter = int(100 + 15 * ((3.0 / current_width) ** 0.5))
        self.max_iter = 100

        current_iter = max(100, min(self.max_iter, 15000))
        
        # Recalculate ONLY the current visible 1000x1000 viewport
        self.img_data = mandelbrot_set(self.xmin, self.xmax, self.ymin, self.ymax, 
                                       self.width, self.height, current_iter)
        
        # Update the existing Matplotlib image canvas with new data and new scale boundaries
        self.im.set_data(self.img_data)
        self.im.set_extent([self.xmin, self.xmax, self.ymin, self.ymax])
        
        # Force Matplotlib to instantly redraw the window canvas
        self.fig.canvas.draw_idle()

    # def self_iter_cap(self):
    #     # Prevent max_iter from becoming lower than 100 or ridiculously high
    #     return max(100, min(self.max_iter, 10000))

if __name__ == '__main__':
    # Fire up the engine!
    viewer = MandelbrotViewer(width=1200, height=1200)
    plt.show()
