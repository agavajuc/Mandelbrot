import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import time

# 1. Compile the core math to machine code
@jit(nopython=True, fastmath=True)
def mandelbrot(c, max_iter):
    z = 0.0j
    for i in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4.0: # Optimizing abs(z) > 2 to avoid square roots
            return i
        z = z * z + c
    return max_iter

# 2. Parallelize the entire 2D grid loop instantly with 'parallel=True' and 'prange'
@jit(nopython=True, parallel=True, fastmath=True)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    mset = np.zeros((height, width))

    # prange automatically splits these loops across all your available CPU threads
    for i in prange(height):
        for j in range(width):
            c = complex(x[j], y[i])
            mset[i, j] = mandelbrot(c, max_iter)
            
    return mset

if __name__ == '__main__':
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 4000, 4000
    max_iter = 100

    print("Computing Mandelbrot set with Numba...")
    start_time = time.time()
    
    mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    
    print(f"Done! Calculation took: {time.time() - start_time:.4f} seconds")

    plt.imshow(mandelbrot_image, extent=[xmin, xmax, ymin, ymax], cmap='hot')
    plt.colorbar()
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()
