import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def mandelbrot(c , max_iter):
    z = 0
    for i in range(max_iter):
        if abs(z) > 2:
            return i
        z = z * z + c
    return max_iter

def compute_row(i, x, y, width, max_iter):
        row = np.zeros(width)
        for j in range(width):
            c = complex(x[j], y[i])
            row[j] = mandelbrot(c, max_iter)
        return row

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    mset = np.zeros((height, width))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(compute_row, [(i, x, y, width, max_iter) for i in range(height)])
    
    # for i in range(height):
    #     mset[i, :] = results[i]
    mset = np.array(results)

    return mset

if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 20000, 20000
    max_iter = 100

    mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

    plt.imshow(mandelbrot_image, extent=[xmin, xmax, ymin, ymax], cmap='hot')
    plt.colorbar()
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()