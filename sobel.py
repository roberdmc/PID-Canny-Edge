from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from convolution import convolution

def sobel_filters(img, verbose=False):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = convolution(img, Kx, average=True, verbose=False)
    Iy = convolution(img, Ky, average=True, verbose=False)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    if verbose:
        plt.imshow(G, cmap='gray')
        plt.title("Output Image using sobel")
        plt.show()

    return (G, theta)