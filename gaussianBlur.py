import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math
from convolution import convolution
from sobel import sobel_filters

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Gaussian Kernel ( {}X{} )".format(size, size))
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()
 
    return kernel_2D

def gaussian_blur(image, kernel_size, verbose=False):
    print('Gaussian Blur:')
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    image = convolution(image, kernel, average=True, verbose=verbose)
    print('Success Gaussian Blur\n')
    return image, kernel