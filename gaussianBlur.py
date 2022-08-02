import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math
from convolution import convolution
from sobel import sobel_filters

#Equation for points of Gaussian kernel, using Univariate Normal Distribution
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False):
    #Initialize kernel vector with 0.0
    kernel = np.zeros((size, size), np.float32)
    m = size//2
    n = size//2
    
    #Calculate kernel points
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            aux1 = 2*np.pi*(sigma**2)
            aux2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            kernel[x+m, y+n] = (1/aux1)*aux2
    
    #Show step image output
    if verbose:
        plt.imshow(kernel, interpolation='none', cmap='gray')
        plt.title("Gaussian Kernel ( {}X{} )".format(size, size))
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
 
    return kernel

def gaussian_blur(image, kernel_size, verbose=False):
    print('Gaussian Blur:')
    #Generate kernel
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    #Operate image with kernel 
    image = convolution(image, kernel, verbose=verbose)
    print('Success Gaussian Blur\n')
    return image, kernel