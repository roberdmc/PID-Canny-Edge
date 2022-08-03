from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from convolution import convolution

#Gradient calculation, to detect the edge intensity and direction
def sobel_filters(img, verbose=False):
    print('Start Sobel edge detection:')
    
    #Define sobel horizontal and vertical filters
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    #Apply filters to the image
    Ix = convolution(img, Kx)
    Iy = convolution(img, Ky)
    
    #Obtain magnitude of the gradient, combining vertical and horizontal edges 
    G = np.sqrt(np.square(Ix) + np.square(Iy))
    #Normalizing output to be between 0 and 255
    G = G / G.max() * 255
    #Store the slope of the gradient
    theta = np.arctan2(Iy, Ix)

    #Print output images
    if verbose:
        plt.imshow(Ix, cmap='gray')
        plt.title("Output Image using sobel horizontal")
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()
        plt.imshow(Iy, cmap='gray')
        plt.title("Output Image using sobel vertical")
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()
        plt.imshow(G, cmap='gray')
        plt.title("Output Image using sobel united")
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()

    print('Finish Sobel edge detection!\n')
    return (G, theta, Ix, Iy)