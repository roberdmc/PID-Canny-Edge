from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from convolution import convolution

def sobel_filters(img, verbose=False):
    print('Sobel:')
    
    #Define sobel horizontal and vertical filters
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    #Apply filters to the image
    Ix = convolution(img, Kx)
    Iy = convolution(img, Ky)
    
    #Obtain magnitude of the gradient, uniting sobel x and y, using hypotenuse for the right angled triangle
    G = np.hypot(Ix, Iy)
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

    print('Success Sobel\n')
    return (G, theta)