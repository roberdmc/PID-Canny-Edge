import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(image, kernel, average=False, verbose=False):
    print("Kernel Shape: {}".format(kernel.shape))
    
    #Print grayscale image
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Grayscale image")
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
 
    #Define the sizes of kernel and submatrix of image
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
 
    #Initialize output matrix with 0
    output = np.zeros(image.shape)
 
    #Define the pad size
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
 
    #Initialize and fill padded image 
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
 
    #Print padded image
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()
 
    #Apply kernel to all points of matriz
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
 
    #Print output image size, for validation
    print("Output Image size: {}".format(output.shape))
 
    #Print output image
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Gaussian Kernel".format(kernel_row, kernel_col))
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()
 
    return output