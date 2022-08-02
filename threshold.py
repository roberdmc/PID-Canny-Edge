import numpy as np
import matplotlib.pyplot as plt

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09, verbose=False):
    print('Double Thresholding:')

    #Used to identify the strong pixels
    highThreshold = img.max() * highThresholdRatio;
    #Used to identify non-relevant pixels
    lowThreshold = highThreshold * lowThresholdRatio;
    
    #Initialize matrix of output image with 0
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    #Initialize vectors
    weak = np.int32(25)
    strong = np.int32(255)
    
    #Set the strong pixels
    strong_i, strong_j = np.where(img >= highThreshold)
    #Set the non-relevant pixels
    zeros_i, zeros_j = np.where(img < lowThreshold)
    #Set the weak pixels
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    #Fill the output image with only strong and weak pixels
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    #Print the output image
    if verbose:
        plt.imshow(res, cmap='gray')
        plt.title("Output for Double Thresholding")
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()
    
    print('Success Double Thresholding\n')
    return (res, weak, strong)