import numpy as np
import matplotlib.pyplot as plt

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09, verbose=False):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    if verbose:
        plt.imshow(res, cmap='gray')
        plt.title("Output for Threshold")
        plt.show()
    
    return (res, weak, strong)