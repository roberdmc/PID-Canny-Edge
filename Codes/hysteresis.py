import matplotlib.pyplot as plt
import numpy as np

def hysteresis(image, weak, strong=255, verbose=False):
    print('Start Edge Tracking by Hysteresis:')
    
    M, N = image.shape
    img = np.zeros((M,N), dtype=np.int32)
    
    img += image
    
    #Transforms weak pixels into strong, if at least one of the pixels around is strong
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    #If have strong pixel around, make the pixel strong
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    #Else, discard pixel
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    
    #Print output image
    if verbose:
        plt.imshow(img, cmap='gray')
        plt.title("Output for Edge Tracking by Hysteresis")
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()

    print('Finish Edge Tracking by Hysteresis!\n')
    return img