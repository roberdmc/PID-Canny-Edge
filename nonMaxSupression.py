import numpy as np
import matplotlib.pyplot as plt

def non_max_suppression(img, D, verbose=False):
    print('Non-Max Suppression:')
    
    M, N = img.shape
    
    #Initialize new matrix with 0 and calculate the angle of slope with theta of sobel step
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    #Check pixels in same direction and set the higher intensity
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #For angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #For angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #For angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #For angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    #Print output image
    if verbose:
        plt.imshow(Z, cmap='gray')
        plt.title("Output for Non-Max Supression")
        plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
        plt.show()
    
    print('Success Non-Max Supression\n')
    return Z