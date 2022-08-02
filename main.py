import cv2
import matplotlib.pyplot as plt
from convolution import convolution
from sobel import sobel_filters
from gaussianBlur import gaussian_blur
from nonMaxSupression import non_max_suppression
from threshold import threshold
from hysteresis import hysteresis

#Configure images for final plot
def plot_image(img, title, rows, columns, index, color='gray'):
    plt.subplot(rows, columns, index)
    plt.imshow(img, cmap=color)
    plt.axis('off')
    plt.title(title)

def convert_images(img):
    #If input image have 3 channels, convert to 1 channel
    if len(img.shape) == 3:
        print("Found 3 Channels : {}".format(img.shape))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Converted to Gray Channel. Size : {}".format(gray_img.shape))
        print()
        return gray_img, original_img
    else:
        print("Image Shape : {}".format(img.shape))
        print()
        return img, img

if __name__ == '__main__':
    #Default or user inputs
    default_input = True
    #Show step by step output images
    verbose = False

    #Default input for tests
    if default_input:
        file = 'lena.png'
        kernel_shape = 5
    #User input
    else:
        print('\nInputs:')
        file = input('Image path: ')
        kernel_shape = int(input('Gaussian Kernel shape: '))
        print('')

    #Read and convert image
    original_img = cv2.imread(file)
    gray_img, original_img = convert_images(original_img)

    #Print original image, if verbose
    if verbose:
        plt.imshow(original_img)
        plt.title("Original image:")
        plt.get_current_fig_manager().window.state('zoomed')
        plt.axis('off')
        plt.show()
    
    #Processing filters
    gaussBlur_img, gauss_kernel = gaussian_blur(gray_img, kernel_shape, verbose=verbose)
    sobel_img, thetaMat, sobel_x, sobel_y = sobel_filters(gaussBlur_img, verbose=verbose)
    nms_img = non_max_suppression(sobel_img, thetaMat, verbose=verbose)
    threshold_img, weak, strong = threshold(nms_img, verbose=verbose)
    hysteresis_img = hysteresis(threshold_img, weak, strong, verbose=verbose)

    #Plot all output images
    rows = 3
    columns = 4
    plot_image(original_img, 'Original:', rows, columns, 1)
    plot_image(gray_img, 'Grayscale:', rows, columns, 2)
    plot_image(gauss_kernel, 'Gaussian Kernel {}X{}:'.format(kernel_shape, kernel_shape), rows, columns, 3)
    plot_image(gaussBlur_img, 'Gaussian Blur:', rows, columns, 4)
    plot_image(sobel_x, 'Horizontal Sobel:', rows, columns, 5)
    plot_image(sobel_y, 'Vertical Sobel:', rows, columns, 6)
    plot_image(sobel_img, 'Sobel edge detection:', rows, columns, 7)
    plot_image(nms_img, 'Non-Max Suppression:', rows, columns, 8)
    plot_image(threshold_img, 'Double Thresholding:', rows, columns, 9)
    plot_image(hysteresis_img, 'Edge Tracking by Hysteresis:', rows, columns, 10)
    plt.tight_layout()
    plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
    plt.show()
    