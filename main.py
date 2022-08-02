import cv2
import matplotlib.pyplot as plt
from convolution import convolution
from sobel import sobel_filters
from gaussianBlur import gaussian_blur
from nonMaxSupression import non_max_suppression
from threshold import threshold
from hysteresis import hysteresis

def plot_image(img, title, rows, columns, index, color='gray'):
    plt.subplot(rows, columns, index)
    plt.imshow(img, cmap=color)
    plt.axis('off')
    plt.title(title)

def convert_images(img, original_img):
    if len(img.shape) == 3:
        print("Found 3 Channels : {}".format(img.shape))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        print("Converted to Gray Channel. Size : {}".format(img.shape))
    else:
        print("Image Shape : {}".format(img.shape))
    print()
    return img, original_img

if __name__ == '__main__':
    default_input = True

    if default_input:
        file = 'test.jpg'
        kernel_shape = 9
    else:
        print('\nInputs:')
        file = input('Image path: ')
        kernel_shape = int(input('Gaussian Kernel shape: '))
        print('')

    img = cv2.imread(file)
    original_img = img

    gray_img, original_img = convert_images(img, original_img)

    verbose = False
    rows = 3
    columns = 3

    print('Gaussian Blur:')
    gaussBlur_img, gauss_kernel = gaussian_blur(gray_img, kernel_shape, verbose=verbose)

    print('Sobel:')
    sobel_img, thetaMat = sobel_filters(gaussBlur_img, verbose=verbose)

    print('Non-Max Suppression:')
    nms_img = non_max_suppression(sobel_img, thetaMat, verbose=verbose)

    print('Threshold:')
    threshold_img, weak, strong = threshold(nms_img, verbose=verbose)

    print('Hysteresis:')
    hysteresis_img = hysteresis(threshold_img, weak, strong, verbose=verbose)

    plot_image(original_img, 'Original:', rows, columns, 1)
    plot_image(gray_img, 'Grayscale:', rows, columns, 2)
    plot_image(gauss_kernel, 'Gaussian Kernel {}X{}:'.format(kernel_shape, kernel_shape), rows, columns, 3)
    plot_image(gaussBlur_img, 'Gaussian Blur:', rows, columns, 4)
    plot_image(sobel_img, 'Sobel:', rows, columns, 5)
    plot_image(nms_img, 'Non-Max Suppression:', rows, columns, 6)
    plot_image(threshold_img, 'Threshold:', rows, columns, 7)
    plot_image(hysteresis_img, 'Hysteresis:', rows, columns, 8)
    plt.tight_layout()
    plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
    plt.show()