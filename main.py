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

def convert_images(image, original_image):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))
    print()
    return image, original_image

if __name__ == '__main__':
    print('\nInputs:')
    file = input('Image path: ')
    kernelShape = int(input('Gaussian Kernel shape: '))
    print('')

    image = cv2.imread(file)
    original_image = image

    image, original_image = convert_images(image, original_image)

    verbose = True
    rows = 3
    columns = 3

    plot_image(original_image, 'Original image:', rows, columns, 1)
    plot_image(image, 'Grayscale image:', rows, columns, 2)

    print('Gaussian Blur:')
    image, kernel = gaussian_blur(image, kernelShape, verbose=verbose)
    plot_image(kernel, 'Gaussian Kernel:', rows, columns, 3)
    plot_image(image, 'Gaussian Blur:', rows, columns, 4)

    print('Sobel:')
    image, thetaMat = sobel_filters(image, verbose=verbose)
    plot_image(image, 'Sobel:', rows, columns, 5)

    print('Non-Max Suppression:')
    image = non_max_suppression(image, thetaMat, verbose=verbose)
    plot_image(image, 'Non-Max Suppression:', rows, columns, 6)

    print('Threshold:')
    image, weak, strong = threshold(image, verbose=verbose)
    plot_image(image, 'Threshold:', rows, columns, 7)

    print('Hysteresis:')
    image = hysteresis(image, weak, strong, verbose=verbose)
    plot_image(image, 'Hysteresis:', rows, columns, 8)

    plt.tight_layout(pad=2)
    plt.get_current_fig_manager().window.state('zoomed') #Toggle fullscreen mode
    plt.show()