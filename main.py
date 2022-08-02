import cv2
import matplotlib.pyplot as plt
from convolution import convolution
from sobel import sobel_filters
from gaussianBlur import gaussian_blur
from nonMaxSupression import non_max_suppression
from threshold import threshold
from hysteresis import hysteresis

if __name__ == '__main__': 
    file = input('Input image: ')
    kernelShape = int(input('Gaussian Kernel shape: '))
    image = cv2.imread(file)

    verbose = True
 
    image = gaussian_blur(image, kernelShape, verbose=verbose)

    image, thetaMat = sobel_filters(image, verbose=verbose)

    image = non_max_suppression(image, thetaMat, verbose=verbose)

    image, weak, strong = threshold(image, verbose=verbose)

    image = hysteresis(image, weak, strong, verbose=verbose)