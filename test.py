#!/usr/bin/env python
import warnings

import scipy.misc
import skimage
import skimage.color
import skimage.transform
import sgm
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path, downsampling=1, downsampling_mode='rescale'):
    """Loads an image from path as a numpy array. Downsamples by the specified power of 2.

    Args:
        downsampling: an integer factor by which to reduce the image size.
        downsampling_mode: the name of the skimage function to use. Options:
        'rescale' (fast) or 'downscale_local_mean' (accurate)
    """
    image = skimage.img_as_float(scipy.misc.imread(image_path))
    if downsampling_mode == 'rescale':
        return skimage.transform.rescale(image, 1.0 / downsampling, mode='constant')
    elif downsampling_mode == 'downscale_local_mean':
        return skimage.transform.downscale_local_mean(
            image, (downsampling, downsampling, 1))

def compute_disparity(images):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gray_images = [skimage.img_as_ubyte(skimage.color.rgb2gray(image), True)
                       for image in images]

    return sgm.compute_disparity(gray_images[0], gray_images[1])


def test():
    file_paths = ['example/left/2.png', 'example/right/2.png']
    images = [load_image(file_path) for file_path in file_paths]

    sgm.initialize_disparity(15, 170)
    disparity = compute_disparity(images)
    sgm.finish_disparity()

    plt.figure()
    plt.imshow(disparity)
    plt.show()

if __name__ == '__main__':
    test()
