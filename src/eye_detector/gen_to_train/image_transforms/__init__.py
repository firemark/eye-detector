import numpy as np
from skimage.color import rgb2ypbpr, rgb2gray, rgb2hsv
from skimage.exposure import equalize_adapthist


def rgb(img):
    return img


def gray(img):
    img = rgb2gray(img)
    return img
    #return equalize_adapthist(img / np.max(img))


IMAGE_TRANSFORMS = {
    'yuv': rgb2ypbpr,
    'gray': gray,
    'rgb': rgb,
    'hsv': rgb2hsv,
}
