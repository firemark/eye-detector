from skimage.color import rgb2ypbpr, rgb2gray, rgb2hsv


def rgb(img):
    return img


IMAGE_TRANSFORMS = {
    'yuv': rgb2ypbpr,
    'gray': rgb2gray,
    'rgb': rgb,
    'hsv': rgb2hsv,
}
