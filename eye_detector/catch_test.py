from time import time

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from matplotlib import pyplot as plt

from eye_detector.heatmap import compute_heatmap, crop_heatmap
from eye_detector.model import load_window


def test(window):
    img = imread("test/image.jpg")
    img = rgb2gray(img)
    img = resize(img, (640, 480))
    plt.imshow(img)
    plt.show()

    t = time()
    heatmap = compute_heatmap(img.shape, window(img))
    croped = crop_heatmap(heatmap)
    print("time:", time() - t)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()

    plt.imshow(img * croped)
    plt.show()


if __name__ == "__main__":
    window = load_window()
    test(window)
