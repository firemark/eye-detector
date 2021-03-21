from time import time
from glob import glob

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import adjust_log
from skimage.io import imread
from matplotlib import pyplot as plt

from eye_detector.heatmap import compute_heatmap, crop_heatmap
from eye_detector.model import load_window


def test(window, img_path, size=None):
    img = imread(img_path)
    img = rgb2gray(img)
    img = adjust_log(img)
    if size:
        img = resize(img, size)
    plt.show()

    t = time()
    heatmap = compute_heatmap(img.shape, window(img))
    heatmap **= 2
    croped = crop_heatmap(heatmap)
    print("time:", time() - t)

    fig, axes = plt.subplots(nrows=2, ncols=2)

    axes[0,0].set_title("image")
    axes[0,0].imshow(img, cmap="gray")

    axes[1,0].set_title("heatmap")
    axes[1, 0].imshow(heatmap / np.max(heatmap), cmap="hot")
    #plt.colorbar(im_heatmap, ax=axes[1, 0], orientation="horizontal")

    axes[1,1].set_title("regions")
    axes[1,1].imshow(img * croped, cmap="gray")

    plt.show()


if __name__ == "__main__":
    window = load_window()
    #test(window, "test/image.jpg", size=(640, 480))

    for path in glob("indata/00000/*.png", recursive=True):
        test(window, path)
