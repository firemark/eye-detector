from time import time
from glob import glob
from random import shuffle

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import random_noise
from skimage.io import imread
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from eye_detector.heatmap import compute_heatmap, crop_heatmap
from eye_detector.model import load_window


def test(window, img_path, size=None):
    img = imread(img_path)
    img = img[:, :, 0:3]
    if size:
        img = resize(img, size)
    img = random_noise(img, var=0.001)

    t = time()
    heatmap = compute_heatmap(img.shape, window(img))
    heatmap **= 2
    croped = crop_heatmap(heatmap)
    tt = time() - t
    return img, heatmap, croped, tt



class Index:

    def __init__(self, window):
        self.ind = 0
        self.window = window
        self.images = glob("indata/00000/*.png", recursive=True)
        shuffle(self.images)

    def setup_plot(self):
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2)
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])

        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next)
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_prev.on_clicked(self.prev)


    def next(self, event):
        self.ind = (self.ind + 1) % len(self.images)
        self.run_test()

    def prev(self, event):
        self.ind = (self.ind - 1) % len(self.images)
        self.run_test()

    def run_test(self):
        window = self.window
        img_path = self.images[self.ind]
        img, heatmap, croped, tt = test(window, img_path)
        max_h = np.max(heatmap)

        self.fig.suptitle(f"time: {tt:6f}s")

        self.axes[0,0].set_title("image")
        self.axes[0,0].imshow(img, cmap="hot")

        self.axes[1,0].set_title("heatmap")
        if max_h != 0:
            self.axes[1, 0].imshow(heatmap / max_h, cmap="hot")
        else:
            self.axes[1, 0].imshow(heatmap, cmap="hot")

        self.axes[1,1].set_title("regions")
        self.axes[1,1].imshow(img * croped, cmap="hot")
        plt.draw()


if __name__ == "__main__":
    window = load_window()
    index = Index(window)
    index.setup_plot()
    index.run_test()
    plt.show()
