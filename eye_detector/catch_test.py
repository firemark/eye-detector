from time import time
from glob import glob
from random import shuffle

import numpy as np
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.util import random_noise
from skimage.io import imread
from skimage.measure import label, regionprops
from matplotlib import patches, pyplot as plt
from matplotlib.widgets import Button

from eye_detector.heatmap import compute_heatmap, crop_heatmap
from eye_detector.model import load_window

FACE_SCALE = 3.5 #2.5
EYENOSE_SCALE = 1.75 #1.5
EYES_SCALE = 1.75
IMAGES_PATH = "indata/to_label/*/*.png"
#IMAGES_PATH = "indata/to_label/*/1621364139545.png"


def test(windows, img_path):
    img = imread(img_path)
    img = img[:, :, 0:3]
    img = random_noise(img, var=0.001)

    t = time()
    face_window, eyenose_window, eyes_window = windows
    face_heatmap, face_croped = face_detection(face_window, img)
    eyenose_heatmap, eyenose_croped = eyenose_detection(eyenose_window, img, face_croped)
    eyes_heatmap, eyes_croped = eyes_detection(eyes_window, img, eyenose_croped)
    tt = time() - t
    heatmaps = face_heatmap, eyenose_heatmap, eyes_heatmap
    crops = face_croped, eyenose_croped, eyes_croped
    return img, heatmaps, crops, tt


def face_detection(window, img):
    heatmap = compute_heatmap(img.shape[0:2], window(img, scale=FACE_SCALE))
    heatmap **= 2
    croped = crop_heatmap(heatmap, limit_ratio=0.4)

    return heatmap, croped


def eyenose_detection(window, img, croped):
    size = img.shape[0:2]
    try:
        region = next(r for r in regionprops(label(croped)))
    except StopIteration:
        return np.zeros(size, float), np.zeros(size, bool)

    y1, x1, y2, x2 = region.bbox
    img = img[y1:y2, x1:x2]
    shape = (y2 - y1, x2 - x1)

    heatmap = compute_heatmap(shape, window(img, scale=EYENOSE_SCALE))
    heatmap **= 2

    resized_heatmap = np.zeros(size, float)
    resized_heatmap[y1:y2, x1:x2] = heatmap

    croped = crop_heatmap(resized_heatmap, limit_ratio=0.4)

    return resized_heatmap, croped


def eyes_detection(window, img, croped):
    size = img.shape[0:2]
    try:
        region = next(r for r in regionprops(label(croped)))
    except StopIteration:
        return np.zeros(size, float), np.zeros(size, bool)

    y1, x1, y2, x2 = region.bbox
    img = img[y1:y2, x1:x2]
    shape = (y2 - y1, x2 - x1)

    heatmap = compute_heatmap(shape, window(img, scale=EYES_SCALE))
    heatmap **= 2

    resized_heatmap = np.zeros(size, float)
    resized_heatmap[y1:y2, x1:x2] = heatmap

    croped = crop_heatmap(resized_heatmap, limit_ratio=0.2)

    return resized_heatmap, croped


class Index:

    def __init__(self):
        self.ind = 0
        self.windows = (
            load_window('face'),
            load_window('eyenose'),
            load_window('eye'),
        )
        self.images = glob(IMAGES_PATH, recursive=True)
        shuffle(self.images)

    def setup_plot(self):
        self.fig, self.axes = plt.subplots(nrows=3, ncols=2)
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])

        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next)
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_prev.on_clicked(self.prev)

    def next(self, event):
        self.ind = (self.ind + 1) % len(self.images)
        self.clear()
        self.run_test()

    def prev(self, event):
        self.ind = (self.ind - 1) % len(self.images)
        self.clear()
        self.run_test()

    def clear(self):
        for x in range(3):
            for y in range(2):
                self.axes[x, y].clear()

    def run_test(self):
        img_path = self.images[self.ind]
        img, heatmaps, crops, tt = test(self.windows, img_path)
        face_heatmap, eyenose_heatmap, eyes_heatmap = heatmaps
        face_window, eyenose_window, eyes_window = self.windows

        def gen_rect(window, scale, color):
            return patches.Rectangle(
                (0, 0),
                *window.get_window_size(scale=scale),
                edgecolor=color,
                linewidth=1.0,
                facecolor="none"
            )

        self.fig.suptitle(f"time: {tt:6f}s")

        self.axes[0, 0].set_title("image")
        self.axes[0, 0].imshow(img)
        self.axes[0, 0].add_patch(gen_rect(face_window, FACE_SCALE, 'r'))
        self.axes[0, 0].add_patch(gen_rect(eyenose_window, EYENOSE_SCALE, 'y'))
        self.axes[0, 0].add_patch(gen_rect(eyes_window, EYES_SCALE, 'b'))

        self.axes[0, 1].set_title("image (gray)")
        self.axes[0, 1].imshow(rgb2gray(img), cmap='gray')

        self.axes[1, 0].set_title("face heatmap")
        self.axes[1, 0].imshow(face_heatmap, cmap="hot")

        self.axes[1, 1].set_title("eyenose heatmap")
        self.axes[1, 1].imshow(eyenose_heatmap, cmap="hot")

        self.axes[2, 0].set_title("eyes heatmap")
        self.axes[2, 0].imshow(eyes_heatmap, cmap="hot")

        self.axes[2, 1].set_title("regions")
        self.axes[2, 1].imshow(self._make_regions_image(img, heatmaps, crops))

        plt.draw()

    def _make_regions_image(self, img, heatmaps, crops):
        face_heatmap, eyenose_heatmap, eyes_heatmap = heatmaps
        face_croped, eyenose_croped, eyes_croped = crops
        img = gray2rgb(rgb2gray(img))
        regions = img.copy()
        if np.max(face_heatmap) != 0:
            regions[face_croped, :] = img[face_croped, :] * [1.0, 0.6, 0.6]
        if np.max(eyenose_heatmap) != 0:
            regions[eyenose_croped, :] = img[eyenose_croped, :] * [1.0, 1.0, 0.6]
        if np.max(eyes_heatmap) != 0:
            regions[eyes_croped, :] = img[eyes_croped, :] * [0.6, 0.6, 1.0]
        return regions



if __name__ == "__main__":
    index = Index()
    index.setup_plot()
    index.run_test()
    plt.show()
