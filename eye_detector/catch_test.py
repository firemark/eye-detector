from time import time
from glob import glob
from random import shuffle

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import random_noise
from skimage.io import imread
from skimage.measure import label, regionprops
from matplotlib import patches, pyplot as plt
from matplotlib.widgets import Button

from eye_detector.heatmap import compute_heatmap, crop_heatmap
from eye_detector.model import load_window


def test(windows, img_path):
    img = imread(img_path)
    img = img[:, :, 0:3]
    img = random_noise(img, var=0.001)

    t = time()
    face_window, eyes_window = windows
    face_heatmap, face_croped = face_detection(face_window, img)
    eyes_heatmap, eyes_croped = eyes_detection(eyes_window, img, face_croped)
    tt = time() - t
    heatmaps = face_heatmap, eyes_heatmap,
    crops = face_croped, eyes_croped
    return img, heatmaps, crops, tt


def face_detection(window, img):
    heatmap = compute_heatmap(img.shape[0:2], window(img, scale=2))
    heatmap **= 2
    croped = crop_heatmap(heatmap, limit_ratio=0.05)

    return heatmap, croped


def eyes_detection(window, img, croped):
    size = img.shape[0:2]
    try:
        region = next(r for r in regionprops(label(croped)))
    except StopIteration:
        return np.zeros(size, float), np.zeros(size, bool)

    x1, y1, x2, y2 = region.bbox
    img = img[y1:y2, x1:x2]
    shape = (y2 - y1, x2 - x1)

    heatmap = compute_heatmap(shape, window(img, scale=2.0))
    heatmap **= 2

    resized_heatmap = np.zeros(size, float)
    resized_heatmap[y1:y2, x1:x2] = heatmap

    croped = crop_heatmap(resized_heatmap, limit_ratio=0.5)

    return resized_heatmap, croped


class Index:

    def __init__(self):
        self.ind = 0
        self.windows = (
            load_window('face'),
            load_window('eye'),
        )
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
        img_path = self.images[self.ind]
        img, heatmaps, crops, tt = test(self.windows, img_path)
        face_heatmap, eyes_heatmap = heatmaps
        face_croped, eyes_croped = crops
        face_max_h = np.max(face_heatmap)
        eyes_max_h = np.max(eyes_heatmap)

        face_window, eyes_window = self.windows
        fw, fh = face_window.get_window_size(scale=2.0)
        ew, eh = eyes_window.get_window_size(scale=2.0)

        face_rect = patches.Rectangle(
            (0, 0),
            fw, fh,
            edgecolor='r',
            linewidth=3.0,
            facecolor="none"
        )

        eyes_rect = patches.Rectangle(
            (0, 0),
            ew, eh,
            edgecolor='b',
            linewidth=3.0,
            facecolor="none"
        )

        self.fig.suptitle(f"time: {tt:6f}s")

        self.axes[0, 0].set_title("image")
        self.axes[0, 0].imshow(img)
        self.axes[0, 0].add_patch(face_rect)
        self.axes[0, 0].add_patch(eyes_rect)

        self.axes[0, 1].set_title("face heatmap")
        if face_max_h != 0:
            self.axes[0, 1].imshow(face_heatmap / face_max_h, cmap="hot")
        else:
            self.axes[0, 1].imshow(face_heatmap, cmap="hot")

        self.axes[1, 0].set_title("eyes heatmap")
        if eyes_max_h != 0:
            self.axes[1, 0].imshow(eyes_heatmap / eyes_max_h, cmap="hot")
        else:
            self.axes[1, 0].imshow(eyes_heatmap, cmap="hot")

        if face_max_h != 0:
            img[face_croped, :] *= [1.0, 0.6, 0.6]
        if eyes_max_h != 0:
            img[eyes_croped, :] *= [0.3, 1.0, 0.3]
        self.axes[1, 1].set_title("regions")
        self.axes[1, 1].imshow(img)
        plt.draw()


if __name__ == "__main__":
    index = Index()
    index.setup_plot()
    index.run_test()
    plt.show()
