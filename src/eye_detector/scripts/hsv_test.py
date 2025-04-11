from time import time
from glob import glob
from random import shuffle

import numpy as np
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.draw import line
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from eye_detector.dlib_model import Model
from eye_detector import pupil_coords

IMAGES_PATH = "indata/to_label/*/*.png"
#IMAGES_PATH = "indata/to_label/*/1621364139545.png"


def test(model, img_path):
    img = imread(img_path)
    img = img[:, :, 0:3]

    t = time()
    landmarks = model.detect_and_get_landmarks(img)
    if landmarks is None:
        return None

    left, lx, ly = model.get_left_eye(img, landmarks)
    right, rx, ry = model.get_right_eye(img, landmarks)

    mask = pupil_coords.to_mask(img, model, landmarks)
    left_mask = model.crop_from_rect(mask, lx, ly).astype(bool)
    right_mask = model.crop_from_rect(mask, rx, ry).astype(bool)

    left_centroid = pupil_coords.get_centroid(left_mask)
    right_centroid = pupil_coords.get_centroid(right_mask)

    left_pupil_mask = pupil_coords.to_pupil_mask(left, left_mask)
    right_pupil_mask = pupil_coords.to_pupil_mask(right, right_mask)

    left_coords = pupil_coords.get_centroid(left_pupil_mask)
    right_coords = pupil_coords.get_centroid(right_pupil_mask)

    tt = time() - t

    # DEBUG
    check_eye_img = model.concat(left, right)
    hsv = rgb2hsv(check_eye_img)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # left[line(*left_centroid, *left_coords)] += (0.0, 0.0, 1.0)
    # if left_coords is not None:
    #     left[left_coords] += (1.0, 0.0, 0.0)
    left[left_mask] += (0.2, 0.0, 0.0)

    # right[line(*right_centroid, *right_coords)] += (0.0, 0.0, 1.0)
    # if right_coords is not None:
    #     right[right_coords] += (1.0, 0.0, 0.0)
    right[right_mask] += (0.2, 0.0, 0.0)

    eye_img = model.concat(left, right)
    pupil_mask = model.concat(left_pupil_mask, right_pupil_mask)
    eye_img[pupil_mask] += (0.1, 0.3, 0.1)

    return check_eye_img, sat, val, eye_img, tt


class Index:

    def __init__(self):
        self.ind = 0
        self.model = Model()
        self.images = glob(IMAGES_PATH, recursive=True)
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
        res = test(self.model, img_path)
        if res is None:
            self.fig.suptitle(f"EYES NOT FOUND")
            x = np.zeros((10, 10, 3))
            self.axes[0, 0].imshow(x)
            self.axes[0, 1].imshow(x)
            self.axes[1, 0].imshow(x)
            self.axes[1, 1].imshow(x)
            return

        rgb, hue, sat, val, tt = res

        self.fig.suptitle(f"time: {tt:6f}s")

        self.axes[0, 0].set_title("image")
        self.axes[0, 0].imshow(rgb)

        self.axes[0, 1].set_title("saturate")
        self.axes[0, 1].imshow(hue, cmap='binary_r')

        self.axes[1, 0].set_title("value")
        self.axes[1, 0].imshow(sat, cmap='binary_r')

        self.axes[1, 1].set_title("mask")
        self.axes[1, 1].imshow(val, cmap='binary_r')

        plt.draw()


if __name__ == "__main__":
    index = Index()
    index.setup_plot()
    index.run_test()
    plt.show()
