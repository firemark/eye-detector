from time import time
from glob import glob
from random import shuffle
from csv import writer

import numpy as np
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.draw import line
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from eye_detector.eye_data_conv_dlib import Model
from eye_detector import pupil_coords

LABELS_PATH = "indata/to_label/*"

CONFIGS = [
    {"index": 0, "color": "red"},
    {"index": 1, "color": "blue"},
    {"index": 2, "color": "green"},
    {"index": 3, "color": "yellow"},
]


def compute(model, img_path):
    img = imread(img_path)
    img = img[:, :, 0:3]

    landmarks = model.detect_and_get_landmarks(img)
    if landmarks is None:
        return None

    left, lx, ly = model.get_left_eye(img, landmarks)
    right, rx, ry = model.get_right_eye(img, landmarks)

    mask = pupil_coords.to_mask(img, model, landmarks)
    left_mask = model.crop_from_rect(mask, lx, ly).astype(bool)
    right_mask = model.crop_from_rect(mask, rx, ry).astype(bool)

    lx = lx.start
    ly = ly.start
    rx = rx.start
    ry = ry.start

    left_centroid = pupil_coords.get_centroid(left_mask, lx, ly)
    right_centroid = pupil_coords.get_centroid(right_mask, rx, ry)

    left_pupil_mask = pupil_coords.to_pupil_mask(left, left_mask)
    right_pupil_mask = pupil_coords.to_pupil_mask(right, right_mask)

    left_coords = pupil_coords.get_centroid(left_pupil_mask, lx, ly)
    right_coords = pupil_coords.get_centroid(right_pupil_mask, rx, ry)
    if left_centroid:
        left_points = model.get_left_eye_points(landmarks)
        left_radius = pupil_coords.compute_radius(left_centroid, left_points[3])
    else:
        left_radius = None

    return (img.shape[0:2], left_centroid, left_coords, left_radius)


def compute_dir(model, dirname, config, global_config):
    t = time()
    print(dirname, config, "...", end=" ", flush=True)

    for i, filename in enumerate(glob(f"{dirname}/*.png", recursive=True)):
        #if i % 8 != 0:
        #    continue
        d = compute(model, filename)
        if d is None:
            continue
        size, eye, pupil, radius = d
        if not eye or not pupil:
            continue
        row = pupil_coords.compute_row(size, eye, pupil, radius)
        global_config["csv"].writerow((config['index'],) + row)
        plt.arrow(
            *row,
            width=0.0005,
            head_width=0.004,
            color=config["color"],
        )

    tt = time() - t
    print("time:", tt)


if __name__ == "__main__":
    model = Model()
    with open("eye_coords_map.csv", "w") as file:
        fields = ["index", "eye_x", "eye_y", "pupil_x", "pupil_y"]
        global_config = {"csv": writer(file)}
        global_config["csv"].writerow(fields)
        for index, dirname in enumerate(glob(LABELS_PATH)):
            compute_dir(model, dirname, CONFIGS[index], global_config)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.show()
