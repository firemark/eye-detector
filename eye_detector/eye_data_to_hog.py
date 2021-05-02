from skimage.transform import resize
from glob import glob
from shutil import rmtree
from os import mkdir
from os.path import basename
import pickle

import numpy as np
from numpy import concatenate, uint8
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.measure import label, regionprops

from eye_detector.model import load_window
from eye_detector.heatmap import crop_heatmap
from eye_detector.const import CLASSES

from eye_detector.capture import (
    multiscale_detect,
    detect_eyes,
)


FACE_SCALES = [4.0]
EYE_SCALES = [2.0]


def get_img_from_bbox(img, region):
    y1, x1, y2, x2 = region.bbox
    eye = img[y1:y2, x1:x2]
    eye = resize(eye, (64, 64 * 3))
    left_eye = eye[:, :64]
    right_eye = eye[:, -64:]
    return concatenate([left_eye, right_eye], axis=1)


def generate_and_save(klass, filepath, face_window, eye_window):
    img = imread(filepath)[:, :, 0:3]

    face_heatmap = multiscale_detect(img, face_window, FACE_SCALES)
    if np.sum(face_heatmap) == 0:
        return False
    face_croped = crop_heatmap(face_heatmap, limit_ratio=0.2)
    eye_heatmap = detect_eyes(img, eye_window, face_croped, EYE_SCALES)
    if np.sum(eye_heatmap) == 0:
        return False
    eye_croped = crop_heatmap(eye_heatmap, limit_ratio=0.4)
    regions = regionprops(label(eye_croped))

    if len(regions) != 1:
        return False

    eyenose_img = get_img_from_bbox(img, regions[0])

    name = basename(filepath)
    img_to_save =  eyenose_img * 0xFF
    imsave(f"middata/transformed_label/{klass}/{name}", img_to_save.astype(uint8))

    return True


if __name__ == "__main__":
    rmtree("middata/transformed_label", ignore_errors=True)
    mkdir("middata/transformed_label")

    face_window = load_window('face')
    eye_window = load_window('eye')

    tot = 0
    succ = 0

    for klass in CLASSES:
        mkdir(f"middata/transformed_label/{klass}")
        for filepath in glob(f"indata/to_label/{klass}/*.png"):
            is_succ = generate_and_save(klass, filepath, face_window, eye_window)
            if is_succ:
                print("\033[92m.\033[0m", flush=True, end="")
                succ +=  1
            else:
                print("\033[91mX\033[0m", flush=True, end="")
            tot += 1

    print()
    print("---")
    print("tot", tot)
    print("succ", succ)

