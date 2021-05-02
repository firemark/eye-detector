import pickle
from sys import argv
from time import time

import cv2
import numpy as np

from skimage.transform import resize
from skimage.measure import label, regionprops

from eye_detector.model import load_window
from eye_detector.cam_func import init_win, del_win
from eye_detector.heatmap import compute_heatmap, crop_heatmap

FACE_SCALES = [4.0]
EYE_SCALES = [2.0]


def detect_and_generate_heatmap(frame, window, scale):
    size = frame.shape[0:2]
    heatmap = compute_heatmap(size, window(frame, scale=scale))
    return heatmap


def multiscale_detect(frame, window, scales):
    heatmap = sum(
        detect_and_generate_heatmap(frame, window, scale)
        for scale in scales
    )
    heatmap **= 2
    return heatmap


def detect_eyes(frame, window, croped, scales):
    size = frame.shape[0:2]

    try:
        region = next(r for r in regionprops(label(croped)))
    except StopIteration:
        return np.zeros(size, float)

    y1, x1, y2, x2 = region.bbox
    frame = frame[y1:y2, x1:x2]

    eye_heatmap = multiscale_detect(frame, window, scales)

    resized_heatmap = np.zeros(size, float)
    resized_heatmap[y1:y2, x1:x2] = eye_heatmap
    return resized_heatmap


def draw_heatmap(frame, croped, color=None):
    for region in regionprops(label(croped)):
        if region.area < 100:
            continue
        y1, x1, y2, x2 = region.bbox
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color or (0x00, 0x50, 0xFF),
            2,
        )


def main():
    face_window = load_window('face')
    eye_window = load_window('eye')
    cap = init_win()
    i = 0

    while True:
        ret, frame = cap.read()

        t = time()

        if i % 5 == 0:
            face_heatmap = multiscale_detect(frame, face_window, FACE_SCALES)
            face_croped = crop_heatmap(face_heatmap, limit_ratio=0.2)
            i = 1
        else:
            i += 1

        eye_heatmap = detect_eyes(frame, eye_window, face_croped, EYE_SCALES)
        eye_croped = crop_heatmap(eye_heatmap, limit_ratio=0.4)

        draw_heatmap(frame, face_croped, color=(0xFF, 0x50, 0x50))
        draw_heatmap(frame, eye_croped, color=(0x00, 0x50, 0xFF))
        print("time:", time() - t)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    del_win(cap)


if __name__ == "__main__":
    main()
