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

SCALES = [4.0]


def detect_and_generate_heatmap(frame, window, scale):
    height, width = frame.shape[0:2]
    size = (int(width / scale), int(height / scale))
    resized_frame = cv2.resize(frame, size)
    rh, rw = resized_frame.shape[0:2]
    heatmap = compute_heatmap((rw, rh), window(resized_frame))
    return resize(heatmap, (width, height))


def detect_eyes(frame, window, croped, scale):
    size = frame.shape[0:2]

    try:
        region = next(r for r in regionprops(label(croped)))
    except StopIteration:
        return np.zeros(size, float)

    x1, y1, x2, y2 = region.bbox
    frame = frame[y1:y2, x1:x2]
    eye_heatmap = detect_and_generate_heatmap(frame, window, scale)

    resized_heatmap = np.zeros(size, float)
    resized_heatmap[x1:x2, y1:y2] = eye_heatmap
    return resized_heatmap


def draw_heatmap(croped, color=None):
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


if __name__ == "__main__":
    face_window = load_window('face')
    eye_window = load_window('eye')
    cap = init_win()
    i = 0

    while True:
        ret, frame = cap.read()

        t = time()

        if i % 5 == 0:
            face_heatmap = sum(
                detect_and_generate_heatmap(frame, face_window, scale)
                for scale in SCALES
            )
            face_heatmap **= 2
            face_croped = crop_heatmap(face_heatmap, limit_ratio=0.2)
            i = 1
        else:
            i += 1

        eye_heatmap = detect_eyes(frame, eye_window, face_croped, scale=2.0)
        eye_heatmap **= 2
        eye_croped = crop_heatmap(eye_heatmap, limit_ratio=0.4)

        draw_heatmap(face_croped, color=(0xFF, 0x50, 0x50))
        draw_heatmap(eye_croped, color=(0x00, 0x50, 0xFF))
        print("time:", time() - t)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    del_win(cap)
