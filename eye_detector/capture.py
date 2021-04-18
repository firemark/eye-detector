import pickle
from sys import argv
from time import time

import cv2

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


def draw_heatmap(heatmap, color=None, limit_ratio=0.5):
    croped = crop_heatmap(heatmap, limit_ratio)
    for region in regionprops(label(croped)):
        if region.area < 100:
            continue
        y1, x1, y2, x2 = region.bbox
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color or (0, 0x50, 0xFF),
            2,
        )


if __name__ == "__main__":
    face_window = load_window('face')
    cap = init_win()

    while True:
        ret, frame = cap.read()

        t = time()
        heatmap = sum(
            detect_and_generate_heatmap(frame, face_window, scale)
            for scale in SCALES
        )
        heatmap **= 2
        draw_heatmap(heatmap, limit_ratio=0.2)
        print("time:", time() - t)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    del_win(cap)
