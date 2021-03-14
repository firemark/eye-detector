import pickle
from sys import argv
from time import time

import cv2

from skimage.transform import resize
from skimage.measure import label, regionprops

from eye_detector.model import load_window
from eye_detector.cam_func import init_win, del_win
from eye_detector.heatmap import compute_heatmap, crop_heatmap


def detect_and_generate_heatmap(frame, window, scale):
    height, width = frame.shape[0:2]
    size = (int(width / scale), int(height / scale))
    resized_frame = cv2.resize(frame, size)
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    heatmap = compute_heatmap(gray.shape[::-1], window(gray))
    return resize(heatmap, (width, height))


def draw_heatmap(heatmap, color=None):
    croped = crop_heatmap(heatmap, 0.7)
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
    SCALES = [1.5, 2.0]

    window = load_window()
    cap = init_win()

    while True:
        ret, frame = cap.read()

        t = time()
        heatmap = sum(
            detect_and_generate_heatmap(frame, window, scale)
            for scale in SCALES
        )
        draw_heatmap(heatmap)
        print("time:", time() - t)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(500) & 0xFF
        if key == ord('q'):
            break

    del_win(cap)
