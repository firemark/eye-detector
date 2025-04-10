from datetime import datetime
from os import makedirs
from random import random

import cv2

from eye_detector.const import CLASSES, ROWS
from eye_detector.const import W, H, PW, PH
from eye_detector.cam_func import init_win, del_win, draw_camera

it = 0
record = False


def init(dirname):
    makedirs(dirname, exist_ok=True)
    for c in CLASSES:
        makedirs(f"{dirname}/{c}", exist_ok=True)


init('indata/to_label')
init('indata/to_label_test')
cap = init_win()

while True:
    ret, frame = cap.read()
    draw_camera(
        frame,
        it,
        border_color=(0,0xFF,0) if record else (0xFF,0,0),
    )
    key = cv2.waitKey(10) & 0xFF
    if record:
        timestamp = int(datetime.now().timestamp() * 1000)
        c = CLASSES[it % len(CLASSES)]
        dirname = "indata/to_label" if random() > 0.2 else "indata/to_label_test"
        cv2.imwrite(f"{dirname}/{c}/{timestamp}.png", frame)
    if key == ord('q'):
        break
    if key == ord('r'):
        record = not record
    elif key == ord('s'):
        record = False
        it = (it + 1) % len(CLASSES)


del_win(cap)
