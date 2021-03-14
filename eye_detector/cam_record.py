from datetime import datetime
from os import makedirs
from random import random

import cv2

from const import CLASSES, ROWS
from const import W, H, PW, PH
from cam_func import init_win, del_win, draw_camera

it = 0
record = False


def init(dirname):
    makedirs(dirname, exist_ok=True)
    for c in CLASSES:
        makedirs(f"{dirname}/{c}", exist_ok=True)


init('data')
init('testdata')
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
        dirname = "data" if random() > 0.2 else "testdata"
        cv2.imwrite(f"{dirname}/{c}/{timestamp}.png", frame)
    if key == ord('q'):
        break
    if key == ord('r'):
        record = not record
    elif key == ord('s'):
        record = False
        it = (it + 1) % len(CLASSES)


del_win(cap)
