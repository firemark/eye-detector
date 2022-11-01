from time import time

import cv2

from eye_detector.cam_func import init_win, del_win
from eye_detector.capture_dlib.models import EnrichedModel
from eye_detector.capture_dlib.loop import loop


def main():
    cap = init_win()
    model = EnrichedModel()

    cap.start()

    while True:
        t0 = time()
        frame = loop(model, cap)
        t1 = time()
        print(f"time: {(t1 - t0) * 1e3:0.3f}ms")

        cv2.imshow("frame", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    del_win(cap)


if __name__ == "__main__":
    main()
