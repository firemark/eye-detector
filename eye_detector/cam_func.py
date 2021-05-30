import cv2

from eye_detector.const import ROWS, PW, PH, W, H


def init_win(title='frame'):
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    return cv2.VideoCapture(4)
    return cv2.VideoCapture(0)


def del_win(cap):
    cap.release()
    cv2.destroyAllWindows()


def draw_camera(frame, it, title='frame', border_color=255):
    show_frame = cv2.resize(frame, (W, H))
    #cv2.putText(
    #    show_frame,
    #    f"Frame {i:03d}",
    #    (PW * x, PH * y + 50),
    #    cv2.FONT_HERSHEY_SIMPLEX,
    #    1,
    #    255,
    #)

    cv2.imshow(title, show_frame)


def draw_it(frame, it):
    h, w = frame.shape[0:2]
    x = it % ROWS
    y = it // ROWS
    pw = w // 2
    ph = h // 2
    cv2.rectangle(
        frame,
        (pw * x, ph * y),
        (pw * (x + 1), ph * (y + 1)),
        (0xFF, 0xFF, 0xFF),
        5,
    )
