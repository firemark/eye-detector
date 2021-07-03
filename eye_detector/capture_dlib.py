import pickle
from sys import argv
from time import time

import cv2
import numpy as np
import torch
import dlib

from eye_detector.cam_func import init_win, del_win, draw_it


def draw_face(frame, face, color):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )


def draw_landmarks(frame, landmarks):
    if not landmarks:
        return

    def draw_landmarks_range(start, stop, color):
        for index in range(start, stop):
            point = landmarks.part(index)
            cv2.circle(frame, (point.x, point.y), 2, color, -1)

    def draw_rect(left_index, right_index, color):
        ll_p = landmarks.part(left_index)
        lr_p = landmarks.part(right_index)

        size = lr_p.x - ll_p.x
        size_a = int(size * 0.8)
        half = size // 2
        cx = (ll_p.x + lr_p.x) // 2
        cy = (ll_p.y + lr_p.y) // 2

        cv2.rectangle(
            frame,
            (cx - size_a, cy - half),
            (cx + size_a, cy + half),
            (0x50, 0x50, 0xFF),
            2,
        )

    # jawline
    draw_landmarks_range(0, 17, color=(0x20, 0xFF, 0xFF))
    # nose bridge
    draw_landmarks_range(27, 31, color=(0x20, 0x88, 0xFF))
    # left eye
    draw_landmarks_range(36, 42, color=(0x50, 0x50, 0xFF))
    draw_rect(36, 39, color=(0x50, 0x50, 0xFF))
    # right eye
    draw_landmarks_range(42, 48, color=(0xFF, 0x50, 0x50))
    draw_rect(42, 45, color=(0xFF, 0x50, 0x50))


def draw_cross(frame, landmarks):
    mean = lambda p: int(sum(p) / 6)
    left = _land(mean, landmarks, (36, 42))
    right = _land(mean, landmarks, (42, 48))
    point_up = landmarks.part(27)
    point_down = landmarks.part(30)

    up = (point_up.x, point_up.y)
    down = (point_down.x, point_down.y)

    cv2.line(frame, left, right, (0x10, 0xFF, 0x20), 1)
    cv2.line(frame, up, down, (0xC0, 0xFF, 0x20), 1)


_saved_matrix = None
def get_camera_matrix(frame):
    global _saved_matrix

    dist_coeffs = np.zeros((4,1), dtype="double") # Assuming no lens distortion

    if _saved_matrix is not None:
        return _saved_matrix, dist_coeffs

    h, w = frame.shape[0:2]
    f = w
    camera_matrix = np.array([
        [f, 0, h / 2],
        [0, f, w / 2],
        [0, 0, 1],
    ], dtype="double")
    _saved_matrix = camera_matrix

    return camera_matrix, dist_coeffs


def compute_pose(frame, landmarks):
    # https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    image_points = np.array([
        (p.x, p.y) for p in [
            landmarks.part(30),  # Nose tip
            landmarks.part(8),   # Chin
            landmarks.part(36),  # Left eye left corner
            landmarks.part(45),  # Right eye right corner
            landmarks.part(48),  # Left Mouth corner
            landmarks.part(54),  # Right Mouth corner
        ]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0),     # Right Mouth corner
    ])

    camera_matrix, dist_coeffs = get_camera_matrix(frame)
    (success, rot_vec, pos_vec) = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        # flags=cv2.CV_ITERATIVE,
    )

    if not success:
        return None

    return rot_vec, pos_vec


def draw_pose(frame, landmarks, pose):
    rot_vec, pos_vec = pose
    camera_matrix, dist_coeffs = get_camera_matrix(frame)
    points, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 500.0)]),
        rot_vec,
        pos_vec,
        camera_matrix,
        dist_coeffs,
    )
    point = landmarks.part(30)
    start = (point.x, point.y)
    stop = tuple(points[0, 0].astype(int))
    cv2.line(frame, start, stop, (0, 0, 0), 3)


def _land(func, landmarks, slice):
    if isinstance(func, tuple):
        func_x, func_y = func
    else:
        func_x = func
        func_y = func

    start, stop = slice
    points = [landmarks.part(index) for index in range(start, stop)]
    return func_x(p.x for p in points), func_y(p.y for p in points)



def main():
    cap = init_win()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    i = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t = time()

        for face in detector(gray):
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)
            draw_cross(frame, landmarks)
            pose = compute_pose(frame, landmarks)
            if pose is not None:
                draw_pose(frame, landmarks, pose)

        print("time:", time() - t)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    del_win(cap)


if __name__ == "__main__":
    main()
