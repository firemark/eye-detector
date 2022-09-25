import pickle
from sys import argv
from time import time
from math import degrees, tan
from collections import deque

import cv2
import numpy as np
import torch
import dlib
from scipy.spatial.transform import Rotation

from eye_detector.cam_func import init_win, del_win, draw_it
from eye_detector.eye_data_conv_dlib import Model
from eye_detector.model import load_model
from eye_detector import pupil_coords


class EnrichedModel(Model):

    def __init__(self):
        super().__init__()
        self.pupil_coords_model = load_model("eye-pupil")


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
    #draw_rect(36, 39, color=(0x50, 0x50, 0xFF))
    # right eye
    draw_landmarks_range(42, 48, color=(0xFF, 0x50, 0x50))
    #draw_rect(42, 45, color=(0xFF, 0x50, 0x50))


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


def _land(func, landmarks, slice):
    if isinstance(func, tuple):
        func_x, func_y = func
    else:
        func_x = func
        func_y = func

    start, stop = slice
    points = [landmarks.part(index) for index in range(start, stop)]
    return func_x(p.x for p in points), func_y(p.y for p in points)


def draw_text(image, text, p, scale=1):
    cv2.putText(image, text, p, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, 255)


def draw_pupil_coords(frame, eye_xy, pupil_xy, radius):
    if eye_xy is None:
        return
    cv2.circle(frame, eye_xy, 3, (0x99, 0x99, 0x99), cv2.FILLED)
    if pupil_xy is None:
        return
    cv2.line(frame, eye_xy, pupil_xy, (0xFF, 0xFF, 0xFF), 1)


class EyeCache:
    tx = 0
    ty = 0
    c = 0

    def __init__(self, deque_size=5):
        self.x = deque(maxlen=deque_size)
        self.y = deque(maxlen=deque_size)

    def update(self, x, y):
        self.tx += x
        self.ty += y
        self.c += 1
        if self.c >= 1:
            self.x.append(self.tx / self.c)
            self.y.append(self.ty / self.c)
            self.tx = 0
            self.ty = 0
            self.c = 0

class ScreenBox:

    def __init__(self, leftdown_position, width, height, roll=0.0, pitch=0.0, yaw=0.0):
        euler_angles = np.array([yaw, pitch, roll])
        rotation = Rotation.from_euler('xyz', euler_angles, degrees=True)
        position = leftdown_position + [width / 2, height / 2, 0]
        points = rotation.apply(np.array([
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [0.0, height, 0.0],
        ])) + position

        self.position = position
        self.width = width
        self.height = height
        self.normal = np.cross(points[1] - points[0], points[2] - points[1])
        self.plane_offset = -self.normal.dot(points[0]) # D parameter
        self.inv_rotation = rotation.inv()
        print(self.inv_rotation.as_matrix(), self.position, self.plane_offset, self.normal)

    def intersect(self, direction, position):
        # We have equations:
        # (x,y,z) = direction * t + position
        # x = direction[0] + t + position[0]
        # y = direction[1] + t + position[1]
        # z = direction[2] + t + position[2]
        # Ax + Bx + Cy + D = 0
        # normal = (A, B, C)
        # So result is t = -(normal.dot(position) + D) / normal.dot(direction)
        # And we need put computed t to (x,y,z) equations
        normal = self.normal
        divisor = normal.dot(direction)
        if divisor == 0.0:
            return None

        dividend = -(normal.dot(position) + self.plane_offset)
        t = dividend / divisor
        if t <= 0.0:
            return None

        global_xyz = direction * t + position
        local_xyz = self.inv_rotation.apply(global_xyz)# + self.position
        return local_xyz[:2] + [self.width / 2, self.height / 2]


eyecache_left = EyeCache()
eyecache_right = EyeCache()
screen_box = ScreenBox(
    leftdown_position=np.array([0.0, 0.0, 0.0]),
    #leftdown_position=np.array([-0.27, -0.03, -0.05]),
    width=1.0, # 0.57
    height=0.5, # 0.32
)


def draw_text_pupil_coords(frame, prefix, shift, eye_xy, pupil_xy, radius, eye_xyz, eyecache):
    if eye_xy is None:
        return
    if pupil_xy is None:
        return
    eye_dx = eye_xy[0] - pupil_xy[0]
    eye_dy = eye_xy[1] - pupil_xy[1]

    x_angle = -pupil_coords.compute_angle(eye_dx, radius)
    y_angle = -pupil_coords.compute_angle(eye_dy, radius)
    #deg_x = degrees(x_angle)
    #deg_y = degrees(y_angle)
    #deg_x = dx / radius * 100.0
    #deg_y = dy / radius * 100.0

    dz = -eye_xyz[2]
    direction = np.array([dz * tan(x_angle) * 3.14, dz * tan(y_angle), dz])
    #direction = np.array([0.0, 0.0, dz])
    local_xy = screen_box.intersect(direction, eye_xyz)

    if local_xy is not None:
        eyecache.update(local_xy[0], local_xy[1])

    for i, (x, y) in enumerate(zip(eyecache.x, eyecache.y)):
        if x < 0 or x > screen_box.width:
            continue
        if y < 0 or y > screen_box.height:
            continue
        pixel_x = int(x * frame.shape[1] / screen_box.width)
        pixel_y = int(y * frame.shape[0] / screen_box.height)
        cv2.circle(frame, (pixel_x, pixel_y), 5, (0x0, 0x00, int(0xff / len(eyecache.x) * i)), cv2.FILLED)

    if len(eyecache.x) > 0:
        text_xy = (30, shift)
        draw_text(frame, f"{prefix}: {eyecache.x[-1]:+08.3f}, {eyecache.y[-1]:+08.3f}", text_xy)


def get_index(eye, size, model: EnrichedModel) -> int:
    h, w = size
    eye_xy, pupil_xy, *_ = eye
    if not eye_xy  or not pupil_xy:
        return None
    eye_x = eye_xy[0] / w - 0.5
    eye_y = eye_xy[1] / h - 0.5
    pupil_x = (pupil_xy[0] - eye_xy[0]) / w
    pupil_y = (pupil_xy[1] - eye_xy[1]) / h
    row = [eye_x, eye_y, pupil_x, pupil_y]
    outputs = model.pupil_coords_model.predict([row])
    return outputs[0]


def loop(model: EnrichedModel, cap):
    color_frame, depth_frame = cap.get_frames()

    landmarks = model.detect_and_get_landmarks(color_frame)
    if landmarks is None:
        return color_frame

    left = pupil_coords.get_left_coords(color_frame, model, landmarks)
    right = pupil_coords.get_right_coords(color_frame, model, landmarks)

    left_3d = cap.to_3d(left[0], depth_frame[int(left[0][1]), int(left[0][0])])
    right_3d = cap.to_3d(right[0], depth_frame[int(right[0][1]), int(right[0][0])])
    #index = get_index(left, frame.shape[0:2], model)

    #frame[ly, lx][lmask] = (0x00, 0x00, 0xFF)
    #frame[ry, rx][rmask] = (0xFF, 0x00, 0xFF)
    #if index is not None:
    #    draw_it(frame, index)

    draw_landmarks(color_frame, landmarks)
    draw_pupil_coords(color_frame, *left)
    draw_pupil_coords(color_frame, *right)
    draw_text_pupil_coords(color_frame, "left ", 25, *left, left_3d, eyecache_left)
    draw_text_pupil_coords(color_frame, "right", 50, *right, right_3d, eyecache_right)
    #draw_text(frame, f"index: {index}", (0, 75))
    return color_frame


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
