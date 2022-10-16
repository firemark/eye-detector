from dataclasses import dataclass
from time import time
from math import degrees, tan
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
import dlib
from scipy.spatial.transform import Rotation

from eye_detector.cam_func import init_win, del_win, draw_it
from eye_detector.eye_data_conv_dlib import Model
from eye_detector.model import load_model
from eye_detector import pupil_coords
from eye_detector.pupil_coords import EyeCoords


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


def draw_pupil_coords(frame, eye_coords):
    if eye_coords.eye_centroid is None:
        return
    cv2.circle(frame, eye_coords.eye_centroid, 3, (0x99, 0x99, 0x99), cv2.FILLED)

    if eye_coords.pupil_centroid is None:
        return
    cv2.line(frame, eye_coords.eye_centroid, eye_coords.pupil_centroid, (0xFF, 0xFF, 0xFF), 1)


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
        if self.c >= 2:
            self.x.append(self.tx / self.c)
            self.y.append(self.ty / self.c)
            self.tx = 0
            self.ty = 0
            self.c = 0


class ScreenBox:

    def __init__(self, leftdown_position, width, height, roll=0.0, pitch=0.0, yaw=0.0):
        euler_angles = np.array([yaw, pitch, roll])
        rotation = Rotation.from_euler('xyz', euler_angles, degrees=True)
        center_position = leftdown_position + [width / 2, height / 2, 0]
        points = rotation.apply(np.array([
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [0.0, height, 0.0],
        ])) + center_position

        self.center_position = center_position
        self.leftdown_position = leftdown_position
        self.width = width
        self.height = height
        self.normal = np.cross(points[1] - points[0], points[2] - points[1])
        self.plane_offset = -self.normal.dot(points[0])  # D parameter
        self.inv_rotation = rotation.inv()

        print(self.center_position)

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
        local_xyz = self.inv_rotation.apply(global_xyz) + self.center_position
        return local_xyz[:2] + [self.width / 2, self.height / 2]


eyecache_left = EyeCache()
eyecache_right = EyeCache()
screen_box = ScreenBox(
    #leftdown_position=np.array([0.0, 0.0, 0.0]),
    leftdown_position=np.array([-0.27, -0.03, -0.05]),
    width=0.57, # 0.57
    height=0.32, # 0.32
)


def angles_to_direction_vector(angles_xy):
    return np.array([tan(angles_xy[0]), tan(angles_xy[1]), 1])


def rotate_normal_by_angles(normal, angles_xy):
    return Rotation.from_euler('xy', angles_xy).apply(normal)


def to_unit_vector(vec: np.array):
    if vec is None:
        return vec
    return vec / np.linalg.norm(vec)


def point_to_screen(frame, x, y):
    if x < 0 or x > screen_box.width:
        return None
    if y < 0 or y > screen_box.height:
        return None
    pixel_x = int(x * frame.shape[1] / screen_box.width)
    pixel_y = int(y * frame.shape[0] / screen_box.height)
    return (pixel_x, pixel_y)


def draw_text_pupil_coords(frame, prefix, shift, color: np.ndarray, eyecache: EyeCache):
    for i, (x, y) in enumerate(zip(eyecache.x, eyecache.y)):
        screen_point = point_to_screen(frame, x, y)
        if screen_point is None:
            continue
        blend_color = (color * i / len(eyecache.x)).astype(int).tolist()
        cv2.circle(frame, screen_point, 5, blend_color, cv2.FILLED)

    if len(eyecache.x) > 0:
        text_xy = (30, shift)
        draw_text(frame, f"{prefix}: {eyecache.x[-1]:+08.3f}, {eyecache.y[-1]:+08.3f}", text_xy)


def draw_3d_vec(frame, cap, direction_xyz, point_xyz, length, color):
    direction_vec = to_unit_vector(direction_xyz) * length
    point_a = cap.from_3d(point_xyz)
    point_b = cap.from_3d(point_xyz + direction_vec)

    if np.any(np.isnan(point_a)) or np.any(np.isnan(point_b)):
        return

    cv2.line(frame, point_a.astype(int), point_b.astype(int), color, 2)


def compute_face_normal(landmarks, cap, depth_frame):
    #return np.array([0.0, 0.0, -1.0])
    points = [
        cap.to_3d(np.array([p.x, p.y]), depth_frame) for p in [
            landmarks.part(8),  # Chin
            landmarks.part(45),  # Right eye right corner
            landmarks.part(36),  # Left eye left corner
        ]
    ]

    if any(p is None for p in points):
        return None

    return np.cross(points[1] - points[0], points[2] - points[1])


@dataclass
class Eye3D:
    eye_xyz: np.array
    pupil_xyz: np.array
    direction: np.array


def draw_angles(frame, prefix, shift, direction):
    camera_normal = np.array([0.0, 0.0, -1.0])
    xz_vec = camera_normal[[0, 2]] - direction[[0, 2]]
    yz_vec = camera_normal[[1, 2]] - direction[[1, 2]]
    yaw_distance = np.linalg.norm(xz_vec)
    pitch_distance = np.linalg.norm(yz_vec)

    yaw = 2 * np.arcsin(yaw_distance / 2.0) * (1 if xz_vec[0] < 0 else -1)
    pitch = 2 * np.arcsin(pitch_distance / 2.0) * (1 if yz_vec[0] < 0 else -1)

    #print(prefix, degrees(yaw), degrees(pitch))

    text_xy = (30, shift)
    draw_text(frame, f"{prefix}: {degrees(yaw):+08.3f}, {degrees(pitch):+08.3f}", text_xy)


def update_pointer_coords(eyecache: EyeCache, eye_3d: Optional[Eye3D]):
    if eye_3d is None:
        return

    local_xy = screen_box.intersect(eye_3d.direction, eye_3d.pupil_xyz)
    if local_xy is None:
        return

    eyecache.update(local_xy[0], local_xy[1])


def compute_eye_3d(cap, depth_frame, face_normal, eye_coords: EyeCoords) -> Optional[Eye3D]:
    if face_normal is None or eye_coords.eye_centroid is None or eye_coords.pupil_centroid is None:
        return None

    eye_xyz = cap.to_3d(eye_coords.eye_centroid, depth_frame)
    pupil_xyz = cap.to_3d(eye_coords.pupil_centroid, depth_frame)
    eye_corner_point_xyz = cap.to_3d(eye_coords.eye_corner_point, depth_frame)

    if eye_xyz is None or pupil_xyz is None or eye_corner_point_xyz is None:
        return None

    diameter = 0.025
    center_of_eye = eye_xyz - face_normal * diameter / 2
    direction = to_unit_vector(pupil_xyz - center_of_eye)
    return Eye3D(eye_xyz, pupil_xyz, direction)


def draw_rectangle(frame, eyecache_left, eyecache_right):
    if len(eyecache_left.x) == 0 or len(eyecache_right.x) == 0:
        return

    # eyecache = eyecache_left if eyecache_left.x[-1] < 0 else eyecache_right
    # x = eyecache.x[-1]
    # y = eyecache.y[-1]

    x = (eyecache_left.x[-1] + eyecache_right.x[-1]) / 2
    y = (eyecache_left.y[-1] + eyecache_right.y[-1]) / 2

    screen_point = point_to_screen(frame, x, y)

    if screen_point is None:
        return

    W_SIZE = frame.shape[1] // 4
    H_SIZE = frame.shape[0] // 3

    pixel_x = (screen_point[0] // W_SIZE) * W_SIZE
    pixel_y = (screen_point[1] // H_SIZE) * H_SIZE

    cv2.rectangle(frame, (pixel_x, pixel_y), (pixel_x + W_SIZE, pixel_y + H_SIZE), (0x00, 0xFF, 0xFF), 5)


def loop(model: EnrichedModel, cap):
    color_frame, depth_frame = cap.get_frames()
    color_frame = cv2.flip(color_frame, 1)
    depth_frame = cv2.flip(depth_frame, 1)

    landmarks = model.detect_and_get_landmarks(color_frame)
    if landmarks is None:
        return color_frame

    face_normal = to_unit_vector(compute_face_normal(landmarks, cap, depth_frame))

    left = pupil_coords.get_left_coords(color_frame, model, landmarks)
    right = pupil_coords.get_right_coords(color_frame, model, landmarks)

    left_3d = compute_eye_3d(cap, depth_frame, face_normal, left)
    right_3d = compute_eye_3d(cap, depth_frame, face_normal, right)

    update_pointer_coords(eyecache_left, left_3d)
    update_pointer_coords(eyecache_right, right_3d)

    #index = get_index(left, frame.shape[0:2], model)
    #frame[ly, lx][lmask] = (0x00, 0x00, 0xFF)
    #frame[ry, rx][rmask] = (0xFF, 0x00, 0xFF)
    #if index is not None:
    #    draw_it(frame, index)

    color_frame[left.y, left.x][left.pupil_mask] = np.array([0xFF, 0x00, 0x00], dtype=np.uint8)
    color_frame[right.y, right.x][right.pupil_mask] = np.array([0xFF, 0x00, 0x00], dtype=np.uint8)

    draw_landmarks(color_frame, landmarks)
    #draw_pupil_coords(color_frame, left)
    #draw_pupil_coords(color_frame, right)
    draw_text_pupil_coords(color_frame, "left ", shift=25, color=np.array([0x00, 0xFF, 0x00]), eyecache=eyecache_left)
    draw_text_pupil_coords(color_frame, "right", shift=50, color=np.array([0x00, 0x00, 0xFF]), eyecache=eyecache_right)


    if face_normal is not None:
        draw_angles(color_frame, "normal", 75, face_normal)

    if left_3d:
        draw_3d_vec(color_frame, cap, face_normal, left_3d.eye_xyz, 0.1, (0x00, 0xFF, 0x00))
        draw_3d_vec(color_frame, cap, left_3d.direction, left_3d.pupil_xyz, 0.05, (0x00, 0xFF, 0xFF))
        draw_angles(color_frame, "left", 100, left_3d.direction)

    if right_3d:
        draw_3d_vec(color_frame, cap, face_normal, right_3d.eye_xyz, 0.1, (0x00, 0xFF, 0x00))
        draw_3d_vec(color_frame, cap, right_3d.direction, right_3d.pupil_xyz, 0.05, (0x00, 0xFF, 0xFF))
        draw_angles(color_frame, "right", 125, right_3d.direction)

    draw_rectangle(color_frame, eyecache_left, eyecache_right)

    #draw_text(frame, f"index: {index}", (0, 75))
    return color_frame


def main():
    cap = init_win()
    model = Model()

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
