from math import degrees

import cv2
import numpy as np

from eye_detector.pupil_coords import EyeCoords
from eye_detector.capture_dlib.models import EyeCache, EnrichedModel, ScreenBox
from eye_detector.capture_dlib.utils import to_unit_vector


def draw_text(image, text, p, scale=1):
    cv2.putText(image, text, p, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, 255)


def draw_rectangle(frame, model: EnrichedModel):
    if len(model.eyecache_left.x) == 0 or len(model.eyecache_right.x) == 0:
        return

    x = (model.eyecache_left.x[-1] + model.eyecache_right.x[-1]) / 2
    y = (model.eyecache_left.y[-1] + model.eyecache_right.y[-1]) / 2

    screen_point = model.screen_box.point_to_screen(frame, x, y)

    if screen_point is None:
        return

    W_SIZE = frame.shape[1] // 4
    H_SIZE = frame.shape[0] // 3

    pixel_x = (screen_point[0] // W_SIZE) * W_SIZE
    pixel_y = (screen_point[1] // H_SIZE) * H_SIZE

    cv2.rectangle(frame, (pixel_x, pixel_y), (pixel_x + W_SIZE, pixel_y + H_SIZE), (0x00, 0xFF, 0xFF), 5)


def draw_text_pupil_coords(screen_box: ScreenBox, frame, prefix, shift, color: list, eyecache: EyeCache):
    color = np.ndarray(color, dtype=np.uint8)
    for i, (x, y) in enumerate(zip(eyecache.x, eyecache.y)):
        screen_point = screen_box.point_to_screen(frame, x, y)
        if screen_point is None:
            continue
        blend_color = (color * i / len(eyecache.x)).astype(int).tolist()
        cv2.circle(frame, screen_point, 5, blend_color, cv2.FILLED)

    if len(eyecache.x) > 0:
        text_xy = (30, shift)
        draw_text(frame, f"{prefix}: {eyecache.x[-1]:+08.3f}, {eyecache.y[-1]:+08.3f}", text_xy)


def draw_landmarks(frame, landmarks):
    if not landmarks:
        return

    def draw_landmarks_range(start, stop, color):
        for index in range(start, stop):
            point = landmarks.part(index)
            cv2.circle(frame, (point.x, point.y), 2, color, -1)

    # jawline
    draw_landmarks_range(0, 17, color=(0x20, 0xFF, 0xFF))
    # nose bridge
    draw_landmarks_range(27, 31, color=(0x20, 0x88, 0xFF))
    # left eye
    draw_landmarks_range(36, 42, color=(0x50, 0x50, 0xFF))
    # right eye
    draw_landmarks_range(42, 48, color=(0xFF, 0x50, 0x50))


def draw_pupil_coords(frame, eye_coords):
    if eye_coords.eye_centroid is None:
        return
    cv2.circle(frame, eye_coords.eye_centroid, 3, (0x99, 0x99, 0x99), cv2.FILLED)

    if eye_coords.pupil_centroid is None:
        return
    cv2.line(frame, eye_coords.eye_centroid, eye_coords.pupil_centroid, (0xFF, 0xFF, 0xFF), 1)


def draw_3d_vec(frame, cap, direction_xyz, point_xyz, length, color):
    direction_vec = to_unit_vector(direction_xyz) * length
    point_a = cap.from_3d(point_xyz)
    point_b = cap.from_3d(point_xyz + direction_vec)

    if np.any(np.isnan(point_a)) or np.any(np.isnan(point_b)):
        return

    cv2.line(frame, point_a.astype(int), point_b.astype(int), color, 2)


def draw_pupil_mask(frame, coords: EyeCoords, color):
    frame[coords.y, coords.x][coords.pupil_mask] = np.array(color, dtype=np.uint8)


def draw_axes(point, cap, color_frame,  rot_matrix):
    if point is None or rot_matrix is None:
        return

    k = 0.05
    xx = rot_matrix.apply([+1.0, 0.0, 0.0])
    yy = rot_matrix.apply([0.0, +1.0, 0.0])
    zz = rot_matrix.apply([0.0, 0.0, -1.0])
    p = point - (xx + yy) * k / 2.0
    draw_3d_vec(color_frame, cap, xx, p, k, (0x00, 0x00, 0xFF))
    draw_3d_vec(color_frame, cap, yy, p, k, (0xFF, 0x00, 0x00))
    draw_3d_vec(color_frame, cap, zz, p, k, (0x00, 0xFF, 0x00))
    draw_3d_vec(color_frame, cap, yy, p + xx * k, k, (0xFF, 0x00, 0x00))
    draw_3d_vec(color_frame, cap, zz, p + xx * k, k, (0x00, 0xFF, 0x00))
    draw_3d_vec(color_frame, cap, xx, p + yy * k, k, (0x00, 0x00, 0xFF))
    draw_3d_vec(color_frame, cap, zz, p + yy * k, k, (0x00, 0xFF, 0x00))
    draw_3d_vec(color_frame, cap, zz, p + (xx + yy) * k, k, (0x00, 0xFF, 0x00))
