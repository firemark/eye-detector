from typing import Optional
from time import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from eye_detector.pupil_coords import EyeCoords
from eye_detector.capture_dlib.utils import to_unit_vector
from eye_detector.capture_dlib.models import EyeCache, Eye3D, ScreenBox


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


def update_pointer_coords(screen_box: ScreenBox, eyecache: EyeCache, eye_3d: Optional[Eye3D]):
    if eye_3d is None:
        return

    local_xy = screen_box.intersect(eye_3d.direction, eye_3d.pupil_xyz)
    if local_xy is None:
        return

    eyecache.update(local_xy[0], local_xy[1])


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


def compute_rotation_matrix(landmarks, cap, depth_frame) -> Rotation:
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ], dtype=float)

    points = np.array([
        [p.x, p.y] for p in [
            landmarks.part(30),
            landmarks.part(8),
            landmarks.part(36),
            landmarks.part(45),
            landmarks.part(48),
            landmarks.part(54),
        ]
    ], dtype=float)

    _, rotvec, _ = cv2.solvePnP(model_points, points, cap.camera_matrix, np.zeros((4, 1), dtype=float))
    return Rotation.from_rotvec(rotvec[..., 0])

def compute_rotation_matrix2(landmarks, cap, depth_frame) -> Rotation:
    points = [
        cap.to_3d(np.array([p.x, p.y]), depth_frame) for p in [
            landmarks.part(5),  # Left chin
            landmarks.part(11),  # Right chin
            landmarks.part(36),  # Left eye left corner
            landmarks.part(45),  # Right eye right corner
        ]
    ]

    if any(p is None for p in points):
        return None

    eye_len_half = np.linalg.norm(points[1] - points[2]) / 2.0
    side_a_len = np.linalg.norm(points[0] - points[1])
    side_b_len = np.linalg.norm(points[0] - points[2])
    sides_len = (side_a_len + side_b_len) / 2.0
    face_height = np.sqrt(sides_len ** 2 - eye_len_half ** 2)

    chin_len_half = np.linalg.norm(points[0] - points[1]) / 2.0
    eye_len_half = np.linalg.norm(points[2] - points[3]) / 2.0
    a_height = np.linalg.norm(points[0] - points[2])
    b_height = np.linalg.norm(points[1] - points[3])
    height = (a_height + b_height) / 2.0

    points_to_rotate = np.array([
        [-chin_len_half, 0, 0],  # Left chin
        [+chin_len_half, 0, 0],  # Right chin
        [-eye_len_half, height, 0],  # Left eye corner
        [+eye_len_half, height, 0],  # Right eye corner
    ], dtype=float)

    points = np.array(points)
    points -= np.sum(points, axis=0) / len(points)
    points_to_rotate -= np.sum(points_to_rotate, axis=0) / len(points_to_rotate)
    print(points)
    print(points_to_rotate)
    rotation, _ = Rotation.align_vectors(points_to_rotate, points)
    return rotation


def compute_rotation_matrix3(landmarks, cap, depth_frame) -> Optional[Rotation]:
    #face = compute_face_normal(landmarks, cap, depth_frame)
    #return Rotation.from_rotvec(face) if face is not None else None

    points = [
        cap.to_3d(np.array([p.x, p.y]), depth_frame) for p in [
            landmarks.part(30),
            landmarks.part(8),
            landmarks.part(36),
            landmarks.part(45),
            landmarks.part(48),
            landmarks.part(54),
        ]
    ]

    if any(p is None for p in points):
        return None

    points_to_rotate = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ], dtype=float)

    rotation, _ = Rotation.align_vectors(points_to_rotate, points)
    return rotation
