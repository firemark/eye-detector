from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
from torch import FloatTensor

from eye_detector.capture_dlib.models import EyeCache, Eye3D, NetModel, ScreenBox, EyeCoords, EnrichedModel
from eye_detector.train_gaze.dataset import HEIGHT, WIDTH




def compute_eye_3d_net(model: NetModel, to_3d, rot_matrix: Rotation, eye_coords: EyeCoords) -> Optional[Eye3D]:
    if rot_matrix is None or eye_coords.centroid is None:
        return None

    eye_xyz = to_3d(eye_coords.centroid)
    if eye_xyz is None:
        return None

    rgb_img = np.float32(eye_coords.image)
    transformed_eye = model.net_transform(rgb_img).reshape((1, 3, HEIGHT, WIDTH))
    rot_matrix_flat = rot_matrix.as_mrp().reshape((1, 3))

    results = model.net((
        FloatTensor(rot_matrix_flat),
        FloatTensor(transformed_eye),
    ))
    direction = results[0].detach().numpy()
    direction = -to_unit_vector(direction)
    return Eye3D(eye_xyz, direction)


def update_pointer_coords(screen_box: ScreenBox, eyecache: EyeCache, eye_3d: Optional[Eye3D]):
    if eye_3d is None:
        return

    local_xy = screen_box.intersect(eye_3d.direction, eye_3d.eye_xyz)
    if local_xy is None:
        return

    eyecache.update(local_xy[0], local_xy[1])





def compute_rotation_matrix2(landmarks, to_3d) -> Rotation:
    points = [
        to_3d([p.x, p.y]) for p in [
            landmarks.part(5),  # Left chin
            landmarks.part(11),  # Right chin
            landmarks.part(36),  # Left eye left corner
            landmarks.part(45),  # Right eye right corner
        ]
    ]

    if any(p is None for p in points):
        return None

    chin_len_half = np.linalg.norm(points[0] - points[1]) / 2.0
    eye_len_half = np.linalg.norm(points[2] - points[3]) / 2.0
    a_height = np.linalg.norm(points[0] - points[2])
    b_height = np.linalg.norm(points[1] - points[3])
    height = (a_height + b_height) / 2.0

    points_to_rotate = np.array([
        [-chin_len_half, 0, 0],  # Left chin
        [+chin_len_half, 0, 0],  # Right chin
        [-eye_len_half, -height, 0],  # Left eye corner
        [+eye_len_half, -height, 0],  # Right eye corner
    ], dtype=float)

    points = np.array(points)
    points -= np.sum(points, axis=0) / len(points)
    points_to_rotate -= np.sum(points_to_rotate, axis=0) / len(points_to_rotate)
    rotation, _ = Rotation.align_vectors(points, points_to_rotate)
    return rotation


def compute_rotation_matrix3(landmarks, to_3d) -> Optional[Rotation]:
    points = [
        to_3d([p.x, p.y]) for p in [
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

    points = np.array(points)
    points -= np.sum(points, axis=0) / len(points)
    points_to_rotate -= np.sum(points_to_rotate, axis=0) / len(points_to_rotate)
    rotation, _ = Rotation.align_vectors(points, points_to_rotate)
    return rotation * Rotation.from_euler('x', 180, degrees=True)
