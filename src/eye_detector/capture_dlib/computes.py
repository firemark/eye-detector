from typing import Optional

from matplotlib.image import imsave
import numpy as np
from scipy.spatial.transform import Rotation
from torch import FloatTensor

from eye_detector.pupil_coords import EyeCoords as EyeCoordsPupil
from eye_detector.capture_dlib.utils import to_unit_vector
from eye_detector.capture_dlib.models import EyeCache, Eye3D, NetModel, ScreenBox, EyeCoords, EnrichedModel
from eye_detector.train_gaze.dataset import HEIGHT, WIDTH


def compute_eye_3d(to_3d, face_normal, eye_coords: EyeCoordsPupil) -> Optional[Eye3D]:
    if face_normal is None or eye_coords.eye_centroid is None or eye_coords.pupil_centroid is None:
        return None

    eye_xyz = to_3d(eye_coords.eye_centroid)
    pupil_xyz = to_3d(eye_coords.pupil_centroid)

    if eye_xyz is None or pupil_xyz is None:
        return None


    diameter = 0.025
    center_of_eye = eye_xyz - face_normal * diameter / 2
    direction = to_unit_vector(pupil_xyz - center_of_eye)
    return Eye3D(pupil_xyz, direction)


def compute_eye_3d_net2(model: NetModel, to_3d, rot_matrix: Rotation, left: EyeCoords, right: EyeCoords) -> Optional[Eye3D]:
    if rot_matrix is None or left.centroid is None or right.centroid is None:
        return None

    left_xyz = to_3d(left.centroid)
    if left_xyz is None:
        return None

    right_xyz = to_3d(right.centroid)
    if right_xyz is None:
        return None

    xyz = (left_xyz + right_xyz) / 2
    rot_matrix_flat = rot_matrix.as_matrix().reshape((1, 9))

    results = model.net((
        FloatTensor(rot_matrix_flat),
        FloatTensor(_get_eye(left, model)),
        FloatTensor(_get_eye(right, model)),
    ))
    return Eye3D(xyz, results[0].detach().numpy())


def _get_eye(eye: EyeCoords, model: NetModel):
    rgb_img = np.float32(eye.image.swapaxes(0, 1).swapaxes(0, 2))
    return model.net_transform(rgb_img).reshape((1, 3, HEIGHT, WIDTH))


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


def compute_face_normal(landmarks, to_3d):
    points = [
        to_3d([p.x, p.y]) for p in [
            landmarks.part(8),  # Chin
            landmarks.part(45),  # Right eye right corner
            landmarks.part(36),  # Left eye left corner
        ]
    ]

    if any(p is None for p in points):
        return None

    vec = np.cross(points[0] - points[1], points[2] - points[1])
    return to_unit_vector(vec)


def compute_rotation_matrix1(landmarks, to_3d) -> Rotation:
    points = [
        to_3d([p.x, p.y]) for p in [
            landmarks.part(8),  # Chin
            landmarks.part(45),  # Right eye right corner
            landmarks.part(36),  # Left eye left corner
        ]
    ]

    if any(p is None for p in points):
        return None

    eye_len_half = np.linalg.norm(points[1] - points[2]) / 2.0
    side_a_len = np.linalg.norm(points[0] - points[1])
    side_b_len = np.linalg.norm(points[0] - points[2])
    sides_len = (side_a_len + side_b_len) / 2.0
    face_height = np.sqrt(sides_len ** 2 - eye_len_half ** 2)

    points_to_rotate = np.array([
        [0, 0, -face_height],  # chin
        [0, +eye_len_half, 0],  # Right eye corner
        [0, -eye_len_half, 0],  # Left eye corner
    ], dtype=float)

    points = np.array(points)
    points -= np.sum(points, axis=0) / len(points)
    points_to_rotate -= np.sum(points_to_rotate, axis=0) / len(points_to_rotate)
    rotation, _ = Rotation.align_vectors(points, points_to_rotate)
    return rotation

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
