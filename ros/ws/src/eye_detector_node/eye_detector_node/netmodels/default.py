import numpy as np
from scipy.spatial.transform import Rotation
from torch import FloatTensor

from eye_detector.capture_dlib.models import Eye3D, NetModel, EyeCoords, EnrichedModel
from eye_detector.train_gaze.dataset import HEIGHT, WIDTH

from .interface import Helper, Publishers, INetModel
from .utils import heading_to_rotation



class DefaultModel(INetModel):

    def __init__(self) -> None:
        self.model = EnrichedModel()
        self.net_model = NetModel("outdata/net.pth")

    def calc(self, color_frame, helper: Helper, publishers: Publishers):
        landmarks = self.model.detect_and_get_landmarks(color_frame)
        if landmarks is None:
            return

        rot_matrix = compute_rotation_matrix(landmarks, helper.to_3d)
        left = EyeCoords.create(*self.model.get_left_eye(color_frame, landmarks))
        right = EyeCoords.create(*self.model.get_right_eye(color_frame, landmarks))

        if left:
            publishers.left_eye.publish(helper.to_img(left.image))

        if right:
            publishers.right_eye.publish(helper.to_img(right.image))

        eye_3d = compute_eye_3d(self.net_model, helper.to_3d, rot_matrix, left, right)
        if eye_3d:
            q = heading_to_rotation(eye_3d.direction)
            publishers.face.publish(helper.to_pose(eye_3d.eye_xyz, rot_matrix))
            publishers.pose.publish(helper.to_pose(eye_3d.eye_xyz, q))
            publishers.point.publish(helper.to_point(eye_3d.eye_xyz - eye_3d.direction))


def compute_eye_3d(model: NetModel, to_3d, rot_matrix: Rotation, left: EyeCoords, right: EyeCoords) -> Eye3D | None:
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


def compute_rotation_matrix(landmarks, to_3d) -> Rotation:
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


