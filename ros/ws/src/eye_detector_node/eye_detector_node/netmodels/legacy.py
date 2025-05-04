import numpy as np

from eye_detector.capture_dlib.models import EnrichedModel, Eye3D
from eye_detector.pupil_coords import EyeCoords as EyeCoordsPupil
from eye_detector.pupil_coords import get_left_coords, get_right_coords

from .interface import Helper, Publishers, INetModel
from .utils import to_unit_vector, heading_to_rotation



class LegacyModel(INetModel):

    def __init__(self) -> None:
        self.model = EnrichedModel()

    def calc(self, color_frame, helper: Helper, publishers: Publishers):
        landmarks = self.model.detect_and_get_landmarks(color_frame)
        if landmarks is None:
            return

        color = color_frame.copy().astype(np.float32) / 255.0
        normal = compute_face_normal(landmarks, helper.to_3d)

        left_eye = get_left_coords(color, self.model, landmarks)
        left_eye_3d = compute_eye_3d(helper.to_3d, normal, left_eye)

        right_eye = get_right_coords(color, self.model, landmarks)
        right_eye_3d = compute_eye_3d(helper.to_3d, normal, right_eye)

        if left_eye_3d:
            image = self.__crop_eye(color, left_eye)
            publishers.left_eye.publish(helper.to_img(image))

        if right_eye_3d:
            image = self.__crop_eye(color, right_eye)
            publishers.right_eye.publish(helper.to_img(image))

        if left_eye_3d:
            face = heading_to_rotation(normal)
            q = heading_to_rotation(left_eye_3d.direction)

            publishers.face.publish(helper.to_pose(left_eye_3d.eye_xyz, face))
            publishers.pose.publish(helper.to_pose(left_eye_3d.eye_xyz, q))
            publishers.point.publish(helper.to_point(left_eye_3d.eye_xyz - left_eye_3d.direction))

    def __crop_eye(self, color, eye: EyeCoordsPupil):
        image = color[eye.y, eye.x]
        m = eye.eye_mask & ~eye.pupil_mask
        image[m] = image[m] * 0.5 + [0.0, 0.25, 0.0]
        m = eye.pupil_mask
        image[m] = image[m] * 0.5 + [0.25, 0.0, 0.0]
        color[eye.eye_centroid[1], eye.eye_centroid[0]] = [0, 1, 0]
        color[eye.pupil_centroid[1], eye.pupil_centroid[0]] = [0, 0, 1]
        return image


def compute_eye_3d(to_3d, face_normal, eye_coords: EyeCoordsPupil) -> Eye3D:
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