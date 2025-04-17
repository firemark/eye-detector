from .interface import Helper, Publishers, INetModel

from eye_detector.capture_dlib.models import EnrichedModel
from eye_detector.capture_dlib.computes import compute_eye_3d, compute_face_normal
from eye_detector import pupil_coords

import numpy as np


class LegacyModel(INetModel):

    def __init__(self) -> None:
        self.model = EnrichedModel()

    def calc(self, color_frame, helper: Helper, publishers: Publishers):
        landmarks = self.model.detect_and_get_landmarks(color_frame)
        if landmarks is None:
            return

        color = color_frame.copy().astype(np.float32) / 255.0
        normal = compute_face_normal(landmarks, helper.to_3d)

        left_eye = pupil_coords.get_left_coords(color, self.model, landmarks)
        left_eye_3d = compute_eye_3d(helper.to_3d, normal, left_eye)

        right_eye = pupil_coords.get_left_coords(color, self.model, landmarks)
        right_eye_3d = compute_eye_3d(helper.to_3d, normal, right_eye)

        if left_eye_3d:
            image = self.__crop_eye(color, left_eye)
            publishers.left_eye.publish(helper.to_img(image))

        if right_eye_3d:
            image = self.__crop_eye(color, right_eye)
            publishers.right_eye.publish(helper.to_img(image))

        if left_eye_3d:
            face = helper.heading_to_rotation(normal)
            q = helper.heading_to_rotation(left_eye_3d.direction)

            publishers.face.publish(helper.to_pose(left_eye_3d.eye_xyz, face))
            publishers.pose.publish(helper.to_pose(left_eye_3d.eye_xyz, q))
            publishers.point.publish(helper.to_point(left_eye_3d.eye_xyz - left_eye_3d.direction))

    def __crop_eye(self, color, eye: pupil_coords.EyeCoords):
        image = color[eye.y, eye.x]
        m = eye.eye_mask & ~eye.pupil_mask
        image[m] = image[m] * 0.5 + [0.0, 0.25, 0.0]
        m = eye.pupil_mask
        image[m] = image[m] * 0.5 + [0.25, 0.0, 0.0]
        if eye.eye_centroid is not None:
            color[eye.eye_centroid[1], eye.eye_centroid[0]] = [0, 1, 0]
        if eye.pupil_centroid is not None:
            color[eye.pupil_centroid[1], eye.pupil_centroid[0]] = [0, 0, 1]
        return image