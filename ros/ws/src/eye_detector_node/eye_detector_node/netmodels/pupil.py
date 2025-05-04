import numpy as np
from sys import stderr
from torch import FloatTensor

from eye_detector.capture_dlib.models import EnrichedModel, NetModel, EyeCoords
from eye_detector.train_gaze.dataset import HEIGHT, WIDTH
from eye_detector.pupil_coords import EyeCoords as EyeCoordsPupil

from .interface import Helper, Publishers, INetModel
from .utils import heading_to_rotation
from .legacy import compute_face_normal, compute_eye_3d



class PupilModel(INetModel):

    def __init__(self) -> None:
        self.model = EnrichedModel()
        self.net_model = NetModel("outdata/net.pth")

    def calc(self, color_frame, helper: Helper, publishers: Publishers):
        landmarks = self.model.detect_and_get_landmarks(color_frame)
        if landmarks is None:
            return

        color = color_frame.copy().astype(np.float32) / 255.0
        normal = compute_face_normal(landmarks, helper.to_3d)
        left = self.__create(self.model.get_left_eye(color_frame, landmarks))
        right = self.__create(self.model.get_right_eye(color_frame, landmarks))


        if left:
            image = self.__crop_eye(color, left)
            publishers.left_eye.publish(helper.to_img(image))

        if right:
            image = self.__crop_eye(color, right)
            publishers.right_eye.publish(helper.to_img(image))

        eye = compute_eye_3d(helper.to_3d, normal, left) if left else None

        if eye:
            print(eye.eye_xyz, file=stderr)
            face = heading_to_rotation(normal)
            q = heading_to_rotation(eye.direction)

            publishers.face.publish(helper.to_pose(eye.eye_xyz, face))
            publishers.pose.publish(helper.to_pose(eye.eye_xyz, q))
            publishers.point.publish(helper.to_point(eye.eye_xyz - eye.direction))

    def __create(self, t) -> EyeCoordsPupil:
        eye = EyeCoords.create(*t)
        return EyeCoordsPupil(
            x=eye.x,
            y=eye.y,
            eye_centroid=eye.centroid,
            pupil_centroid=self.__compute_pupil(eye),
        )

    def __crop_eye(self, color, eye: EyeCoordsPupil):
        color[eye.eye_centroid[1], eye.eye_centroid[0]] = [0, 1, 0]
        color[eye.pupil_centroid[1], eye.pupil_centroid[0]] = [0, 0, 1]
        return color[eye.y, eye.x]

    def __compute_pupil(self, eye: EyeCoords) -> np.ndarray | None:
        results = self.net_model.net(FloatTensor(self.__get_eye(eye)))
        pupil = results[0].detach().numpy()
        pupil = np.array([-pupil[1], pupil[0]])
        size = np.array([eye.x.stop - eye.x.start, eye.y.stop - eye.y.start])
        start = np.array([eye.x.start, eye.y.start])
        return ((pupil + 1.0) / 2 * size).astype(int) + start

    def __get_eye(self, eye: EyeCoords):
        rgb_img = np.float32(eye.image.swapaxes(0, 1).swapaxes(0, 2))
        return self.net_model.net_transform(rgb_img).reshape((1, 3, HEIGHT, WIDTH))