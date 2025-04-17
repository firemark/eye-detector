from .interface import Helper, Publishers, INetModel

from eye_detector.capture_dlib.models import EnrichedModel, EyeCoords, NetModel
from eye_detector.capture_dlib.computes import compute_eye_3d_net2, compute_rotation_matrix1


class DefaultModel(INetModel):

    def __init__(self) -> None:
        self.model = EnrichedModel()
        self.net_model = NetModel("outdata/net.pth")

    def calc(self, color_frame, helper: Helper, publishers: Publishers):
        landmarks = self.model.detect_and_get_landmarks(color_frame)
        if landmarks is None:
            return

        rot_matrix = compute_rotation_matrix1(landmarks, helper.to_3d)
        left = EyeCoords.create(*self.model.get_left_eye(color_frame, landmarks))
        right = EyeCoords.create(*self.model.get_right_eye(color_frame, landmarks))

        if left:
            publishers.left_eye.publish(helper.to_img(left.image))

        if right:
            publishers.right_eye.publish(helper.to_img(right.image))

        eye_3d = compute_eye_3d_net2(self.net_model, helper.to_3d, rot_matrix, left, right)
        if eye_3d:
            q = helper.heading_to_rotation(eye_3d.direction)
            publishers.face.publish(helper.to_pose(eye_3d.eye_xyz, rot_matrix))
            publishers.pose.publish(helper.to_pose(eye_3d.eye_xyz, q))
            publishers.point.publish(helper.to_point(eye_3d.eye_xyz - eye_3d.direction))
