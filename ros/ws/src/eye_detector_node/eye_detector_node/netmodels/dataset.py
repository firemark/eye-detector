from .interface import Helper, Publishers, INetModel

from scipy.spatial.transform import Rotation


class DatasetModel(INetModel):

    def __init__(self, dataset) -> None:
        self.d = dataset
        self.i = 0

    def calc(self, color_frame, helper: Helper, publishers: Publishers):
        xyz = (0.5, 0.0, 0.25)
        (rot, left, right), gaze = self.d[self.i]
        self.i += 1

        q = helper.heading_to_rotation(gaze.numpy())
        rot = Rotation.from_matrix(rot.reshape(3, 3)) * Rotation.from_quat([0, 0, 1, 0])

        publishers.face.publish(helper.to_pose(xyz, rot))
        publishers.pose.publish(helper.to_pose(xyz, q))
        publishers.point.publish(helper.to_point(xyz - gaze.numpy()))
        publishers.left_eye.publish(helper.to_img(self.__img_to_numpy(left)))
        publishers.right_eye.publish(helper.to_img(self.__img_to_numpy(right)))

    @staticmethod
    def __img_to_numpy(img):
        return img.numpy().swapaxes(0, 2).swapaxes(0, 1)