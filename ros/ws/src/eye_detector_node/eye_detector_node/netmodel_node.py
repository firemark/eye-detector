from eye_detector.train_gaze.dataset import MPIIIGazeDataset, SynthGazeDataset
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped

from cv_bridge import CvBridge
import numpy as np


from .netmodels.interface import INetModel, Helper, Publishers
from .netmodels.default import DefaultModel
from .netmodels.legacy import LegacyModel
from .netmodels.dataset import DatasetModel


class NetModelNode(Node):
    model: INetModel

    def __init__(self):
        super().__init__("netmodel")
        self.bridge = CvBridge()
        self.color_frame = None
        self.depth_frame = None
        self.intrinsics = None
        self.helper = Helper(self.bridge, self.get_clock())

        self.declare_parameter("model", "default")
        model = self.get_parameter("model").value.strip().lower()

        match model:
            case "default":
                self.model = DefaultModel()
            case "legacy":
                self.model = LegacyModel()
            case "dataset_mpii":
                self.model = DatasetModel(
                    dataset=MPIIIGazeDataset(root="indata/MPIIGaze")
                )
            case "dataset_synth":
                self.model = DatasetModel(
                    dataset=SynthGazeDataset(root="indata/SynthEyes_data")
                )
            case model:
                raise RuntimeError(f"unknown model {model}")

        self.color_sub = self.create_subscription(
            Image,
            "~/color",
            self.color_cb,
            qos_profile_sensor_data,
        )
        self.depth_sub = self.create_subscription(
            Image,
            "~/depth",
            self.depth_cb,
            qos_profile_sensor_data,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "~/camera_info",
            self.camera_info_cb,
            1,
        )

        self.__publishers = Publishers(
            pose=self.create_publisher(
                PoseStamped,
                "~/pose",
                qos_profile_sensor_data,
            ),
            face=self.create_publisher(
                PoseStamped,
                "~/face",
                qos_profile_sensor_data,
            ),
            point=self.create_publisher(PointStamped, "~/point", 1),
            left_eye=self.create_publisher(
                Image,
                "~/eye/left",
                qos_profile_sensor_data,
            ),
            right_eye=self.create_publisher(
                Image,
                "~/eye/right",
                qos_profile_sensor_data,
            ),
        )

    def color_cb(self, msg):
        self.color_frame = self.bridge.imgmsg_to_cv2(msg)
        self._calc()

    def depth_cb(self, msg):
        self.helper.depth_frame = self.bridge.imgmsg_to_cv2(msg)
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg)

    def camera_info_cb(self, msg):
        # Thanks to:
        # https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
        self.intrinsics = (
            np.array([msg.k[2], msg.k[5]]),
            np.array([msg.k[0], msg.k[4]]),
        )
        self.helper.intrinsics = self.intrinsics
        self.destroy_subscription(self.camera_info_sub)

    def _calc(self):
        if not self.__is_ready():
            return
        self.model.calc(self.color_frame, self.helper, self.__publishers)

    def __is_ready(self):
        return all(
            [
                self.color_frame is not None,
                self.depth_frame is not None,
                self.intrinsics is not None,
            ]
        )


def main(args=None):
    rclpy.init(args=args)
    node = NetModelNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
