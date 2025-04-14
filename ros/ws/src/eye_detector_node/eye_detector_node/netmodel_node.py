from functools import partial
from sys import stderr
from eye_detector.scripts.train_net import create_dataset
from eye_detector.train_gaze.dataset import MPIIIGazeDataset
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import numpy as np

from eye_detector.capture_dlib.models import EnrichedModel, Eye3D, EyeCoords, NetModel
from eye_detector.capture_dlib.computes import compute_eye_3d, compute_eye_3d_net2, compute_face_normal, compute_rotation_matrix1

from eye_detector import pupil_coords


class NetModelNode(Node):

    def __init__(self):
        super().__init__("netmodel")
        self.bridge = CvBridge()
        self.model = EnrichedModel()
        self.net_model = NetModel("outdata/net.pth")
        self.color_frame = None
        self.depth_frame = None
        self.intrinsics = None

        self.declare_parameter('model', 'default') 
        model = self.get_parameter('model').value.strip().lower()

        match model:
            case 'legacy':
                self._calc_cb = self._calc_legacy
            case 'dataset':
                self.d = MPIIIGazeDataset(root="indata/MPIIGaze", eye="left")
                self.i = 0
                self._calc_cb = self._calc_debug
            case 'default':
                self._calc_cb = self._calc
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

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "~/pose",
            qos_profile_sensor_data,
        )
        self.face_pub = self.create_publisher(
            PoseStamped,
            "~/face",
            qos_profile_sensor_data,
        )
        self.left_eye_pub = self.create_publisher(
            Image,
            "~/eye/left",
            qos_profile_sensor_data,
        )
        self.right_eye_pub = self.create_publisher(
            Image,
            "~/eye/right",
            qos_profile_sensor_data,
        )

    def color_cb(self, msg):
        self.color_frame = self.bridge.imgmsg_to_cv2(msg)
        self._calc_cb()

    def depth_cb(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg)

    def camera_info_cb(self, msg):
        # Thanks to:
        # https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
        self.intrinsics = (
            np.array([msg.k[2], msg.k[5]]),
            np.array([msg.k[0], msg.k[4]]),
        )
        self.destroy_subscription(self.camera_info_sub)

    def _calc(self):
        if not self.__is_ready():
            return

        landmarks = self.model.detect_and_get_landmarks(self.color_frame)
        if landmarks is None:
            return

        rot_matrix = compute_rotation_matrix1(landmarks, self.__to_3d)
        left = EyeCoords.create(*self.model.get_left_eye(self.color_frame, landmarks))
        right = EyeCoords.create(*self.model.get_right_eye(self.color_frame, landmarks))

        if left:
            self.left_eye_pub.publish(self.__to_img(left.image))

        if right:
            self.right_eye_pub.publish(self.__to_img(right.image))

        eye_3d = compute_eye_3d_net2(self.net_model, self.__to_3d, rot_matrix, left, right)
        if eye_3d:
            self.face_pub.publish(self.__to_pose(eye_3d.eye_xyz, rot_matrix))
            self.pose_pub.publish(self.__to_pose(eye_3d.eye_xyz, Rotation.from_rotvec(eye_3d.direction)))

    def __to_img(self, image):
        image = (image * 255).astype(np.uint8)
        return self.bridge.cv2_to_imgmsg(image, "rgb8")

    def _calc_debug(self):
        xyz = (0.5, 0.0, 0.0)
        (rot, left, right), gaze = self.d[self.i]
        self.i += 1

        self.face_pub.publish(self.__to_pose(xyz, Rotation.from_rotvec(rot)))
        self.pose_pub.publish(self.__to_pose(xyz, Rotation.from_rotvec(gaze)))

    def _calc_legacy(self):
        if not self.__is_ready():
            return

        landmarks = self.model.detect_and_get_landmarks(self.color_frame)
        if landmarks is None:
            return

        normal = compute_face_normal(landmarks, self.__to_3d)
        find_and_pub = partial(self.__find_and_pub_legacy, landmarks, normal)
        find_and_pub(pupil_coords.get_left_coords, self.pose_pub)

    def __find_and_pub_legacy(self, landmarks, face_normal, cb, pub):
        eye = cb(self.color_frame, self.model, landmarks)
        eye_3d = compute_eye_3d(self.__to_3d, face_normal, eye)

        if eye_3d:
            pub.publish(self.__to_pose(eye_3d.eye_xyz, Rotation.from_mrp(eye_3d.direction)))

    def __is_ready(self):
        return all(
            [
                self.color_frame is not None,
                self.depth_frame is not None,
                self.intrinsics is not None,
            ]
        )

    def __to_3d(self, p):
        depth = self.depth_frame[int(p[1]), int(p[0])] * 1e-3  # type: ignore
        if depth <= 0.0:
            return None
        x, y = (p - self.intrinsics[0]) / self.intrinsics[1] * depth  # type: ignore
        return np.array([depth, -x, -y])

    def __to_pose(self, pos, rot) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "camera_color_frame"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]

        quaternion = rot.as_quat()
        pose.pose.orientation.x = quaternion[1]
        pose.pose.orientation.y = quaternion[2]
        pose.pose.orientation.z = quaternion[3]
        pose.pose.orientation.w = quaternion[0]

        return pose


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
