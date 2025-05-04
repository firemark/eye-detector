from abc import ABC, abstractmethod
from dataclasses import dataclass

from rclpy.publisher import Publisher
from geometry_msgs.msg import PoseStamped, PointStamped
import numpy as np


@dataclass
class Publishers:
    pose: Publisher
    point: Publisher
    face: Publisher
    left_eye: Publisher
    right_eye: Publisher


class Helper:
    bridge: "CvBridge"

    def __init__(self, bridge, clock):
        self.bridge = bridge
        self.clock = clock

    def to_3d(self, p):
        depth = self.depth_frame[int(p[1]), int(p[0])] * 1e-3  # type: ignore
        if depth <= 0.0:
            return None
        x, y = (p - self.intrinsics[0]) / self.intrinsics[1] * depth  # type: ignore
        return np.array([depth, -x, -y])

    def to_point(self, pos) -> PointStamped:
        point = PointStamped()
        point.header.frame_id = "camera_color_frame"
        point.header.stamp = self.clock.now().to_msg()

        point.point.x = pos[0]
        point.point.y = pos[1]
        point.point.z = pos[2]
        return point

    def to_pose(self, pos, rot) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "camera_color_frame"
        pose.header.stamp = self.clock.now().to_msg()

        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]

        quaternion = rot.as_quat()
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        return pose

    def to_img(self, image):
        image = (image * 255).astype(np.uint8)
        return self.bridge.cv2_to_imgmsg(image, "rgb8")


class INetModel(ABC):

    @abstractmethod
    def calc(self, color_frame, helper: Helper, publishers: Publishers):
        ...

