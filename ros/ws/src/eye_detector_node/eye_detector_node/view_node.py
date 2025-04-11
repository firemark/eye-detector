import rclpy
import cv2
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image

from cv_bridge import CvBridge


class ViewNode(Node):

    def __init__(self):
        super().__init__("view")
        self.bridge = CvBridge()
        self.frame = None
        self.video_sub = self.create_subscription(
            Image,
            "/camera/D435/color/image_raw",
            self.video_cb,
            qos_profile_sensor_data,
        )

    def video_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg)

    def draw(self):
        if self.frame is None:
            return
        cv2.imshow("frame", self.frame)


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = ViewNode()

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        rclpy.spin(minimal_subscriber)
    finally:
        cv2.destroyAllWindows()
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
