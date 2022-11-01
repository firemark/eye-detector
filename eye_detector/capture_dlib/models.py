from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from eye_detector.model import load_model
from eye_detector.eye_data_conv_dlib import Model


@dataclass
class Eye3D:
    eye_xyz: np.ndarray
    pupil_xyz: np.ndarray
    direction: np.ndarray


class EnrichedModel(Model):

    def __init__(self):
        super().__init__()
        self.pupil_coords_model = load_model("eye-pupil")
        self.eyecache_left = EyeCache()
        self.eyecache_right = EyeCache()
        self.screen_box = ScreenBox(
            # leftdown_position=np.array([0.0, 0.0, 0.0]),
            leftdown_position=np.array([-0.27, -0.03, -0.05]),
            width=0.57,  # 0.57
            height=0.32,  # 0.32
        )


class EyeCache:
    tx = 0
    ty = 0
    c = 0

    def __init__(self, deque_size=5):
        self.x = deque(maxlen=deque_size)
        self.y = deque(maxlen=deque_size)

    def update(self, x, y):
        self.tx += x
        self.ty += y
        self.c += 1
        if self.c >= 2:
            self.x.append(self.tx / self.c)
            self.y.append(self.ty / self.c)
            self.tx = 0
            self.ty = 0
            self.c = 0


class ScreenBox:

    def __init__(self, leftdown_position, width, height, roll=0.0, pitch=0.0, yaw=0.0):
        euler_angles = np.array([yaw, pitch, roll])
        rotation = Rotation.from_euler('xyz', euler_angles, degrees=True)
        center_position = leftdown_position + [width / 2, height / 2, 0]
        points = rotation.apply(np.array([
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [0.0, height, 0.0],
        ])) + center_position

        self.center_position = center_position
        self.leftdown_position = leftdown_position
        self.width = width
        self.height = height
        self.normal = np.cross(points[1] - points[0], points[2] - points[1])
        self.plane_offset = -self.normal.dot(points[0])  # D parameter
        self.inv_rotation = rotation.inv()

        print(self.center_position)

    def intersect(self, direction, position):
        # We have equations:
        # (x,y,z) = direction * t + position
        # x = direction[0] + t + position[0]
        # y = direction[1] + t + position[1]
        # z = direction[2] + t + position[2]
        # Ax + Bx + Cy + D = 0
        # normal = (A, B, C)
        # So result is t = -(normal.dot(position) + D) / normal.dot(direction)
        # And we need put computed t to (x,y,z) equations
        normal = self.normal
        divisor = normal.dot(direction)
        if divisor == 0.0:
            return None

        dividend = -(normal.dot(position) + self.plane_offset)
        t = dividend / divisor
        if t <= 0.0:
            return None

        global_xyz = direction * t + position
        local_xyz = self.inv_rotation.apply(global_xyz) + self.center_position
        return local_xyz[:2] + [self.width / 2, self.height / 2]

    def point_to_screen(self, frame, x, y):
        if x < 0 or x > self.width:
            return None
        if y < 0 or y > self.height:
            return None
        pixel_x = int(x * frame.shape[1] / self.width)
        pixel_y = int(y * frame.shape[0] / self.height)
        return pixel_x, pixel_y
