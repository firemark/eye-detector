import numpy as np
import cv2

from eye_detector.const import ROWS, PW, PH, W, H


class Cam:

    def start(self):
        raise NotImplementedError("start")

    def close(self):
        raise NotImplementedError("close")

    def get_frames(self):
        raise NotImplementedError("get_frames")

    def to_3d(self, xy, depth_frame):
        raise NotImplementedError("to_3d")


class DefaultCam(Cam):

    def __init__(self, n=0):
        self._cap = cv2.VideoCapture(n)

    def start(self):
        pass

    def close(self):
        self._cap.release()

    def get_frames(self):
        return self._cap.read()[1], None


class RealsenseCam(Cam):

    def __init__(self):
        import pyrealsense2 as rs
        self._pipeline = rs.pipeline()
        self.depth_scale = None

    def start(self):
        import pyrealsense2 as rs
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self._intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [self._intrinsics.fx, 0, self._intrinsics.ppx],
            [0, self._intrinsics.fy, self._intrinsics.ppy],
            [0, 0, self._intrinsics.ppx],
        ], dtype=float)

    def close(self):
        self._pipeline.stop()

    def get_frames(self):
        frames = self._align.process(self._pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()

        color_frame = np.asanyarray(color.get_data()) if color else None
        depth_frame = np.asanyarray(depth.get_data()) * self._depth_scale if depth else None

        return color_frame, depth_frame

    def to_3d(self, xy, depth_frame):
        import pyrealsense2 as rs
        x, y = xy
        depth = depth_frame[int(y), int(x)]
        if depth == 0.0:
            return None
        return np.array(rs.rs2_deproject_pixel_to_point(self._intrinsics, xy, depth))

    def from_3d(self, xyz):
        import pyrealsense2 as rs
        return np.array(rs.rs2_project_point_to_pixel(self._intrinsics, xyz))

def init_win(title='frame'):
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    return RealsenseCam()
    #return DefaultCam(0)
    return DefaultCam(0)


def del_win(cam):
    cam.close()
    cv2.destroyAllWindows()


def draw_camera(frame, it, title='frame', border_color=255):
    show_frame = cv2.resize(frame, (W, H))
    #cv2.putText(
    #    show_frame,
    #    f"Frame {i:03d}",
    #    (PW * x, PH * y + 50),
    #    cv2.FONT_HERSHEY_SIMPLEX,
    #    1,
    #    255,
    #)

    cv2.imshow(title, show_frame)


def draw_it(frame, it):
    h, w = frame.shape[0:2]
    x = it % ROWS
    y = it // ROWS
    pw = w // 2
    ph = h // 2
    cv2.rectangle(
        frame,
        (pw * x, ph * y),
        (pw * (x + 1), ph * (y + 1)),
        (0xFF, 0xFF, 0xFF),
        5,
    )
