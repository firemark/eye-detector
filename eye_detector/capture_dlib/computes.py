from typing import Optional

import numpy as np

from eye_detector.pupil_coords import EyeCoords
from eye_detector.capture_dlib.utils import to_unit_vector
from eye_detector.capture_dlib.models import EyeCache, Eye3D, ScreenBox


def compute_eye_3d(cap, depth_frame, face_normal, eye_coords: EyeCoords) -> Optional[Eye3D]:
    if face_normal is None or eye_coords.eye_centroid is None or eye_coords.pupil_centroid is None:
        return None

    eye_xyz = cap.to_3d(eye_coords.eye_centroid, depth_frame)
    pupil_xyz = cap.to_3d(eye_coords.pupil_centroid, depth_frame)
    eye_corner_point_xyz = cap.to_3d(eye_coords.eye_corner_point, depth_frame)

    if eye_xyz is None or pupil_xyz is None or eye_corner_point_xyz is None:
        return None

    diameter = 0.025
    center_of_eye = eye_xyz - face_normal * diameter / 2
    direction = to_unit_vector(pupil_xyz - center_of_eye)
    return Eye3D(eye_xyz, pupil_xyz, direction)


def update_pointer_coords(screen_box: ScreenBox, eyecache: EyeCache, eye_3d: Optional[Eye3D]):
    if eye_3d is None:
        return

    local_xy = screen_box.intersect(eye_3d.direction, eye_3d.pupil_xyz)
    if local_xy is None:
        return

    eyecache.update(local_xy[0], local_xy[1])


def compute_face_normal(landmarks, cap, depth_frame):
    #return np.array([0.0, 0.0, -1.0])
    points = [
        cap.to_3d(np.array([p.x, p.y]), depth_frame) for p in [
            landmarks.part(8),  # Chin
            landmarks.part(45),  # Right eye right corner
            landmarks.part(36),  # Left eye left corner
        ]
    ]

    if any(p is None for p in points):
        return None

    vec = np.cross(points[1] - points[0], points[2] - points[1])
    return to_unit_vector(vec)
