import cv2

from eye_detector import pupil_coords
from eye_detector.capture_dlib.computes import (
    compute_face_normal,
    compute_eye_3d,
    update_pointer_coords,
)
from eye_detector.capture_dlib.models import EnrichedModel
from eye_detector.capture_dlib.draw import (
    draw_rectangle,
    draw_landmarks,
    draw_text_pupil_coords,
    draw_3d_vec,
    draw_pupil_mask,
    draw_angles,
)


def loop(model: EnrichedModel, cap):
    color_frame, depth_frame = cap.get_frames()
    color_frame = cv2.flip(color_frame, 1)
    depth_frame = cv2.flip(depth_frame, 1)

    landmarks = model.detect_and_get_landmarks(color_frame)
    if landmarks is None:
        return color_frame

    face_normal = compute_face_normal(landmarks, cap, depth_frame)

    left = pupil_coords.get_left_coords(color_frame, model, landmarks)
    right = pupil_coords.get_right_coords(color_frame, model, landmarks)

    left_3d = compute_eye_3d(cap, depth_frame, face_normal, left)
    right_3d = compute_eye_3d(cap, depth_frame, face_normal, right)

    update_pointer_coords(model.screen_box, model.eyecache_left, left_3d)
    update_pointer_coords(model.screen_box, model.eyecache_right, right_3d)

    draw_pupil_mask(color_frame, coords=left, color=[0xFF, 0x00, 0x00])
    draw_pupil_mask(color_frame, coords=right, color=[0xFF, 0x00, 0x00])

    draw_landmarks(color_frame, landmarks)
    draw_text_pupil_coords(model.screen_box, color_frame, "left ", shift=25, color=[0x00, 0xFF, 0x00], eyecache=model.eyecache_left)
    draw_text_pupil_coords(model.screen_box, color_frame, "right", shift=50, color=[0x00, 0x00, 0xFF], eyecache=model.eyecache_right)

    if face_normal is not None:
        draw_angles(color_frame, "normal", 75, face_normal)

    if left_3d:
        draw_3d_vec(color_frame, cap, face_normal, left_3d.eye_xyz, 0.1, (0x00, 0xFF, 0x00))
        draw_3d_vec(color_frame, cap, left_3d.direction, left_3d.pupil_xyz, 0.05, (0x00, 0xFF, 0xFF))
        draw_angles(color_frame, "left", 100, left_3d.direction)

    if right_3d:
        draw_3d_vec(color_frame, cap, face_normal, right_3d.eye_xyz, 0.1, (0x00, 0xFF, 0x00))
        draw_3d_vec(color_frame, cap, right_3d.direction, right_3d.pupil_xyz, 0.05, (0x00, 0xFF, 0xFF))
        draw_angles(color_frame, "right", 125, right_3d.direction)

    draw_rectangle(color_frame, model)

    return color_frame
