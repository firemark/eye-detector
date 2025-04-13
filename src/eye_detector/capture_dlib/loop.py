import cv2

from eye_detector.capture_dlib.computes import (
    compute_rotation_matrix1,
    update_pointer_coords, compute_eye_3d_net,
)
from eye_detector.capture_dlib.models import EnrichedModel, EyeCoords
from eye_detector.capture_dlib.draw import (
    draw_rectangle,
    draw_landmarks,
    draw_text_pupil_coords,
    draw_3d_vec,
    draw_axes, draw_eye_rect,
)


def loop(model: EnrichedModel, cap):
    color_frame, depth_frame = cap.get_frames()
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    color_frame = cv2.flip(color_frame, 1)
    depth_frame = cv2.flip(depth_frame, 1)

    landmarks = model.detect_and_get_landmarks(color_frame)
    if landmarks is None:
        return color_frame

    rot_matrix = compute_rotation_matrix1(landmarks, lambda p: cap.to_3d([p.x, p.y], depth_frame))

    left = EyeCoords.get_left(color_frame, model, landmarks)
    right = EyeCoords.get_right(color_frame, model, landmarks)

    draw_eye_rect(color_frame, left)
    draw_eye_rect(color_frame, right)

    left_3d = compute_eye_3d_net(cap, model, depth_frame, rot_matrix, left)
    #right_3d = compute_eye_3d_net(cap, model, depth_frame, rot_matrix, right)
    right_3d = None

    update_pointer_coords(model.screen_box, model.eyecache_left, left_3d)
    update_pointer_coords(model.screen_box, model.eyecache_right, left_3d)

    draw_landmarks(color_frame, landmarks)
    draw_text_pupil_coords(model.screen_box, color_frame, "left ", shift=25, color=[0x00, 0xFF, 0x00], eyecache=model.eyecache_left)
    draw_text_pupil_coords(model.screen_box, color_frame, "right", shift=50, color=[0x00, 0x00, 0xFF], eyecache=model.eyecache_right)

    axes_2dpoint = landmarks.part(62)
    draw_axes(cap.to_3d([axes_2dpoint.x, axes_2dpoint.y], depth_frame), cap, color_frame, rot_matrix)

    if left_3d:
        draw_3d_vec(color_frame, cap, left_3d.direction, left_3d.eye_xyz, 0.05, (0x00, 0xFF, 0xFF))

    if right_3d:
        draw_3d_vec(color_frame, cap, right_3d.direction, right_3d.eye_xyz, 0.05, (0x00, 0xFF, 0xFF))

    draw_rectangle(color_frame, model)

    return color_frame
