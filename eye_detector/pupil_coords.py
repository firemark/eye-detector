import numpy as np
from skimage.measure import label, regionprops
from skimage.draw import polygon
from skimage.color import rgb2hsv
from skimage.morphology import opening, disk


def to_mask(img, model, landmarks):
    mask = np.zeros(img.shape[0:2], dtype=bool)
    left_points = model.get_left_eye_points(landmarks)
    right_points = model.get_right_eye_points(landmarks)
    rr, cc = polygon(
        r=[p[1] for p in left_points],
        c=[p[0] for p in left_points],
        shape=mask.shape,
    )
    mask[rr, cc] = True
    rr, cc = polygon(
        r=[p[1] for p in right_points],
        c=[p[0] for p in right_points],
        shape=mask.shape,
    )
    mask[rr, cc] = True
    return mask


def to_eye_mask(eye, x, y, points):
    mask = np.zeros(eye.shape[0:2], dtype=bool)
    x = x.start
    y = y.start
    rr, cc = polygon(
        r=[p[1] - y for p in points],
        c=[p[0] - x for p in points],
        shape=mask.shape,
    )
    mask[rr, cc] = True
    return mask


def get_left_coords(img, model, landmarks, debug=False):
    eye, x, y = model.get_left_eye(img, landmarks)
    points = model.get_left_eye_points(landmarks)
    return _get_coords(eye, x, y, points, points[3], debug)


def get_right_coords(img, model, landmarks, debug=False):
    eye, x, y = model.get_right_eye(img, landmarks)
    points = model.get_right_eye_points(landmarks)
    return _get_coords(eye, x, y, points, points[0], debug)


def _get_coords(eye, x, y, points, point_to_radius, debug=False):
    eye_mask = to_eye_mask(eye, x, y, points)
    pupil_mask = to_pupil_mask(eye, eye_mask)
    eye_centroid = get_centroid(eye_mask, x.start, y.start)
    pupil_centroid = get_centroid(pupil_mask, x.start, y.start)
    radius = compute_radius(eye_centroid, point_to_radius)
    data = eye_centroid, pupil_centroid, radius

    if debug:
        return pupil_mask, x, y, *data
    else:
        return data


def compute_radius(a, b):
    if a is None:
        return None
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx*dx + dy+dy)


def compute_angle(length, radius):
    return 2 * np.arcsin(0.5 * length / radius)


def compute_row(size, eye_xy, pupil_xy, radius):
    h, w = size
    eye_x = eye_xy[0] / w - 0.5
    eye_y = eye_xy[1] / h - 0.5
    pupil_x = (pupil_xy[0] - eye_xy[0]) / radius
    pupil_y = (pupil_xy[1] - eye_xy[1]) / radius
    return eye_x, eye_y, pupil_x, pupil_y


def to_pupil_mask(img, eye_mask):
    hsv = rgb2hsv(img)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    pupil_mask = np.logical_or(
        val < 0.2,
        sat > 0.5,
    )

    total = np.logical_and(eye_mask, pupil_mask)
    return opening(total, disk(3))


def get_centroid(mask, shift_x=0, shift_y=0):
    regions = regionprops(label(mask))

    if len(regions) == 0:
        return None

    regions.sort(key=lambda r: -r.area)
    region = regions[0]
    x, y = region.centroid
    return int(y + shift_x), int(x + shift_y)
