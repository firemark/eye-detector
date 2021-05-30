import pickle

import joblib
import numpy as np
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.transform import resize

from eye_detector.heatmap import compute_heatmap, crop_heatmap
from eye_detector.windows import HogWindow, ImgWindow

DETECT_MODEL_PATH = "outdata/{}_detect.pickle"
TRANSFORM_PATH = "outdata/{}_transform.pickle"


def store_model(model, name):
    with open(DETECT_MODEL_PATH.format(name), "wb") as fp:
        pickle.dump(model, fp)


def store_transform(transform, name):
    with open(TRANSFORM_PATH.format(name), "wb") as fp:
        pickle.dump(transform, fp)


def load_model(name):
    with open(DETECT_MODEL_PATH.format(name), "rb") as fp:
        return pickle.load(fp)


def load_transform(name):
    with open(TRANSFORM_PATH.format(name), "rb") as fp:
        return pickle.load(fp)


def load_window(name, model=None, transform=None):
    model = model or load_model(name)
    transform = transform or load_transform(name)

    if type(transform).__name__.startswith("Hog"):
        eye_shape = joblib.load(f"outdata/x_{name}_shape")
        img_shape = eye_shape[0:2]
        return HogWindow(
            hog=transform,
            model=model,
            patch_size=img_shape,
        )
    else:
        # OMG BROKEN
        return ImgWindow(
            transform=transform,
            model=model,
            patch_size=(64, 64),
            step=16,
        )


class FullModel:

    def __init__(
        self,
        *,
        face_scales,
        eye_scales,
        face_limit_ratio=0.2,
        eye_limit_ratio=0.4,
    ):
        self.face_window = load_window('face')
        self.eye_window = load_window('eye')
        self.face_scales = face_scales
        self.eye_scales = eye_scales
        self.face_limit_ratio = face_limit_ratio
        self.eye_limit_ratio = eye_limit_ratio

    def detect(self, frame, with_debug_regions=False):
        faces_region = self.get_faces_region(frame)
        eyes_region = self.get_eyes_region(frame, faces_region)
        eyes_img = self._change_region_to_eye_only_img(frame, eyes_region)
        if with_debug_regions:
            if eyes_img is None:
                return None, None, None
            return faces_region, eyes_region, eyes_img

        return eye_img

    def get_faces_region(self, frame):
        faces_croped = self.detect_faces(frame)
        if not self._is_crop_valid(faces_croped):
            return None

        return self._get_region(faces_croped)

    def get_eyes_region(self, frame, faces_region):
        if faces_region is None:
            return None

        eyes_croped = self.detect_eyes(frame, faces_region)
        if not self._is_crop_valid(eyes_croped):
            return None

        return self._get_region(eyes_croped)

    def detect_faces(self, frame):
        heatmap = self.comp_heatmap_faces(frame)
        return self.crop_faces(heatmap)

    def detect_eyes(self, frame, faces_croped):
        heatmap = self.comp_heatmap_eyes(frame, faces_croped)
        return self.crop_eyes(heatmap)

    def comp_heatmap_faces(self, frame):
        return self._multiscale_detect(frame, self.face_window, self.face_scales)

    def comp_heatmap_eyes(self, frame, faces_region):
        size = frame.shape[0:2]

        y1, x1, y2, x2 = faces_region.bbox
        frame = frame[y1:y2, x1:x2]

        eye_heatmap = self._multiscale_detect(frame, self.eye_window, self.eye_scales)

        resized_heatmap = np.zeros(size, float)
        resized_heatmap[y1:y2, x1:x2] = eye_heatmap
        return resized_heatmap

    def crop_faces(self, heatmap):
        if heatmap is None:
            return None
        return crop_heatmap(heatmap, limit_ratio=self.face_limit_ratio)

    def crop_eyes(self, heatmap):
        if heatmap is None:
            return None
        return crop_heatmap(heatmap, limit_ratio=self.eye_limit_ratio)

    @staticmethod
    def _is_crop_valid(croped):
        return (
            croped is not None
            and np.any(croped)
        )

    @staticmethod
    def _get_region(croped):
        regions = regionprops(label(croped))
        if len(regions) != 1:
            # TODO - "smart" algorithm
            return None
        return regions[0]

    @staticmethod
    def _multiscale_detect(frame, window, scales):
        size = frame.shape[0:2]
        heatmap = np.sum(
            compute_heatmap(size, window(frame, scale=scale))
            for scale in scales
        )
        return heatmap ** 2

    @staticmethod
    def _change_region_to_eye_only_img(frame, region):
        if region is None:
            return None
        y1, x1, y2, x2 = region.bbox
        eye = frame[y1:y2, x1:x2]
        eye = resize(eye, (64, 64 * 3))
        left_eye = eye[:, :64]
        right_eye = eye[:, -64:]
        return np.concatenate([left_eye, right_eye], axis=1)
