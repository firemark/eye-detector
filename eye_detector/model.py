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
        eyenose_limit_ratio=0.4,
        eye_limit_ratio=0.6,
    ):
        self.face_window = load_window('face')
        self.eye_window = load_window('eye')
        self.eyenose_window = load_window('eyenose')
        self.face_scales = face_scales
        self.eye_scales = eye_scales
        self.face_limit_ratio = face_limit_ratio
        self.eye_limit_ratio = eye_limit_ratio
        self.eyenose_limit_ratio = eyenose_limit_ratio

    def detect(self, frame, with_debug_regions=False):
        faces_region = self.get_faces_region(frame)
        eyenose_region = self.get_eyenose_region(frame, faces_region)
        eye_regions = self.get_eye_regions(frame, eyenose_region)
        eyes_img = self._change_region_to_eye_only_img(
            frame,
            eye_regions,
        )
        if with_debug_regions:
            if eyes_img is None:
                return None, None, None
            return faces_region, eyenose_region, eyes_region, eyes_img

        return eyes_img

    def get_faces_region(self, frame):
        faces_croped = self.detect_faces(frame)
        if not self._is_crop_valid(faces_croped):
            return None

        return self._get_region(faces_croped)

    def get_eyenose_region(self, frame, faces_region):
        if faces_region is None:
            return None

        eyenose_croped = self.detect_eyenose(frame, faces_region)
        if not self._is_crop_valid(eyenose_croped):
            return None

        return self._get_region(eyenose_croped)

    def get_eye_regions(self, frame, eyenose_region):
        if eyenose_region is None:
            return []

        eyes_croped = self.detect_eyes(frame, eyenose_region)
        if not self._is_crop_valid(eyes_croped):
            return []

        return self._get_regions(eyes_croped)

    def detect_faces(self, frame):
        heatmap = self.comp_heatmap_faces(frame)
        return self.crop_faces(heatmap)

    def detect_eyenose(self, frame, faces_croped):
        heatmap = self.comp_heatmap_eyenose(frame, faces_croped)
        return self.crop_eyenose(heatmap)

    def detect_eyes(self, frame, eyenose_croped):
        heatmap = self.comp_heatmap_eyes(frame, eyenose_croped)
        return self.crop_eyes(heatmap)

    def comp_heatmap_faces(self, frame):
        return self._multiscale_detect(frame, self.face_window, self.face_scales)

    def comp_heatmap_eyenose(self, frame, faces_region):
        new_frame = self._crop_frame(frame, faces_region)
        heatmap = self._multiscale_detect(new_frame, self.eyenose_window, self.eye_scales)
        return self._resize_heatmap(frame, faces_region, heatmap)

    def comp_heatmap_eyes(self, frame, eyenose_region):
        new_frame = self._crop_frame(frame, eyenose_region)
        heatmap = self._multiscale_detect(new_frame, self.eye_window, self.eye_scales)
        return self._resize_heatmap(frame, eyenose_region, heatmap)

    def crop_faces(self, heatmap):
        if heatmap is None:
            return None
        return crop_heatmap(heatmap, limit_ratio=self.face_limit_ratio)

    def crop_eyenose(self, heatmap):
        if heatmap is None:
            return None
        return crop_heatmap(heatmap, limit_ratio=self.eyenose_limit_ratio)

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
    def _crop_frame(frame, region):
        y1, x1, y2, x2 = region.bbox
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _resize_heatmap(old_frame, region, heatmap):
        size = old_frame.shape[0:2]
        y1, x1, y2, x2 = region.bbox
        resized_heatmap = np.zeros(size, float)
        resized_heatmap[y1:y2, x1:x2] = heatmap
        return resized_heatmap

    @classmethod
    def _get_region(cls, croped):
        regions = cls._get_regions(croped)
        if len(regions) != 1:
            # TODO - "smart" algorithm
            return None
        return regions[0]

    @staticmethod
    def _get_regions(croped):
        return regionprops(label(croped))

    @staticmethod
    def _multiscale_detect(frame, window, scales):
        size = frame.shape[0:2]
        heatmap = np.sum(
            compute_heatmap(size, window(frame, scale=scale))
            for scale in scales
        )
        return heatmap ** 2

    @classmethod
    def _change_region_to_eye_only_img(cls, frame, regions):
        if len(regions) != 2:
            return None
        left, right = regions
        left_eye = cls._crop_frame(frame, left)
        right_eye = cls._crop_frame(frame, right)
        left_eye = resize(left_eye, (32, 32))
        right_eye = resize(right_eye, (32, 32))
        return np.concatenate([left_eye, right_eye], axis=1)
