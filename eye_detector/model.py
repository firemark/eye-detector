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
        eye_limit_ration=0.4,
    ):
        self.face_window = load_window('face')
        self.eye_window = load_window('eye')
        self.face_scales = face_scales
        self.eye_scales = eye_scales
        self.face_limit_ratio = face_limit_ratio
        self.eye_limit_ratio = eye_limit_ratio

    def detect(self, frame):
        faces_croped = self.detect_faces(frame)
        eyes_croped = self.detect_eyes(frame, faces_croped)

        try:
            region = next(r for r in regionprops(label(eyes_croped)))
        except StopIteration:
            return None

        return self._change_region_to_eye_only_img(fame, region)


    def detect_faces(self, frame):
        heatmap = self.comp_heatmap_faces(frame)
        return self.crop_faces(heatmap)

    def detect_eyes(self, frame, faces_croped):
        heatmap = self.comp_heatmap_eyes(frame, faces_croped)
        return self.crop_eyes(heatmap)

    def comp_heatmap_faces(self, frame):
        return self._multiscale_detect(frame, self.face_window, self.face_scales)

    def comp_heatmap_eyes(self, frame, croped):
        size = frame.shape[0:2]

        try:
            region = next(r for r in regionprops(label(croped)))
        except StopIteration:
            return np.zeros(size, float)

        y1, x1, y2, x2 = region.bbox
        frame = frame[y1:y2, x1:x2]

        eye_heatmap = self._multiscale_detect(frame, self.eye_window, self.eye_scales)

        resized_heatmap = np.zeros(size, float)
        resized_heatmap[y1:y2, x1:x2] = eye_heatmap
        return resized_heatmap

    def crop_faces(self, heatmap):
        return crop_heatmap(heatmap, limit_ratio=self.face_limit_ratio)

    def crop_eyes(self, heatmap):
        return crop_heatmap(heatmap, limit_ratio=self.eye_limit_ratio)

    @staticmethod
    def _multiscale_detect(frame, window, scales):
        size = frame.shape[0:2]
        heatmap = sum(
            compute_heatmap(size, window(frame, scale=scale))
            for scale in scales
        )
        return heatmap ** 2

    @staticmethod
    def _change_region_to_eye_only_img(frame, region):
        y1, x1, y2, x2 = region.bbox
        eye = img[y1:y2, x1:x2]
        eye = resize(eye, (64, 64 * 3))
        left_eye = eye[:, :64]
        right_eye = eye[:, -64:]
        return concatenate([left_eye, right_eye], axis=1)
