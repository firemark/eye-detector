from glob import glob
from os import mkdir
from os.path import basename, isfile
from contextlib import suppress

import cv2
import dlib
import numpy as np
from skimage.io import imread, imsave

from eye_detector.const import CLASSES


def load(filepath):
    return imread(filepath)[:, :, 0:3]


def gen_filepath_to_save(klass, filepath):
    name = basename(filepath)
    return f"middata/transformed_label/{klass}/{name}"


def save(filepath_to_save, img):
    img_to_save = img * 0xFF
    imsave(filepath_to_save, img_to_save.astype(np.uint8))


class Model:
    RIGHT_EYE_RANGE = slice(42, 48)
    LEFT_EYE_RANGE = slice(36, 42)
    RIGHT_EYE = 36, 39
    LEFT_EYE = 42, 45

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def detect(self, frame):
        landmarks = self.detect_and_get_landmarks(frame)
        if landmarks is None:
            return None

        left_eye, _, _ = self.get_left_eye(frame, landmarks)
        right_eye, _, _ = self.get_right_eye(frame, landmarks)
        return self.concat(left_eye, right_eye)

    def detect_and_get_landmarks(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) != 1:
            return None

        face = faces[0]
        return self.predictor(gray, face)

    @classmethod
    def concat(cls, left_eye, right_eye):
        return np.concatenate([left_eye, right_eye], axis=1)

    @classmethod
    def get_left_eye(cls, frame, landmarks):
        return cls._get_eye(frame, landmarks, cls.LEFT_EYE, cls.LEFT_EYE_RANGE)

    @classmethod
    def get_right_eye(cls, frame, landmarks):
        return cls._get_eye(frame, landmarks, cls.RIGHT_EYE, cls.RIGHT_EYE_RANGE)

    @classmethod
    def get_left_eye_points(cls, landmarks):
        return cls._get_points(landmarks, cls.LEFT_EYE_RANGE)

    @classmethod
    def get_right_eye_points(cls, landmarks):
        return cls._get_points(landmarks, cls.RIGHT_EYE_RANGE)

    @staticmethod
    def _get_points(landmarks, eye_range):
        gen = range(eye_range.start, eye_range.stop)
        return np.array([(p.x, p.y) for p in (landmarks.part(i) for i in gen)])

    @classmethod
    def _get_eye(cls, frame, landmarks, left_right_indexes, eye_range):
        points = cls._get_points(landmarks, eye_range)
        x, y = cls._get_rect(landmarks, left_right_indexes, points)
        eye = cls.crop_from_rect(frame, x, y)
        #eye = cv2.normalize(eye, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        return eye, x, y

    @classmethod
    def crop_from_rect(cls, frame, x, y):
        eye = frame[y, x]
        #eye = resize(eye, (60, 40))
        return eye

    @staticmethod
    def _get_rect(landmarks, left_right_indexes, points):
        ll_p = landmarks.part(left_right_indexes[0])
        lr_p = landmarks.part(left_right_indexes[1])

        size = (lr_p.x - ll_p.x) * 0.5
        size_w = int(size) * 2
        size_h = int(size / 1.5) * 2
        #cx = (lr_p.x + ll_p.x) // 2
        #cy = (lr_p.y + ll_p.y) // 2
        cx, cy = np.sum(points, axis=0) // len(points)

        return [
            slice(cx - size_w // 2, cx + size_w // 2),
            slice(cy - size_h // 2, cy + size_h // 2),
        ]


if __name__ == "__main__":
    with suppress(FileExistsError):
        mkdir("middata/transformed_label")

    model = Model()
    tot = 0
    skip = 0
    succ = 0

    with suppress(KeyboardInterrupt):
        for klass in CLASSES:
            with suppress(FileExistsError):
                mkdir(f"middata/transformed_label/{klass}")

            for filepath in glob(f"indata/to_label/{klass}/*.png"):
                filepath_to_save = gen_filepath_to_save(klass, filepath)
                if isfile(filepath_to_save):
                    skip += 1
                    tot += 1
                    continue
                frame = load(filepath)
                img = model.detect(frame)
                if img is not None:
                    print("\033[92m.\033[0m", flush=True, end="")
                    save(filepath_to_save, img)
                    succ +=  1
                else:
                    print("\033[91mX\033[0m", flush=True, end="")
                tot += 1

    print()
    print("---")
    print("total", tot, sep="\t\t")
    print("skipped", skip, sep="\t\t")
    print("with success", succ, sep="\t")

