import os
from glob import glob
from itertools import chain
from random import sample
from math import ceil

from numpy import newaxis, count_nonzero
from sklearn.feature_extraction.image import PatchExtractor
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage import io


class EyeDataLoader:

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def load(self):
        for path in self.paths:
            yield path

    def chunk_it(self):
        d = []
        for img in self.load():
            d.append(img)
            if len(d) > self.chunk_size:
                yield d
                d = []
        if d:
            yield d

    @classmethod
    def assert_args(self, args):
        pass


class MrlEyeDataLoader(EyeDataLoader):
    EYE_STATE_COL = 4
    LIGHT_CONDITION_COL = 6

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = glob("indata/mrlEyes_2018_01/**/*.png", recursive=True)

    @classmethod
    def assert_args(self, args):
        if args.dataset == 'mrl' and args.image_transform != 'gray':
            raise AssertionError("MRL dataset support only gray images")

    def load(self):
        for path in super().load():
            filename = os.path.basename(path).partition(".")[0]
            features = filename.split("_")
            if features[self.EYE_STATE_COL] != "1":
                continue
            if features[self.LIGHT_CONDITION_COL] != "1":
                continue
            yield path

    @staticmethod
    def load_image(filepath):
        img = io.imread(filepath)
        return resize(gray2rgb(img), [64, 64])


class SynthEyeDataLoader(EyeDataLoader):

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = glob("indata/SynthEyes_data/**/*.png", recursive=True)

    @staticmethod
    def load_image(filepath):
        img = io.imread(filepath)[:, :, 0:3]
        return resize(img, [64, 64])


class HelenEyeDataLoader(EyeDataLoader):

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = list(self.get_paths_from_annotations())

    def get_paths_from_annotations(self):
        print("GENERATING PATHS TO IMAGES")
        annotation_paths = glob("indata/helen/annotation/**/*.txt", recursive=True)
        for path in annotation_paths:
            with open(path) as file:
                filename = file.readline().strip()
                seekpath = f"indata/helen/**/{filename}.jpg"
                filepaths = glob(seekpath, recursive=True)
                if len(filepaths) == 0:
                    # probably a test file. Skip.
                    continue
                lines = file.readlines()

                # FUT standard
                start = 41 + 17 + 28 * 2
                r_eye_raw = lines[start:start + 20]
                l_eye_raw = lines[start + 20:start + 40]
                file.readlines(20 * 2)

            filepath = filepaths[0]
            yield filepath, self.get_bbox(r_eye_raw)
            yield filepath, self.get_bbox(l_eye_raw)

    @staticmethod
    def get_bbox(raw):
        gen = (o.partition(',') for o in raw)
        xy = [(float(x), float(y)) for x, _, y in gen]
        min_x = round(min(x for x, y in xy)) - 35
        max_x = round(max(x for x, y in xy)) + 35
        min_y = round(min(y for x, y in xy)) - 25
        max_y = round(max(y for x, y in xy)) + 25
        return (min_x, max_x, min_y, max_y)

    @staticmethod
    def load_image(data):
        filepath, bbox = data
        (x1, x2, y1, y2) = bbox
        img = io.imread(filepath)
        img = img[y1:y2, x1:x2]
        return resize(img, [64, 64])


class GenericLoader:

    def load(self, n, parts):
        if n >= len(self.paths):
            paths = self.paths
        else:
            paths = sample(self.paths, k=n)
        parts = ceil(parts * n / len(paths))

        return chain.from_iterable(
            self.extract_patches(
                self.load_image(path),
                parts,
            )
            for path in paths
        )

    @staticmethod
    def extract_patches(img, n):
        extractor = PatchExtractor(
           patch_size=(64, 64),
           max_patches=n,
        )
        return extractor.transform(img[newaxis])

    @staticmethod
    def load_image(filepath):
        img = io.imread(filepath)

        if len(img.shape) == 2:
            return gray2rgb(img)
        if img.shape[2] > 3:
            return img[:, :, 0:3]
        else:
            return img


class RoomDataLoader(GenericLoader):
    ROOM_GROUPS = [
        "buffet",
        "bedroom",
        "livingroom",
        "warehouse",
    ]

    def __init__(self):
        self.paths = list(chain.from_iterable(
            glob(f"indata/outdoor_images/{group}/*.jpg")
            for group in self.ROOM_GROUPS
        ))


class FaceDataLoader(GenericLoader):

    def __init__(self):
        self.paths = glob("indata/face_data/**/*.png", recursive=True)
        from pprint import pprint

    def load(self, n, parts):
        size = 64 * 64
        return (
            img for img in super().load(n, parts)
            if size - count_nonzero(img) < 5
        )
