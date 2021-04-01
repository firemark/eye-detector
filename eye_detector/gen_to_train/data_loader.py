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


class MrlEyeDataLoader(EyeDataLoader):

    def __init__(self, chunk_size):
        super().__init__(chunk_size)
        self.paths = glob("indata/mrlEyes_2018_01/**/*.png", recursive=True)

    def load(self):
        for path in super().load():
            filename = os.path.basename(path).partition(".")[0]
            features = filename.split("_")
            if features[3:6] != ["0", "1", "0"]:
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
