import os
from glob import glob
from itertools import chain
from random import sample

from numpy import newaxis, count_nonzero
from sklearn.feature_extraction.image import PatchExtractor
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import io


class EyeDataLoader:

    def __init__(self, chunk_size):
        self.paths = glob("indata/mrlEyes_2018_01/**/*.png", recursive=True)
        self.chunk_size = chunk_size

    def load(self):
        for path in self.paths:
            filename = os.path.basename(path).partition(".")[0]
            features = filename.split("_")
            if features[3:6] != ["0", "1", "0"]:
                continue

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

    @staticmethod
    def load_image(filepath):
        img = io.imread(filepath)
        return resize(img, [64, 64])


class GenericLoader:

    def load(self, n, parts):
        if n >= len(self.paths):
            paths = self.paths
        else:
            paths = sample(self.paths, k=n)
        parts = int(parts * (len(paths) / n))

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
        if len(img.shape) > 2:
            return rgb2gray(img[:, :, 0:3])
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
        self.paths = glob("indata/face_data/*.png")

    def load(self, n, parts):
        size = 64 * 64
        return (
            img for img in super().load(n, parts)
            if size - count_nonzero(img) < 5
        )
