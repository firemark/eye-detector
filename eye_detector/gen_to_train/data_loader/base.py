import os
from glob import glob
from itertools import chain
from random import sample
from math import ceil

from numpy import newaxis

from sklearn.feature_extraction.image import PatchExtractor
from skimage.color import gray2rgb
from skimage import io


class ImgDataLoader:

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


class MultiImgDataLoader(ImgDataLoader):

    def __init__(self, chunk_size, eye_loaders):
        super().__init__(chunk_size)
        self.eyes = eye_loaders

    @classmethod
    def assert_args(self, args):
        for eye in self.eyes:
            eye.assert_args(args)

    def load(self):
        for ind, eye in enumerate(self.eyes):
            for path in eye.load():
                yield ind, path

    def load_image(self, filepath):
        ind, path = filepath
        eye = self.eyes[ind]
        return eye.load_image(path)


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

    def extract_patches(self, img, n):
        extractor = PatchExtractor(
           patch_size=self.patch_size,
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
