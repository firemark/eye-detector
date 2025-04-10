from glob import glob
from numpy import count_nonzero

from eye_detector.gen_to_train.data_loader.base import GenericLoader


class FaceDataLoader(GenericLoader):

    def __init__(self, patch_size):
        self.paths = glob("indata/face_data/bioid_face/*.png", recursive=True)
        self.patch_size = patch_size
        self.size = self.patch_size[0] * self.patch_size[1]

    def load(self, n, parts):
        size = self.size
        return (
            img for img in super().load(n, parts)
            if size - count_nonzero(img) < 5
        )
