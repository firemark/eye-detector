from glob import glob
from numpy import count_nonzero

from eye_detector.gen_to_train.data_loader.base import GenericLoader


class FaceDataLoader(GenericLoader):
    PATCH_SIZE = (64, 64)

    def __init__(self):
        self.paths = glob("indata/face_data/**/*.png", recursive=True)

    def load(self, n, parts):
        size = 64 * 64
        return (
            img for img in super().load(n, parts)
            if size - count_nonzero(img) < 5
        )
