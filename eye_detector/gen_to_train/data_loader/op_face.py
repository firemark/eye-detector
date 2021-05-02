from glob import glob
from numpy import count_nonzero

from eye_detector.gen_to_train.data_loader.base import GenericLoader


class FaceDataLoader(GenericLoader):

    def __init__(self, cols=1):
        self.paths = glob("indata/face_data/**/*.png", recursive=True)
        self.patch_size = (64, 64 * cols)
        self.size = self.patch_size[0] * self.patch_size[1]
        self.cols = cols

    def load(self, n, parts):
        size = self.size
        return (
            img for img in super().load(n, parts)
            if size - count_nonzero(img) < 5
        )
