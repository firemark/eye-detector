from skimage.feature import hog
from eye_detector.gen_to_train.transforms.base import Transform


class HogEye(Transform):

    def __init__(self, *, pixels_per_cell, orientations, cells_per_block=None):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block or (2, 2)
        self.orientations = orientations
        self.block_norm = 'L2-Hys'

    def __call__(self, image, visualize=False):
        image = self.image_transform(image)
        return hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=False,
            visualize=visualize,
        )
