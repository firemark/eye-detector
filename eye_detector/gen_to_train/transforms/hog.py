from skimage.feature import hog


class HogEye:

    def __init__(self, *, pixels_per_cell, orientations):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = (2, 2)
        self.orientations = orientations
        self.block_norm = 'L2-Hys'

    def __call__(self, image, visualize=False):
        return hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=False,
            visualize=visualize,
        )
