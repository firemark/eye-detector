from skimage.feature import hog


class HogEye:

    def __init__(self):
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (2, 2)
        self.orientations = 8
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
