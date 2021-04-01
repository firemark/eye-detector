from numpy import histogram
from skimage.feature import local_binary_pattern
from eye_detector.gen_to_train.transforms.base import Transform


class LbpEye(Transform):

    def __init__(self):
        self.radius = 2
        self.n_points = 8 * self.radius
        self.bins = 200
        self.method = "default"

    def __call__(self, image):
        image = self.image_transform(image)
        # todo - multichannel support
        lbp = local_binary_pattern(image, self.n_points, self.radius, self.method)
        vector, _ = histogram(lbp.ravel(), density=True, bins=self.bins)
        return vector
