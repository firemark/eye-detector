from skimage.transform import resize
from skimage.filters import sobel

from eye_detector.gen_to_train.transforms.base import Transform


class SobelEye(Transform):

    def __call__(self, image):
        image = self.image_transform(image)
        image = resize(image, [32, 32])
        return sobel(image)
