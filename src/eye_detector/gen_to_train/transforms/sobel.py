from skimage.transform import resize
from skimage.filters import sobel

from eye_detector.gen_to_train.transforms.base import Transform


class SobelEye(Transform):

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, image):
        image = self.image_transform(image)
        image = resize(image, [self.w, self.h])
        return sobel(image)
