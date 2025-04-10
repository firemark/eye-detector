from skimage.transform import resize
from eye_detector.gen_to_train.transforms.base import Transform


class ScaleEye(Transform):

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, image):
        image = self.image_transform(image)
        return resize(image, [self.w, self.h])
