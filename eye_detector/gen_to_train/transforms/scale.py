from skimage.transform import resize
from eye_detector.gen_to_train.transforms.base import Transform


class ScaleEye(Transform):

    def __call__(self, image):
        image = self.image_transform(image)
        return resize(image, [32, 32])
