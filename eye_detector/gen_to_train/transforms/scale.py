from skimage.transform import resize


class ScaleEye:

    def __call__(self, image):
        return resize(image, [32, 32])
