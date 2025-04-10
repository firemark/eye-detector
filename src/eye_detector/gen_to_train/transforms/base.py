from typing import Callable

class Transform:
    image_transform: Callable

    def set_image_transform(self, image_transform):
        self.image_transform = image_transform
