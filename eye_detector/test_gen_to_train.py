import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.util import random_noise

from eye_detector.gen_to_train.data_loader import MrlEyeDataLoader
from eye_detector.gen_to_train.transforms import HogEye
from eye_detector.gen_to_train.image_transforms import rgb2gray



if __name__ == "__main__":
    eye_loader = MrlEyeDataLoader(chunk_size=1000)
    hog = HogEye(pixels_per_cell=(8, 8), orientations=8)
    hog.set_image_transform(rgb2gray)

    for img_path in eye_loader.load():
        img = eye_loader.load_image(img_path)
        img = random_noise(img, var=1e-4)
        sob = sobel(rgb2gray(img))
        _, trans = hog(img, visualize=True)
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        axes[0].imshow(img)
        axes[1].imshow(trans)
        axes[2].imshow(sob)
        plt.show()
