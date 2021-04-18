from eye_detector.gen_to_train.transforms.hog import HogEye
from eye_detector.gen_to_train.transforms.lbp import LbpEye
from eye_detector.gen_to_train.transforms.scale import ScaleEye
from eye_detector.gen_to_train.transforms.sobel import SobelEye


TRANSFORMS = {
    "hog-8-8": HogEye(pixels_per_cell=(8, 8), orientations=8),
    "hog-8-12": HogEye(pixels_per_cell=(8, 8), orientations=12),
    "hog-8-16": HogEye(pixels_per_cell=(8, 8), orientations=16),
    "hog-16-8": HogEye(pixels_per_cell=(16, 16), orientations=8),
    "hog-16-12": HogEye(pixels_per_cell=(16, 16), orientations=12),
    "hog-16-16": HogEye(pixels_per_cell=(16, 16), orientations=16),
    "hog-24-8": HogEye(pixels_per_cell=(24, 24), orientations=8),
    "hog-32-8": HogEye(pixels_per_cell=(32, 32), orientations=8),
    "hog-64-8": HogEye(pixels_per_cell=(64, 64), orientations=8),
    "lbp": LbpEye(),
    "scale-16": ScaleEye(16, 16),
    "scale-32": ScaleEye(32, 32),
    "scale-64": ScaleEye(64, 64),
    "sobel-32": SobelEye(32, 32),
    "sobel-64": SobelEye(64, 64),
}
