from eye_detector.gen_to_train.transforms.hog import HogEye
from eye_detector.gen_to_train.transforms.lbp import LbpEye
from eye_detector.gen_to_train.transforms.scale import ScaleEye


TRANSFORMS = {
    "hog-8-8": HogEye(pixels_per_cell=(8, 8), orientations=8),
    "hog-8-12": HogEye(pixels_per_cell=(8, 8), orientations=12),
    "hog-8-16": HogEye(pixels_per_cell=(8, 8), orientations=16),
    "hog-16-8": HogEye(pixels_per_cell=(16, 16), orientations=8),
    "hog-16-12": HogEye(pixels_per_cell=(16, 16), orientations=12),
    "hog-16-16": HogEye(pixels_per_cell=(16, 16), orientations=16),
    "lbp": LbpEye(),
    "scale": ScaleEye(),
}
