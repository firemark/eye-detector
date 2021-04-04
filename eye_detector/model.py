import pickle
import joblib

from eye_detector.windows import HogWindow, ImgWindow

EYE_DETECT_MODEL_PATH = "outdata/eye_detect.pickle"
EYE_TRANSFORM_PATH = "outdata/eye_transform.pickle"


def store_model(model):
    with open(EYE_DETECT_MODEL_PATH, "wb") as fp:
        pickle.dump(model, fp)


def store_transform(transform):
    with open(EYE_TRANSFORM_PATH, "wb") as fp:
        pickle.dump(transform, fp)


def load_model():
    with open(EYE_DETECT_MODEL_PATH, "rb") as fp:
        return pickle.load(fp)


def load_transform():
    with open(EYE_TRANSFORM_PATH, "rb") as fp:
        return pickle.load(fp)


def load_window(model=None, transform=None):
    model = model or load_model()
    transform = transform or load_transform()

    if type(transform).__name__.startswith("Hog"):
        eye_shape = joblib.load("outdata/x_eye_shape")
        img_shape = eye_shape[0:2]
        return HogWindow(
            hog=transform,
            model=model,
            patch_size=img_shape,
        )
    else:
        return ImgWindow(
            transform=transform,
            model=model,
            patch_size=(64, 64),
            step=16,
        )
