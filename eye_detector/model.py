import pickle
import joblib

from eye_detector.windows import HogWindow, ImgWindow

DETECT_MODEL_PATH = "outdata/{}_detect.pickle"
TRANSFORM_PATH = "outdata/{}_transform.pickle"


def store_model(model, name):
    with open(DETECT_MODEL_PATH.format(name), "wb") as fp:
        pickle.dump(model, fp)


def store_transform(transform, name):
    with open(TRANSFORM_PATH.format(name), "wb") as fp:
        pickle.dump(transform, fp)


def load_model(name):
    with open(DETECT_MODEL_PATH.format(name), "rb") as fp:
        return pickle.load(fp)


def load_transform(name):
    with open(TRANSFORM_PATH.format(name), "rb") as fp:
        return pickle.load(fp)


def load_window(name, model=None, transform=None):
    model = model or load_model(name)
    transform = transform or load_transform(name)

    if type(transform).__name__.startswith("Hog"):
        eye_shape = joblib.load(f"outdata/x_{name}_shape")
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
