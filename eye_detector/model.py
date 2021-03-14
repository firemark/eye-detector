import pickle
import joblib

from eye_detector.gen_to_train.transforms import HogEye
from eye_detector.windows import HogWindow

EYE_DETECT_MODEL_PATH = "outdata/eye_detect.pickle"



def store_model(model):
    with open(EYE_DETECT_MODEL_PATH, "wb") as fp:
        pickle.dump(model, fp)


def load_model():
    with open(EYE_DETECT_MODEL_PATH, "rb") as fp:
        return pickle.load(fp)


def load_window(model=None):
    if model is None:
        model = load_model()

    eye_shape = joblib.load("outdata/x_eye_shape")
    col, row = eye_shape[0:2]

    return HogWindow(
        hog=HogEye(),
        model=model,
        patch_size=(col, row),
    )
