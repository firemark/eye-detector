from time import time
from glob import glob

from numpy import array
from sklearn.model_selection import train_test_split
from joblib import load


def extract_results(name):
    print("EXTRACT RESULTS", end=" ", flush=True)
    t = time()
    x = []
    y = []
    for path in glob(f"middata/{name}_to_train/*"):
        data = load(path)
        x += (o.ravel() for o in data["x"])
        y += data["y"]
    print("time:", time() - t)
    return x, y


def prepare_data(x, y):
    print("PREPARE TRAIN AND TEST DATA", end=" ", flush=True)
    t = time()
    xx = array(x)
    del x
    yy = array(y)
    del y
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2)
    del xx
    del yy
    print("time:", time() - t)
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)
    return x_train, x_test, y_train, y_test
