from time import time


def train(model, x_train, y_train):
    print("TRAIN", end=" ", flush=True)
    t = time()
    model.fit(x_train, y_train)
    print("time:", time() - t)


def predict(model, x_test):
    print("PREDICT", end=" ", flush=True)
    t = time()
    y_pred = model.predict(x_test)
    print("time:", time() - t)
    return y_pred
