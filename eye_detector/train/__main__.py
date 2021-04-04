from argparse import ArgumentParser

import joblib

from eye_detector.train.models import NAME_TO_MODEL
from eye_detector.train.data import extract_results, prepare_data
from eye_detector.train.results import compute_cross_val_score, print_info
from eye_detector.train.select_model import train, predict
from eye_detector.model import store_model

parser = ArgumentParser(
    description="Train eye detection's model from generated data",
)
parser.add_argument(
    "models",
    metavar="MODEL",
    nargs="+",
    choices=list(NAME_TO_MODEL.keys())
)
parser.add_argument(
    "-l", "--limit",
    default=None,
    type=int,
)


if __name__ == "__main__":
    args = parser.parse_args()
    eye_shape = joblib.load("outdata/x_eye_shape")
    models = [NAME_TO_MODEL[model_name] for model_name in args.models]
    limit = args.limit
    scores = []

    x, y = extract_results()
    x_train, x_test, y_train, y_test = prepare_data(x, y)

    compute_cross_val_score(x_train, y_train)

    for model_func in models:
        print("-" * 20)
        print("---", model_func.__name__)
        print("---")
        model = model_func(x_test, y_test, eye_shape)
        if limit is not None:
            train(model, x_train[:limit], y_train[:limit])
        else:
            train(model, x_train, y_train)

        y_pred = predict(model, x_test)
        score = print_info(y_test, y_pred)
        scores.append((score, model_func.__name__, model))

    _, name, the_best_model = max(scores, key=lambda o: o[0])
    print("THE BEST MODEL:", name)
    store_model(the_best_model)
