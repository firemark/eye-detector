from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from eye_detector.train.models.find import find_params
from eye_detector.train.models.decorator import ModelDecorator


def ridge(x, y, shape):
    grid = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    alpha = find_params(grid, x, y, attr="alpha_", best_attr="alpha_")
    ridge = RidgeClassifier(alpha=alpha)
    return ModelDecorator(ridge)
