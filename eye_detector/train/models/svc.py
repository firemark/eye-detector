from eye_detector.train.models.find import find_params

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC


def linear_svc(x, y):
    svm = LinearSVC(
        dual=False,
        class_weight="balanced",
    )
    grid = GridSearchCV(svm, {'C': [1.0, 2.0, 4.0, 8.0]})
    return find_params(grid, x, y)


def svc(x, y):
    svm = SVC(class_weight={1: 1, 0: 2})
    grid = GridSearchCV(svm, {'C': [1.0, 2.0, 4.0, 8.0]})
    return find_params(grid, x, y)
