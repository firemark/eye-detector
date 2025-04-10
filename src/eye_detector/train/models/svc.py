from sklearn.model_selection import GridSearchCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from eye_detector.train.models.find import find_params
from eye_detector.train.models.decorator import ModelDecorator


def linear_svc(x, y, shape):
    svm = LinearSVC(
        dual=False,
        class_weight="balanced",
    )
    grid = GridSearchCV(svm, {'C': [1.0, 2.0, 4.0, 8.0]})
    return ModelDecorator(find_params(grid, x, y))


def _svc(x, y):
    svm = SVC(
        class_weight="balanced",
    )
    grid = GridSearchCV(svm, {'C': [1.0, 2.0, 4.0, 8.0]})
    return find_params(grid, x, y)


def svc(x, y, shape):
    return ModelDecorator(_svc(x, y))


def rbf_svc(x, y, shape):
    rbf = RBFSampler()

    return ModelDecorator(Pipeline([
        ('rbf', rbf),
        ('svc', _svc(x, y)),
    ]))


def pca_svc(x, y, shape):
    pca = PCA()
    scaler = StandardScaler()

    return ModelDecorator(Pipeline([
        ('scaler', scaler),
        ('pca', pca),
        ('sgd', _svc(x, y)),
    ]))
