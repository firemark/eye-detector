from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from eye_detector.train.models.decorator import ModelDecorator


def sgd(x, y, shape):
    return ModelDecorator(SGDClassifier(
        class_weight="balanced",
        tol=1e-4,
        n_jobs=8,
    ))


def rbf_sgd(x, y, shape):
    rbf = RBFSampler()
    sgd = SGDClassifier(
        class_weight="balanced",
        tol=1e-5,
        n_jobs=8,
    )

    return ModelDecorator(Pipeline([
        ('rbf', rbf),
        ('sgd', sgd),
    ]))


def pca_sgd(x, y, shape):
    pca = PCA()
    sgd = SGDClassifier(
        class_weight="balanced",
        tol=1e-4,
        n_jobs=8,
    )

    return ModelDecorator(Pipeline([
        ('pca', pca),
        ('sgd', sgd),
    ]))
