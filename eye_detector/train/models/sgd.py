from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA


def sgd(x, y):
    return SGDClassifier(
        class_weight="balanced",
        tol=1e-4,
        n_jobs=8,
    )


def rbg_sgd(x, y):
    rbf = RBFSampler()
    sgd = SGDClassifier(
        class_weight="balanced",
        tol=1e-5,
        n_jobs=8,
    )

    return Pipeline([
        ('rbf', rbf),
        ('sgd', sgd),
    ])


def pca_sgd(x, y):
    pca = PCA()
    sgd = SGDClassifier(
        class_weight="balanced",
        tol=1e-4,
        n_jobs=8,
    )

    return Pipeline([
        ('pca', pca),
        ('sgd', sgd),
    ])
