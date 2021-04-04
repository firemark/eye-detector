from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from eye_detector.train.models.decorator import ProbModelDecorator


def bayes(x, y, shape):
    return ProbModelDecorator(GaussianNB())


def gauss_rbf(x, y, shape):
    return ProbModelDecorator(
        GaussianProcessClassifier(1.0 * RBF(1.0))
    )
