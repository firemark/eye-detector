import pickle
from glob import glob
from time import time

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline


MODELS = [
    #find_rbg_sgd,
    #find_pca_sgd,
    find_sgd,
    #find_linear_svc,
    #find_svc,
]
