import numpy as np


def to_unit_vector(vec: np.ndarray):
    if vec is None:
        return vec
    return vec / np.linalg.norm(vec)
