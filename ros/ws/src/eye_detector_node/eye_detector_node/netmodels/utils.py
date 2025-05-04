from scipy.spatial.transform import Rotation
import numpy as np


def heading_to_rotation(b):
    # Todo - optimize this
    a = np.array([1.0, 0.0, 0.0])
    b = b / np.linalg.norm(b)
    n = np.cross(a, b)
    c = np.dot(a, b)
    cos = np.sqrt(1 + c / 2)
    sin = np.sqrt(1 - c / 2)
    nn = n * sin
    q = Rotation.from_quat([*nn, cos])
    q *= Rotation.from_quat([0, 0, 1, 0])
    return q


def to_unit_vector(vec: np.ndarray):
    if vec is None:
        return vec
    return vec / np.linalg.norm(vec)
