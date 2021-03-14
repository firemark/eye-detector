import numpy as np


def compute_heatmap(size, generator):
    heatmap = np.zeros(size, int)
    for window_slice in generator:
        heatmap[window_slice] += 1
    return heatmap


def crop_heatmap(heatmap, limit_ratio=0.5):
    limit = np.max(heatmap) * limit_ratio
    return heatmap >= limit
