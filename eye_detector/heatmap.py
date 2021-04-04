import numpy as np


def compute_heatmap(size, generator):
    heatmap = np.zeros(size, float)
    for *window_slice, score in generator:
        heatmap[window_slice] += score
    return heatmap


def crop_heatmap(heatmap, limit_ratio=0.5):
    limit = np.max(heatmap) * limit_ratio
    return heatmap >= limit
