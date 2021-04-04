import numpy as np


def compute_heatmap(size, generator):
    heatmap = np.zeros(size, float)
    for x, y, score in generator:
        heatmap[x, y] += score
    return heatmap


def crop_heatmap(heatmap, limit_ratio=0.5):
    limit = np.max(heatmap) * limit_ratio
    return heatmap >= limit
