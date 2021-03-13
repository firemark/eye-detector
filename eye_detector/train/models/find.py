from time import time


def find_params(grid, x, y):
    print("FINDING PARAMS...", end=" ", flush=True)
    t = time()
    grid.fit(x[:2000], y[:2000])
    print("THE BEST PARAMS:", grid.best_params_, end=" ", flush=True)
    print("time:", time() - t)

    return grid.best_estimator_
