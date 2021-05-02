from time import time


def find_params(grid, x, y, attr="best_estimator_", best_attr="best_params_"):
    print("FINDING PARAMS...", end=" ", flush=True)
    t = time()
    grid.fit(x[:4000], y[:4000])
    print("THE BEST PARAMS:", getattr(grid, best_attr), end=" ", flush=True)
    print("time:", time() - t)

    return getattr(grid, attr)
