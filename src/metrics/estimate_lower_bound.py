import numpy as np


def lower_bound(k, ground_truth):
    np.random.randint(low=0, high=k, size=ground_truth.shape)
