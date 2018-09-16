import numpy as np


def rounder(ratings):
    max_x = 5
    min_x = 1

    return (np.array([max(min(x, max_x), min_x) for x in ratings]))
