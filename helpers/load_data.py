
import numpy as np

def load_ratings():
    arr = np.genfromtxt('datasets/ratings.dat', usecols=(0, 1, 2), delimiter='::', dtype='int')
    return arr






