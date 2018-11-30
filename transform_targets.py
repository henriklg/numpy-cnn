import numpy as np

def transform_targets(targets):
    """transform targets from a n array with values 0-9 to a nx10 array where
    each row is zero, except at the indice corresponding to the value in the
    original array"""
    n = len(targets)
    new_targets = np.empty([n, 10])
    for i in range(n):
        value = int(targets[i])
        new_targets[i,value] = 1.0
    return new_targets