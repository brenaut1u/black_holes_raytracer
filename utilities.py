import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    return v / n