import numpy as np

c = 2.99e8
h = 6.62607015e-34
nu = 550e9
G = 6.67430e-11
scale=1e12

def normalize(v):
    if len(v.shape) == 1:
        return v / np.linalg.norm(v)
    else:
        n = np.linalg.norm(v, axis=-1)[:, None]
        return v / n
