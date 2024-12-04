import numpy as np

c = 2.99e8
h = 6.62607015e-34
nu = 550e9
G = 6.67430e-11

def normalize(v):
    n = np.linalg.norm(v)
    return v / n