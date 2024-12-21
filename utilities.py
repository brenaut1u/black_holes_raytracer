import numpy as np

c = 299792458
h = 6.62607015e-34
nu = 550e9
G = 6.67430e-11

# softening
scale = 1e12
eps_bodies = 5e-2
eps_photons = 5e-5

def normalize(v):
    if len(v.shape) == 1:
        return v / np.linalg.norm(v)
    else:
        n = np.linalg.norm(v, axis=-1)[:, None]
        return v / n
