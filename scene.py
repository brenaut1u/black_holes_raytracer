import numpy as np
from camera import *
from body import *

class Scene:
    def __init__(self, background_image):
        self.camera = Camera(np.asarray([0, 0, 0]),
                             np.asarray([0, 1, 0]),
                             np.asarray([0, 0, -1e9]),
                             1,
                             1,
                             1000,
                             19 / 9,
                             background_image)

        self.bodies = [BlackHole(np.asarray([0, 0, -1e9]),
                                 np.asarray([0, 0, 0]),
                                 np.asarray([0, 0, 0]),
                                 1.9891e33)]

    def render_animation(self, nb_frames, dt):
        for frame in range(nb_frames):
            pass
