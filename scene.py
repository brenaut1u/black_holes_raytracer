import numpy as np
from matplotlib import pyplot as plt
from camera import *
# from PostNewtonian import *

class Scene:
    def __init__(self, background_image):
        self.camera = Camera(np.asarray([0, 0, 0]),
                             np.asarray([0, 1, 0]),
                             np.asarray([0, 0, -1e9]),
                             0.1,
                             1,
                             1000,
                             16 / 9,
                             background_image,
                             1e10)

        self.positions = np.asarray([[0., 0., -1e9]])
        self.velocities = np.asarray([[0., 0., 0.]])
        self.masses = np.asarray([5e35])


    # TODO: dt should not be constant
    def update_scene(self, dt):
        self.positions, self.velocities = hermite_integrator(self.positions, self.velocities, self.masses, dt)

    def render_animation(self, nb_frames, dt):
        for frame in range(nb_frames):
            # self.update_scene(dt)
            image = self.camera.render(self.positions, self.velocities, self.masses, 0.1)
            plt.axis("off")
            plt.imshow(image)
            plt.show()
