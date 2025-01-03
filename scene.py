import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from camera import *
from singlestepNbody import simulate

class Scene:
    def __init__(self, background_image):
        self.camera = Camera(np.asarray([0, 0, 0]),
                             np.asarray([0, 1, 0]),
                             np.asarray([0, 0, -1e9]),
                             0.3,
                             1,
                             500,
                             16 / 9,
                             background_image,
                             1e10)

        # self.positions = np.asarray([[-2e8, 0., -1e9], [2e8, 0., -1e9]])
        # self.velocities = np.asarray([[0., 1e5, 0.], [0., -1e5, 0.]])
        # self.masses = np.asarray([2e35, 4e35])

        self.positions = np.asarray([[-2e8, 0., -1e9], [2e8, 0., -1e9], [6e8, 0., -1e9]])
        self.velocities = np.asarray([[-1e5, 1e5, 0.], [-1e5, 0e5, 0], [-1e5, -1e5, 0.]])
        self.masses = np.asarray([1e35, 2e35, 1e35])

        self.accelerations_pert_old = np.zeros(self.positions.shape, dtype=np.float64)

    def update_scene(self, dt):
        _, masses, positions, velocities, self.accelerations_pert_old = simulate(
            len(self.masses), self.masses, dt, self.positions, self.velocities, self.accelerations_pert_old
        )

    def render_animation(self, nb_frames, dt, part=-1):
        for frame in range(nb_frames):
            print("Frame", frame)
            img = self.camera.render(self.positions, self.velocities, self.masses, 0.1, part=part)
            image.imsave('out/frame%03d.png' % frame, img)

            self.update_scene(dt)
