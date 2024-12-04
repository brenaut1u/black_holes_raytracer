import numpy as np
from utilities import *

class Body:
    def __init__(self, pos, velocity, acceleration, mass, id):
        self.pos = pos
        self.velocity = velocity
        self.acceleration = acceleration
        self.mass = mass
        self.id = id

    def move(self, t, other_bodies):
        return


class Photon(Body):
    def __init__(self, pos, velocity, acceleration, id):
        super().__init__(pos, velocity, acceleration, h*nu/c**2, id)
        self.normalize_velocity()

        velocity_normalized = normalize(self.velocity)
        self.acceleration -= np.dot(velocity_normalized, self.acceleration) * velocity_normalized

    def normalize_velocity(self):
        self.velocity = normalize(self.velocity) * c

    def move(self, t, other_bodies):
        # To be implemented
        pass

class BlackHole(Body):
    def __init__(self, pos, velocity, acceleration, mass):
        super().__init__(pos, velocity, acceleration, mass)

    def move(self, t, other_bodies):
        # To be implemented
        pass