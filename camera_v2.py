import numpy as np
from utilities import *
from test3 import simulate
from photons import simulate_photon

class Camera:
    def __init__(self, position, up, lookat, focal_length, cam_width, im_width, im_ratio, background_image, background_dist):
        self.position = position
        self.focal_length = focal_length

        self.w = normalize(position - lookat)
        self.v = normalize(up - np.dot(up, self.w) * self.w)
        self.u = np.cross(self.v, self.w)

        self.width = cam_width
        self.height = self.width / im_ratio

        self.im_width = im_width
        self.im_height = int(im_width / im_ratio)

        self.pixel_u = self.u / self.im_width
        self.pixel_v = self.v / self.im_height

        self.top = self.position - self.focal_length * self.w - (self.width / 2) * self.u + (self.height / 2) * self.v

        self.background_image = background_image
        self.background_dist = background_dist

    def render(self, obj_pos, obj_velocities, obj_masses):
        pos_photons = self.top[np.newaxis, :] + np.tile(np.arange(self.im_width)[:, np.newaxis] * self.u * self.width / self.im_width, (self.im_height, 1)) \
                                              - np.repeat(np.arange(self.im_height)[:, np.newaxis] * self.v * self.height / self.im_height, self.im_width, axis=0)

        velocities_photons = c * normalize(pos_photons - self.position)
        masses_photons = (h * nu / c**2) * np.ones(pos_photons.shape[0])

        colors = 0 * pos_photons

        n_bodies = 10
        dt = np.float64(1e3)  # Time step in seconds
        obj_pos = np.random.rand(n_bodies, 3).astype(np.float64) * scale  # in meters
        velocities = np.random.rand(n_bodies, 3).astype(np.float64) * 1e3  # in m/s
        accelerations_pert_old = np.zeros_like(obj_pos, dtype=np.float64)
        masses = np.random.rand(n_bodies).astype(np.float64) * 1e33  # in kg
        masses[0] = np.float64(1e36)  # Super Massive Black Hole SMBH

        # pos_photons += velocities_photons  # TODO: update photons positions and velocities properly 'first step'(I think we could move this into while loop)
        #
        # reached_background = np.linalg.norm(pos_photons - self.position, axis=-1) > self.background_dist  # the photons that travelled beyond a certain distance to the camera
        #
        # p = np.tile(pos_photons[np.newaxis, :, :], (obj_pos.shape[0], 1, 1))
        # o = np.tile(obj_pos[:, np.newaxis, :], (1, pos_photons.shape[0], 1))
        # reached_object = np.min(np.linalg.norm(p - o, axis=-1), axis=0) < 1000  # the photons that are closer than a certain threshold to at least one object

        while np.max(~reached_object & ~reached_background) > 0:

            n_bodies, masses, obj_pos, velocities, accelerations_pert_old=simulate(n_bodies,masses,dt,obj_pos,velocities,accelerations_pert_old)
            pos_photons,velocities_photons,accelerations_pert_old=simulate_photon(n_bodies,masses,dt,obj_pos,velocities,accelerations_pert_old,pos_photons,velocities_photons,masses_photons)

            reached_background = np.linalg.norm(pos_photons - self.position, axis=-1) > self.background_dist  # the photons that travelled beyond a certain distance to the camera

            p = np.tile(pos_photons[np.newaxis, :, :], (obj_pos.shape[0], 1, 1))
            o = np.tile(obj_pos[:, np.newaxis, :], (1, pos_photons.shape[0], 1))
            reached_object = np.min(np.linalg.norm(p - o, axis=-1), axis=0) < 1000  # the photons that are closer than a certain threshold to at least one object

        colors[reached_object, :] = np.zeros(3)  # we assume the photons reached a singularity (would need to be changed if we have other objects such as stars)

        # Computing the position of the pixel on the background image in order to get the pixel's color
        rd = normalize(pos_photons[reached_background, :] - self.position)
        teta = np.arctan2(rd[:, 0], -rd[:, 2])
        phi = np.arctan2(rd[:, 1], np.sqrt(rd[:, 0] ** 2 + rd[:, 2] ** 2))
        px_u = ((teta + np.pi) * (self.background_image.shape[1] - 1) / (2 * np.pi)).astype(int)
        px_v = ((np.pi / 2 - phi) * (self.background_image.shape[0] - 1) / np.pi).astype(int)
        colors[reached_background, :] = self.background_image[tuple(px_v), tuple(px_u)] / 255.0

        image = colors.reshape(self.im_height, self.im_width, 3)
        return image
