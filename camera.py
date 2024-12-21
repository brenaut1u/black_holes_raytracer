import numpy as np

from utilities import *
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

    def render(self, obj_pos, obj_velocities, obj_masses, dt, iter_max=10000, part=-1):
        # If part == -1, we render the full image. Otherwise, we only render a third of it (to allow to split up the render)

        n_bodies = np.int32(obj_pos.shape[0])
        accelerations_pert_old = np.zeros_like(obj_pos, dtype=np.float64)

        if part == -1:
            x_range = np.arange(self.im_width)
        elif part == 0:
            x_range = np.arange(self.im_width // 3)
        elif part == 1:
            x_range = np.arange(self.im_width // 3, 2 * self.im_width // 3)
        elif part == 2:
            x_range = np.arange(2 * self.im_width // 3, self.im_width)

        pos_photons = self.top[np.newaxis, :] + np.tile(x_range[:, np.newaxis] * self.u * self.width / self.im_width, (self.im_height, 1)) \
                                              - np.repeat(np.arange(self.im_height)[:, np.newaxis] * self.v * self.height / self.im_height, len(x_range), axis=0)
        velocities_photons = c * normalize(pos_photons - self.position)
        masses_photons = (h * nu / c**2) * np.ones(pos_photons.shape[0])

        colors = 0 * pos_photons

        pos_photons, velocities_photons, accelerations_pert_old = simulate_photon(n_bodies,
                                                                                  obj_masses,
                                                                                  dt,
                                                                                  obj_pos,
                                                                                  obj_velocities,
                                                                                  accelerations_pert_old,
                                                                                  pos_photons,
                                                                                  velocities_photons,
                                                                                  masses_photons)

        reached_background = np.linalg.norm(pos_photons - self.position, axis=-1) > self.background_dist  # the photons that travelled beyond a certain distance to the camera

        p = np.tile(pos_photons[np.newaxis, :, :], (obj_pos.shape[0], 1, 1))
        o = np.tile(obj_pos[:, np.newaxis, :], (1, pos_photons.shape[0], 1))

        distance_threshold = eps_photons * scale + 10
        reached_object = np.min(np.linalg.norm(p - o, axis=-1), axis=0) < distance_threshold  # the photons that are closer than a certain threshold to at least one object

        still_moving = ~reached_object & ~reached_background

        nb_iter = 0
        while np.max(still_moving) > 0 and nb_iter < iter_max:

            print("    Photons still moving:", len(still_moving[still_moving > 0]),
                  "; photons fallen into singularity:", len(reached_object[reached_object == 1]),
                  "; photons that reached background: ", len(reached_background[reached_background == 1]),
                  "; iterations: ", nb_iter)

            try:
                (pos_photons[still_moving],
                 velocities_photons[still_moving],
                 accelerations_pert_old[still_moving]) = simulate_photon(n_bodies,
                                                                         obj_masses,
                                                                         dt,
                                                                         obj_pos,
                                                                         obj_velocities,
                                                                         accelerations_pert_old[still_moving],
                                                                         pos_photons[still_moving],
                                                                         velocities_photons[still_moving],
                                                                         masses_photons[still_moving])

                reached_background[still_moving] = np.linalg.norm(pos_photons[still_moving] - self.position, axis=-1) > self.background_dist  # the photons that travelled beyond a certain distance to the camera

                p = np.tile(pos_photons[still_moving][np.newaxis, :, :], (obj_pos.shape[0], 1, 1))
                o = np.tile(obj_pos[:, np.newaxis, :], (1, pos_photons[still_moving].shape[0], 1))
                reached_object[still_moving] = np.min(np.linalg.norm(p - o, axis=-1), axis=0) < distance_threshold  # the photons that are closer than a certain threshold to at least one object

                still_moving[still_moving] = ~reached_object[still_moving] & ~reached_background[still_moving]
            except:
                print("Numba's mysterious error occured again... but this is fine")
                break

            nb_iter += 1

        colors[reached_object, :] = np.zeros(3)  # we assume the photons reached a singularity (would need to be changed if we have other objects such as stars)

        # Computing the position of the pixel on the background image in order to get the pixel's color
        rd = normalize(pos_photons[reached_background, :] - self.position)
        teta = np.arctan2(rd[:, 0], -rd[:, 2])
        phi = np.arctan2(rd[:, 1], np.sqrt(rd[:, 0] ** 2 + rd[:, 2] ** 2))
        px_u = ((teta + np.pi) * (self.background_image.shape[1] - 1) / (2 * np.pi)).astype(int)
        px_v = ((np.pi / 2 - phi) * (self.background_image.shape[0] - 1) / np.pi).astype(int)
        colors[reached_background, :] = self.background_image[tuple(px_v), tuple(px_u)] / 255.0

        image = colors.reshape(self.im_height, len(x_range), 3)
        return image
