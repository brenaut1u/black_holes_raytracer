import numpy as np
import matplotlib.pyplot as plt
from singlestepNbody import simulate
from photons import simulate_photon
from matplotlib.animation import PillowWriter
import os

#constant
time_steps = 200
dt = np.float64(50)  #

# bodies
positions = np.asarray([[-2e8, 0., -1e9], [2e8, 0., -1e9]])
velocities = np.asarray([[0., 1e5, 0.], [0., -1e5, 0.]])
masses = np.asarray([2e35, 4e35])
n_bodies = len(masses)

accelerations_pert_old = np.zeros(positions.shape, dtype=np.float64)

from utilities import *

position=np.asarray([0, 0, 0])
up=np.asarray([0, 1, 0])
lookat=np.asarray([0, 0, -1e9])
focal_length=0.3
cam_width=1
im_width=10
im_ratio=16/9

s_position = position
s_focal_length = focal_length

s_w = normalize(position - lookat)
s_v = normalize(up - np.dot(up, s_w) * s_w)
s_u = np.cross(s_v, s_w)

s_width = cam_width
s_height = 1#s_width / im_ratio

s_im_width = im_width
s_im_height = im_width#int(im_width / im_ratio)

s_pixel_u = s_u / s_im_width
s_pixel_v = s_v / s_im_height

s_top = s_position - s_focal_length * s_w - (s_width / 2) * s_u + (s_height / 2) * s_v

pos_photons = s_top[np.newaxis, :] + np.tile(np.arange(s_im_width)[:, np.newaxis] * s_u * s_width / s_im_width, (s_im_height, 1)) \
                                              - np.repeat(np.arange(s_im_height)[:, np.newaxis] * s_v * s_height / s_im_height, s_im_width, axis=0)
velocities_photons = c * normalize(pos_photons - s_position)
masses_photons = (h * nu / c**2) * np.ones(pos_photons.shape[0])

accelerations_pert_old_photon = np.zeros(velocities_photons.shape, dtype=np.float64)

accelerations_pert_old_photon = np.zeros(velocities_photons.shape, dtype=np.float64)
print(pos_photons)
print(velocities_photons)


nbody_positions = np.zeros((time_steps, n_bodies, 3), dtype=np.float64)
photon_positions = np.zeros((time_steps, len(pos_photons), 3), dtype=np.float64)

photon_positions[0] = pos_photons
nbody_positions[0] = positions

for t in range(time_steps - 1):
    n_bodies, masses, positions, velocities, accelerations_pert_old = simulate(
        n_bodies, masses, dt, positions, velocities, accelerations_pert_old
    )
    nbody_positions[t + 1] = positions

    # pos_photons, velocities_photons, accelerations_pert_old_photon = simulate_photon(
    #     n_bodies, masses, dt, positions, velocities, accelerations_pert_old_photon, pos_photons, velocities_photons, masses_photons
    # )
    # photon_positions[t + 1] = pos_photons

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

max_scale_xy = 1e10
max_scale_z = 1.1e9
ax.set_xlim(-max_scale_xy, max_scale_xy)
ax.set_ylim(-max_scale_xy, max_scale_xy)
ax.set_zlim(-max_scale_z, 0)


ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# for i in range(photon_positions.shape[1]):
#     ax.plot(photon_positions[:, i, 0], photon_positions[:, i, 1], photon_positions[:, i, 2],
#                        color='blue', label='Photons')
# ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='red', label='N-body')

# ax.scatter(nbody_positions[0, :, 0], nbody_positions[0, :, 1], nbody_positions[0, :, 2])
for i in range(nbody_positions.shape[1]):
    ax.scatter(nbody_positions[:, i, 0], nbody_positions[:, i, 1], nbody_positions[:, i, 2],
                       color='blue', label='Photons')
plt.show(block=True)

photon_traj_x = []
photon_traj_y = []
photon_traj_z = []



def update(frame):
    ax.cla()

    ax.set_xlim(-max_scale_xy, max_scale_xy)
    ax.set_ylim(-max_scale_xy, max_scale_xy)
    ax.set_zlim(-max_scale_z, 0)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    ax.scatter(photon_positions[frame, :, 0], photon_positions[frame, :, 1], photon_positions[frame, :, 2],
               color='blue', s=10, label='Photons')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='red', label='N-body')
    # plt.show(block=True)


# GIF
gif_path = 'nbody_photon_evolution.gif'
frames = time_steps

writer = PillowWriter(fps=30)
with writer.saving(fig, gif_path, dpi=100):
    for t in range(frames):
        update(t)
        writer.grab_frame()

plt.close(fig)

if os.path.exists(gif_path):
    print(f"GIF is saved to: {gif_path}")
else:
    print("GIF generation failed")
