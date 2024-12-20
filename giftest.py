import numpy as np
import matplotlib.pyplot as plt
from singlestepNbody import simulate
from photons import simulate_photon
from matplotlib.animation import PillowWriter
import os

#constant
time_steps = 200
dt = np.float64(0.1)  #

# N body
n_bodies = 1
# positions = np.random.rand(n_bodies, 3).astype(np.float64) * 1e12
# positions[0, :] = [0e11, 0e11, 4e11]
# velocities = np.random.rand(n_bodies, 3).astype(np.float64) * 1e3
# velocities[0,:]=[0,0,0]
# masses = np.random.rand(n_bodies).astype(np.float64) * 1e33
# masses[0] = np.float64(1e36)  # SMBH
# accelerations_pert_old = np.zeros_like(positions, dtype=np.float64)
positions = np.asarray([[0., 0., -1e9]])
velocities = np.asarray([[0., 0., 0.]])
masses = np.asarray([1.9891e36])

# n_bodies = 4
# positions = np.full((n_bodies, 3), 5e11, dtype=np.float64)
# positions[0, :] = [1e11, 5e11, 5e11]
# print(positions)
# # 初始化位置为 (5e11, 5e11, 5e11)
# velocities = np.random.rand(n_bodies, 3).astype(np.float64) * 1e3
# velocities[1, :] = [0,0,0]
# masses = np.random.rand(n_bodies).astype(np.float64) * 1e33
# masses[0] = np.float64(1e37)

# accelerations_pert_old = np.zeros_like(positions, dtype=np.float64)
accelerations_pert_old = np.zeros(positions.shape, dtype=np.float64)

from utilities import *

position=np.asarray([0, 0, 0])
up=np.asarray([0, 1, 0])
lookat=np.asarray([0, 0, -1e9])
focal_length=0.3
cam_width=1
im_width=1
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

# z_coords = np.arange(1e11, 1e12 + 1, 0.5e11, dtype=np.float64).reshape(-1, 1)  #
#
#
# pos_photons = np.hstack((np.full((z_coords.shape[0], 1), 0, dtype=np.float64),  #
#                          np.full((z_coords.shape[0], 1), 5e11, dtype=np.float64),
#                          z_coords))  # z
#
#
# velocities_photons = np.tile([1.0, 0, 0.0], (pos_photons.shape[0], 1)).astype(np.float64) * 299792458
# masses_photons = np.full(pos_photons.shape[0], 6.62607015e-34 * 780e12 / 299792458**2, dtype=np.float64)
accelerations_pert_old_photon = np.zeros(velocities_photons.shape, dtype=np.float64)
#

# c = 299792458
# h = 6.62607015e-34
# nu = 780e12
# x_min, x_max = 0, 1   # x 坐标从 -0.5 到 0.5
# y_min, y_max = 0.3, 0.7   # y 坐标从 -0.2 到 0.2
# z_fixed = 0.2             # z 坐标固定为 -0.3
#
# # 根据需要选择光子数量，例如在 x 方向与 y 方向各取 11 个点形成网格
# num_x = 11
# num_y = 11
#
# # 使用 linspace 在指定区间内生成均匀分布的坐标点
# x_coords = np.linspace(x_min, x_max, num_x, dtype=np.float64)
# y_coords = np.linspace(y_min, y_max, num_y, dtype=np.float64)
#
# # 利用 meshgrid 生成二维坐标网格
# X, Y = np.meshgrid(x_coords, y_coords)
#
# # 将网格展平并组合成 (N,3) 形式的光子位置数组，Z 均为 -0.3
# pos_photons = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, z_fixed, dtype=np.float64)))
#
# # 对光子初始位置进行缩放（如乘以1e12）
# pos_photons *= 1e12
#
#
# # 初始化光子速度方向为沿负 z 轴
# velocities_photons = np.tile([0.0, 0.0, -1.0], (pos_photons.shape[0], 1)).astype(np.float64) * c

# 光子质量（有效能量对应质量）
# masses_photons = np.full(pos_photons.shape[0], (h * nu) / c**2, dtype=np.float64)

accelerations_pert_old_photon = np.zeros(velocities_photons.shape, dtype=np.float64)
print(pos_photons)
print(velocities_photons)


nbody_positions = np.zeros((time_steps, n_bodies, 3), dtype=np.float64)
photon_positions = np.zeros((time_steps, len(pos_photons), 3), dtype=np.float64)

photon_positions[0] = pos_photons

for t in range(time_steps - 1):
    # n_bodies, masses, positions, velocities, accelerations_pert_old = simulate(
    #     n_bodies, masses, dt, positions, velocities, accelerations_pert_old
    # )
    # nbody_positions[t] = positions

    pos_photons, velocities_photons, _ = simulate_photon(
        n_bodies, masses, dt, positions, velocities, accelerations_pert_old_photon, pos_photons, velocities_photons, masses_photons
    )
    photon_positions[t + 1] = pos_photons

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

ax.plot(photon_positions[:, 0, 0], photon_positions[:, 0, 1], photon_positions[:, 0, 2],
                   color='blue', label='Photons')
# ax.scatter(photon_positions[0, :, 0], photon_positions[0, :, 1], photon_positions[0, :, 2],
#                    color='blue', s=10, label='Photons')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='red', label='N-body')
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

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.set_xlim(-max_scale_xy, max_scale_xy)
    # ax.set_ylim(-max_scale_xy, max_scale_xy)
    # ax.set_zlim(-max_scale_z, 0)
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('Z (m)')
    #
    #
    # positions = nbody_positions[frame]
    # ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='red', label='N-body')
    #
    # # for i in range(len(pos_photons)):
    # #     photon_traj_x.append(photon_positions[frame, i, 0])
    # #     photon_traj_y.append(photon_positions[frame, i, 1])
    # #     photon_traj_z.append(photon_positions[frame, i, 2])
    # #     # ax.plot(photon_traj_x, photon_traj_y, photon_traj_z, 'b-', linewidth=0.5, label='Photon trajectory')
    # ax.scatter(photon_positions[0, :, 0], photon_positions[0, :, 1], photon_positions[0, :, 2],
    #            color='blue', s=10, label='Photons')
    # plt.show(block=True)


# GIF
# gif_path = 'nbody_photon_evolution.gif'
# frames = time_steps
#
# writer = PillowWriter(fps=30)
# with writer.saving(fig, gif_path, dpi=100):
#     for t in range(frames):
#         update(t)
#         writer.grab_frame()
#
# # plt.close(fig)
#
# if os.path.exists(gif_path):
#     print(f"GIF is saved to: {gif_path}")
# else:
#     print("GIF generation failed")
