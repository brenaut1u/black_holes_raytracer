#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:47:20 2024

@author: andypan
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utilities import *

@njit(parallel=True)
def compute_accelerations_and_jerks(positions, velocities, masses):
    """
    Compute accelerations and jerks using Newtonian gravity and 1PN corrections.
    """
    n = len(masses)
    accelerations = np.zeros_like(positions)
    jerks = np.zeros_like(positions)

    r_lim=1e9
    for i in prange(n):
        ai = np.zeros(3)
        ji = np.zeros(3)

        for j in range(n):
            if i != j:
                # Relative vectors
                x_ij = positions[i,:] - positions[j,:]
                v_ij = velocities[i,:] - velocities[j,:]
                r_ij2 = x_ij@x_ij
                r_ij = np.sqrt(r_ij2)


                # Newtonian acceleration
                ai -= G * masses[j] * x_ij / r_ij**3

                # 1PN corrections
                vi2 = velocities[i,:] @ velocities[i,:]
                vj2 = velocities[j,:] @ velocities[j,:]
                vi_dot_vj=velocities[i,:] @ velocities[j,:]
                vij_dot_xij = v_ij @ x_ij
                vj_dot_nij = velocities[j] @ (x_ij/r_ij)

                # Post-Newtonian terms (BH)
                pn_correction_ij = x_ij / r_ij**3 * (
                    4 * G * masses[j] / r_ij +
                    5 * G * masses[i] / r_ij -
                    1 * vi2 +
                    4 * vi_dot_vj -
                    2 * vj2 +
                    1.5*(vj_dot_nij**2)
                )
                pn_correction_ij += np.dot(x_ij,4*velocities[i,:]-3*velocities[j,:])*v_ij / r_ij**3
                a_pn = pn_correction_ij * G * masses[j] / c**2

                # calculate cross-terms
                for k in range(n):
                    if k != i and k != j:
                        # Relative vectors
                        x_jk = positions[j,:] - positions[k,:]
                        x_ik = positions[i,:] - positions[k,:]
                        r_jk2 = x_jk @ x_jk
                        r_jk = np.sqrt(r_jk2)

                        r_ik2 = x_ik @ x_ik
                        r_ik = np.sqrt(r_ik2)

                        pn_correction_ik = G * masses[k] * x_ij / r_ij**3 * (
                            1 / r_jk +
                            + 4 / r_ik
                            - 0.5 / r_jk**3 * np.dot(x_ij,x_jk)
                            ) - 3.5 * G * masses[k] * x_jk / (r_ij * r_jk**3)
                        a_pn += pn_correction_ik * G * masses[j] / c**2

                ai += a_pn

                # Compute jerk (time derivative of acceleration)
                # dai/dt=G*mj*(-dr_vec/dt/r^3+3*dr_scal/dt*r_vec/r^4)
                ji += G * masses[j] * (-v_ij / r_ij**3 + 3 * vij_dot_xij * x_ij / r_ij**5)

        accelerations[i,:] = ai
        jerks[i,:] = ji

    return accelerations, jerks

@njit
def hermite_integrator(positions, velocities, masses, dt):
    """
    Advance the system using Hermite integration.
    """
    # Predict positions and velocities
    accelerations, jerks = compute_accelerations_and_jerks(positions, velocities, masses)
    positions_pred = positions + velocities * dt + 0.5 * accelerations * dt**2 + (1 / 6) * jerks * dt**3
    velocities_pred = velocities + accelerations * dt + 0.5 * jerks * dt**2
    velocities_old = velocities

    # Evaluate accelerations and jerks
    accelerations_new, jerks_new = compute_accelerations_and_jerks(positions_pred, velocities_pred, masses)

    # Correct positions and velocities
    velocities += 0.5 * (accelerations + accelerations_new) * dt + (1 / 12) * (jerks - jerks_new) * dt**2
    speed = np.sqrt(np.sum(velocities**2,axis=1))
    speed_capped = np.clip(speed,0,c)
    for i in prange(len(speed)):
        velocities[i,:] *= speed_capped[i] / speed[i]
    positions += 0.5 * (velocities + velocities_old) * dt + (1 / 12) * (accelerations - accelerations_new) * dt**2

    return positions, velocities


# Parameters
n_bodies = 100
time_steps = 50000
dt = 1e3  # Time step in seconds

# Initialize positions, velocities, and masses
positions = np.random.rand(n_bodies, 3) * 1e12  # Example: in meters
#velocities = np.random.rand(n_bodies, 3) * 1e3   # Example: in m/s
velocities = np.zeros_like(positions)
masses = np.random.rand(n_bodies) * 1e31        # Example: in kg
#masses = np.zeros(n_bodies)
#masses[0]=1e20
#masses[1]=1e33


# Set up the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits
ax.set_xlim(-1e11, 1e11)
ax.set_ylim(-1e11, 1e11)
ax.set_zlim(-1e11, 1e11)
#ax.set_xlim(np.min(positions[:,0]), np.max(positions[:,0]))
#ax.set_ylim(np.min(positions[:,1]), np.max(positions[:,1]))
#ax.set_zlim(np.min(positions[:,2]), np.max(positions[:,2]))
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("N-body Simulation in Real-Time")

scat = ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    s=30
                 )

ax.set_xlim(-1e12, 1e12)
ax.set_ylim(-1e12, 1e12)
ax.set_zlim(-1e12, 1e12)
plt.autoscale(False)

def animate(frame):
    global positions, velocities
    positions, velocities = hermite_integrator(positions, velocities, masses, dt)
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    return (scat, )

ani = animation.FuncAnimation(
    fig=fig, func=animate, interval=100, blit=False)

plt.show(block=True)
