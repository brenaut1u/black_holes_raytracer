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
#from utilities import *

# Constants
G = 6.67430e-11  # Gravitational constant in N*m^2/s^2
c = 299792458    # Speed of light in m/s
h = 6.62607015e-34
nu = 550e9
#eps = np.finfo(float).eps
scale=1e12

@njit(parallel=True)
def compute_accelerations_and_jerks(positions, velocities, masses, accelerations_pert_old, dt):
    """
    Compute accelerations and jerks using Newtonian gravity and 1PN corrections.
    """
    n = len(masses)
    accelerations = np.zeros_like(positions)
    jerks = np.zeros_like(positions)
    accelerations_pert = np.zeros_like(positions)

    eps=5e-2 # softening
    for i in prange(n):
        ai = np.zeros(3)
        ji = np.zeros(3)
        ai_pert = np.zeros(3)

        for j in range(n):
            if i != j and masses[j] >= 10*h*nu/c**2: # excluding influence of photons
                # Relative vectors
                x_ij = (positions[i,:] - positions[j,:])
                v_ij = velocities[i,:] - velocities[j,:]
                r_ij2 = x_ij @ x_ij
                r_ij = np.sqrt(r_ij2)
                r_ij += eps*scale


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
                pn_correction_ij += np.dot(x_ij,(4*velocities[i,:]-3*velocities[j,:]))*v_ij / r_ij**3
                a_pn = pn_correction_ij * G * masses[j] / c**2

                # calculate cross-terms
                for k in range(n):
                    if k != i and k != j and masses[k] >= 10*h*nu/c**2: # excluding influence of photons
                        # Relative vectors
                        x_jk = positions[j,:] - positions[k,:]
                        x_ik = positions[i,:] - positions[k,:]
                        r_jk2 = x_jk @ x_jk
                        r_jk = np.sqrt(r_jk2)
                        r_jk += eps*scale

                        r_ik2 = x_ik @ x_ik
                        r_ik = np.sqrt(r_ik2)
                        r_ik += eps*scale

                        pn_correction_ik = G * masses[k] * x_ij / r_ij**3 * (
                            1 / r_jk +
                            + 4 / r_ik
                            - 0.5 / r_jk**3 * np.dot(x_ij,x_jk)
                            ) - 3.5 * G * masses[k] * x_jk / (r_ij * r_jk**3)
                        a_pn += pn_correction_ik * G * masses[j] / c**2

                ai += a_pn
                ai_pert += a_pn

                # Compute Newtonian jerk (time derivative of Newtonian acceleration)
                # dai/dt=G*mj*(-dr_vec/dt/r^3+3*dr_scal/dt*r_vec/r^4)
                ji += G * masses[j] * (-v_ij / r_ij**3 + 3 * vij_dot_xij * x_ij / r_ij**5)

        accelerations[i,:] = ai
        accelerations_pert[i,:] = ai_pert
        jerks[i,:] = ji
        # jerk is calculated by adding Newtonian jerk with numerical derivative of post-Newtonian acceleration
        if np.max(accelerations_pert_old) > 0:
            jerks[i,:] += (accelerations_pert[i,:] - accelerations_pert_old[i,:]) / dt

    return accelerations, jerks, accelerations_pert

@njit(parallel=True)
def hermite_integrator(positions, velocities, masses, accelerations_pert_old, dt):
    """
    Advance the system using Hermite integration.
    """
    # Predict positions and velocities
    accelerations, jerks, accelerations_pert = compute_accelerations_and_jerks(positions, velocities, masses, accelerations_pert_old, dt)
    positions_pred = positions + velocities * dt + 0.5 * accelerations * dt**2 + (1 / 6) * jerks * dt**3
    velocities_pred = velocities + accelerations * dt + 0.5 * jerks * dt**2
    velocities_old = velocities

    # Evaluate accelerations and jerks
    accelerations_new, jerks_new, accelerations_pert_new = compute_accelerations_and_jerks(positions_pred, velocities_pred, masses, accelerations_pert, dt)

    # Correct positions and velocities
    velocities += 0.5 * (accelerations + accelerations_new) * dt + (1 / 12) * (jerks - jerks_new) * dt**2
    speed = np.sqrt(np.sum(velocities**2,axis=1))
    speed_capped = np.clip(speed,0,c)
    for i in prange(len(speed)):
        if masses[i] <= 10*h*nu/c**2:
            velocities[i,:] *= c/np.sqrt(velocities[i,:]@velocities[i,:])
        else:
            velocities[i,:] *= speed_capped[i] / speed[i]
    positions += 0.5 * (velocities + velocities_old) * dt + (1 / 12) * (accelerations - accelerations_new) * dt**2
    accelerations_pert = accelerations_pert_new

    return positions, velocities, accelerations_pert

if __name__ == "__main__":
    # Parameters
    n_bodies = 100
    time_steps = 50000
    dt = 1e3  # Time step in seconds

    #scale=1e10 # in kilometers

    # Initialize positions, velocities, and masses
    positions = np.random.rand(n_bodies, 3) * scale  # in meters
    #positions = np.zeros([n_bodies, 3])
    #positions[1,:] = np.asarray([scale,scale,scale])
    velocities = np.random.rand(n_bodies, 3) * 1e3  # in m/s
    #velocities = np.zeros_like(positions)
    accelerations_pert_old = np.zeros_like(positions)
    # masses = np.random.rand(n_bodies) * 1e33      # in kg
    masses = np.ones(n_bodies) * 1e10
    masses[0] = 1e35
    #masses = np.zeros(n_bodies)
    #masses[0]=1e30
    #masses[1]=1e33


    # Set up the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits

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

    ax.set_xlim(0, scale)
    ax.set_ylim(0, scale)
    ax.set_zlim(0, scale)
    plt.autoscale(False)

    def animate(frame):
        global positions, velocities, accelerations_pert_old
        positions, velocities, accelerations_pert_old = hermite_integrator(positions, velocities, masses, accelerations_pert_old, dt)
        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        return (scat, )

    ani = animation.FuncAnimation(
        fig=fig, func=animate, interval=100, blit=False)

    plt.show(block=True)
