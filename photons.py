#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:47:20 2024

@author: andypan
"""

import os  # For file management
from PIL import Image  # For GIF creation
import numpy as np
from numba import njit, prange
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utilities import *

@njit(parallel=True)
def compute_accelerations_and_jerks(positions, velocities, masses, accelerations_pert_old, dt,pos_photons,velocities_photons,masses_photons):
    """
    Compute accelerations and jerks using Newtonian gravity and 1PN corrections.
    """

    n = len(masses)
    nphotons = len(masses_photons)
    # accelerations = np.zeros_like(pos_photons)
    # jerks = np.zeros_like(pos_photons)
    # accelerations_pert = np.zeros_like(pos_photons)
    accelerations = np.zeros_like(pos_photons, dtype=np.float64)
    jerks = np.zeros_like(pos_photons, dtype=np.float64)
    accelerations_pert = np.zeros_like(pos_photons, dtype=np.float64)
    eps=5e-2 # softening

    for i in prange(nphotons):
        ai = np.zeros(3)
        ji = np.zeros(3)
        ai_pert = np.zeros(3)

        for j in range(n):
            # Relative vectors
            x_ij = pos_photons[i,:] - positions[j,:]
            v_ij = velocities_photons[i,:] - velocities[j,:]
            r_ij2 = x_ij @ x_ij
            r_ij = np.sqrt(r_ij2)
            r_ij += eps*scale


            # Newtonian acceleration
            ai -= G * masses[j] * x_ij / r_ij**3

            # Compute Newtonian jerk (time derivative of Newtonian acceleration)
            # dai/dt=G*mj*(-dr_vec/dt/r^3+3*dr_scal/dt*r_vec/r^4)
            ji += G * masses[j] * (-v_ij / r_ij**3 + 3 * vij_dot_xij * x_ij / r_ij**5)

        accelerations[i,:] = ai
        accelerations_pert[i,:] = ai_pert
        jerks[i,:] = ji

    return accelerations, jerks, accelerations_pert

@njit(parallel=True)
def hermite_integrator_photon(positions, velocities, masses, accelerations_pert_old, dt,pos_photons,velocities_photons,masses_photons):
    """
    Advance the system using Hermite integration.
    """
    # Compute accelerations and jerks
    accelerations, jerks, accelerations_pert = compute_accelerations_and_jerks(positions, velocities, masses, accelerations_pert_old,dt,pos_photons,velocities_photons,masses_photons)
    
    velocities_photons_old = velocities_photons
    # Update velocities and positions
    velocities_photons += accelerations * dt + 1/2 * jerks * dt**2
    speed = np.sqrt(np.sum(velocities**2,axis=1))
    for i in range(len(speed)):
        velocities_photons[i,:] *= c/np.sqrt(velocities_photons[i,:]@velocities_photons[i,:])
        accelerations[i,:] = (velocities_photons[i,:] - velocities_photons_old[i,:]) / dt
        

    pos_photons += velocities_photons * dt + 1 / 2 * accelerations * dt**2 #+ 1 / 6 * jerks * dt**3

    return pos_photons, velocities_photons, accelerations_pert

def simulate_photon(n_bodies,masses,dt,positions,velocities,accelerations_pert_old,pos_photons,velocities_photons,masses_photons):

    pos_photons,velocities_photons,acc = hermite_integrator_photon(positions, velocities, masses,accelerations_pert_old,dt,pos_photons,velocities_photons,masses_photons)

    return pos_photons,velocities_photons,acc


# Parameters
# n_bodies = 10
# time_steps = 300
# dt = 1e3  # Time step in seconds
# positions = np.random.rand(n_bodies, 3) * scale  # in meters
# velocities = np.random.rand(n_bodies, 3) * 1e3  # in m/s
# accelerations_pert_old = np.zeros_like(positions)
# masses = np.random.rand(n_bodies) * 1e33  # in kg
# masses[0]=1e36 #Super Massive Black Hole SMBH
#
#
# pos_photons = np.array([
#     [0, 0, 0],  # 光子1的位置
#     [0, 0, 1],  # 光子2的位置
#     [0, 0, 2]   # 光子3的位置
# ])
#
# # 定义光子的速度方向
# velocities_photons = np.array([
#     [1, 0, 0],  # 光子1的速度方向
#     [1, 0, 0],  # 光子2的速度方向
#     [1, 0, 0]   # 光子3的速度方向
# ]) * c  # 乘以光速
# masses_photons=(h * 780e12 / c**2) * np.ones(pos_photons.shape[0])

# n_bodies = 10
# time_steps = 300
# dt = np.float64(1e3)  # Time step in seconds
# positions = np.random.rand(n_bodies, 3).astype(np.float64) * scale  # in meters
# velocities = np.random.rand(n_bodies, 3).astype(np.float64) * 1e3  # in m/s
# accelerations_pert_old = np.zeros_like(positions, dtype=np.float64)
# masses = np.random.rand(n_bodies).astype(np.float64) * 1e33  # in kg
# masses[0] = np.float64(1e36)  # Super Massive Black Hole SMBH
#
# pos_photons = np.array([
#     [0.0, 0.0, 0.0],  # 光子1的位置
#     [0.0, 0.0, 1.0],  # 光子2的位置
#     [0.0, 0.0, 2.0]   # 光子3的位置
# ], dtype=np.float64)
#
# # 定义光子的速度方向
# velocities_photons = np.array([
#     [1.0, 0.0, 0.0],  # 光子1的速度方向
#     [1.0, 0.0, 0.0],  # 光子2的速度方向
#     [1.0, 0.0, 0.0]   # 光子3的速度方向
# ], dtype=np.float64) * c  # 乘以光速
#
# masses_photons = np.full(pos_photons.shape[0], h * 780e12 / c**2, dtype=np.float64)


# simulate_photon(n_bodies,masses,dt,positions,velocities,accelerations_pert_old,pos_photons,velocities_photons,masses_photons)
