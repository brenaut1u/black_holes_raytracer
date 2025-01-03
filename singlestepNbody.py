#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:47:20 2024

@author: andypan
"""

import os  # For file management
import numpy as np
from numba import njit, prange
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utilities import *

@njit(parallel=True)
def compute_accelerations_and_jerks(positions, velocities, masses, accelerations_pert_old, dt):
    """
    Compute accelerations and jerks using Newtonian gravity and 1PN corrections.
    """
    n = len(masses)
    accelerations = np.zeros_like(positions)
    jerks = np.zeros_like(positions)
    accelerations_pert = np.zeros_like(positions)

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
                r_ij += eps_bodies*scale


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
                        r_jk += eps_bodies*scale

                        r_ik2 = x_ik @ x_ik
                        r_ik = np.sqrt(r_ik2)
                        r_ik += eps_bodies*scale

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
    # Compute acceleration and jerks
    accelerations, jerks, accelerations_pert = compute_accelerations_and_jerks(positions, velocities, masses, accelerations_pert_old, dt)

    # Update positions and velocities
    velocities += accelerations * dt + 1/2 * jerks * dt**2
    speed = np.sqrt(np.sum(velocities**2,axis=1))
    speed_capped = np.clip(speed,0,c)
    for i in prange(len(speed)):
        if masses[i] <= 10*h*nu/c**2:
            velocities[i,:] *= c/np.sqrt(velocities[i,:]@velocities[i,:])
        else:
            velocities[i,:] *= speed_capped[i] / speed[i]
    positions += velocities * dt + 1 / 2 * accelerations * dt**2 + 1/6 * jerks * dt**3

    return positions, velocities, accelerations_pert

def simulate(n_bodies,masses,dt,positions,velocities,accelerations_pert_old):

    positions, velocities, accelerations_pert_old = hermite_integrator(positions, velocities, masses,accelerations_pert_old, dt)

    return n_bodies,masses,positions, velocities, accelerations_pert_old