from PDE_Types import *
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import Dimension, log

import copy
import numpy as np

log("definition of variables")

DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)

log("calculate initial conditions")

resolution = np.array([40] * DIM.value)
Lx = 1
Ly = Lx
stepper = Richtmeyer2step(F, np.array([Lx, Ly]), resolution)

alpha = None


# TODO: initial values
def gresho_vortex(x: np.ndarray, Mmax=None, period=1) -> np.ndarray:
    primitive = np.empty((*x.shape[:-1], 4))

    # offset domain
    center = np.array([Lx / 2, Ly / 2])
    X = x[..., 0] - center[0]
    Y = x[..., 1] - center[1]

    # get angle
    global alpha
    alpha = np.arctan2(Y, X)

    # calculate radius
    r2 = X ** 2 + Y ** 2
    r = np.sqrt(r2)

    # density
    rho = 1.
    primitive[..., 0] = rho

    # reference speed
    qr = 0.4 * np.pi * Lx / period

    # define u_phi
    u = np.zeros(x.shape[:-1])
    inner_ring = r < 0.2
    u[inner_ring] = 5. * r[inner_ring] * qr
    outer_ring = (0.2 <= r) & (r < 0.4)
    u[outer_ring] = (2. - 5. * r[outer_ring]) * qr

    # split u_phi int ux and uy
    primitive[..., 1] = u * -np.sin(alpha) # or (-Y/r)
    primitive[..., 2] = u * np.cos(alpha)  # or (+X/r)
    gamma = F.gamma

    # background pressure
    if Mmax is not None:
        p0 = rho / (gamma * Mmax ** 2) - 0.5
    else:
        p0 = 5.

    primitive[..., 3] = p0

    # pressure disturbance
    inner_ring = r < 0.4
    primitive[inner_ring, 3] += 25. / 2 * r2[inner_ring]
    primitive[outer_ring, 3] += 4. * (1. - 5. * r[outer_ring] - np.log(0.2) + np.log(r[outer_ring]))
    primitive[r >= 0.4, 3] += -2. + 4. * np.log(2)

    # convert to conserved variables
    return F.primitive_to_conserved(primitive)


def u_phi(grid: np.ndarray) -> np.ndarray:
    vels = stepper.grid_no_ghost[..., 1:3] / stepper.grid_no_ghost[..., 0:1]
    u = -vels[..., 0] * np.sin(alpha) + vels[..., 1] * np.cos(alpha)
    return u


M = 0.1
t = 1
stepper.initial_cond(lambda x: gresho_vortex(x, M))

plotter = Plotter(1, action="show", writeout=1, dim=stepper.dim)

if plotter.ncomp == 1:
    # plotter.write(alpha[..., np.newaxis], 0)
    # plotter.write(np.sqrt(stepper.grid_no_ghost[..., 1] ** 2 + stepper.grid_no_ghost[..., 2] ** 2)[..., np.newaxis], 0)
    plotter.write(u_phi(stepper.grid_no_ghost)[..., np.newaxis], 0)
else:
    # plotter.write(F.conserved_to_primitive(stepper.grid_no_ghost), 0)
    plotter.write(stepper.grid_no_ghost, 0)

T = t * 1
time = 0.05
while time < T:
    dt = stepper.cfl() / 4
    stepper.step(dt)

    if plotter.ncomp == 1:
        # plotter.write(np.sqrt(stepper.grid_no_ghost[..., 1] ** 2 + stepper.grid_no_ghost[..., 2] ** 2)[..., np.newaxis], dt)
        plotter.write(u_phi(stepper.grid_no_ghost)[..., np.newaxis], 0)
    else:
        # plotter.write(F.conserved_to_primitive(stepper.grid_no_ghost), dt)
        plotter.write(stepper.grid_no_ghost, dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finalize()
