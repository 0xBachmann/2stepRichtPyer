from PDE_Types import *
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import Dimension, log, avg_x
from intitial import gresho_vortex

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

center = np.array([Lx / 2, Ly / 2])
avg_coords = [avg_x(coord) for coord in stepper.coords]
X, Y = np.meshgrid(*avg_coords, indexing='ij')
X -= center[0]
Y -= center[1]

# get angle
alpha = np.arctan2(Y, X)


def u_phi(grid: np.ndarray) -> np.ndarray:
    vels = stepper.grid_no_ghost[..., 1:3] / stepper.grid_no_ghost[..., 0:1]
    u = vels[..., 0] * np.sin(alpha) - vels[..., 1] * np.cos(alpha)
    return u


M = 0.01
t = 1
stepper.initial_cond(lambda x: gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi * Lx / 1))

plotter = Plotter(1, action="saveb", writeout=10, dim=stepper.dim, filename="gresho_vortex.mp4")


def plot(dt, plot_mach=True):
    if plotter.ncomp == 1:
        if plot_mach:
            plotter.write(F.mach(stepper.grid_no_ghost)[..., np.newaxis], dt)
        else:
            plotter.write(u_phi(stepper.grid_no_ghost)[..., np.newaxis], dt)
    else:
        # plotter.write(F.conserved_to_primitive(stepper.grid_no_ghost), dt)
        plotter.write(stepper.grid_no_ghost, dt)


plot(0)

fact = 1
T = t * fact
time = 0.
while time < T:
    dt = stepper.cfl() * fact
    stepper.step(dt)

    plot(dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finalize()
