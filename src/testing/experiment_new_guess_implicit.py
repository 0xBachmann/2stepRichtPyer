from src.PDE_Types import *
from src.plotter import Plotter
from src.richtmyer_two_step_scheme import Richtmyer2step, Richtmyer2stepImplicit
from src.two_step_richtmyer_util import Dimension, log, avg_x
from src.intitial import gresho_vortex

import copy
import numpy as np

log("definition of variables")

DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)

log("calculate initial conditions")

resolution = np.array([80] * DIM.value)
Lx = 1
Ly = Lx
stepper = Richtmyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-8)
stepper_explicit = Richtmyer2step(F, np.array([Lx, Ly]), resolution)


center = np.array([Lx / 2, Ly / 2])

M = 0.1
t = 1.
stepper.initial_cond(lambda x: gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi * Lx / 1))

plotter = Plotter(1, action="show", writeout=1, dim=stepper.dim)


def plot(stepper, dt):
    if plotter.ncomp == 1:
        plotter.write(F.mach(stepper.grid_no_ghost)[..., np.newaxis], dt)
    else:
        # plotter.write(F.conserved_to_primitive(stepper.grid_no_ghost), dt)
        plotter.write(stepper.grid_no_ghost, dt)


fact = 10
T = 1
time = 0.
while time < T:
    dt = stepper.cfl() * fact
    # stepper_explicit.grid_no_ghost = stepper.grid_no_ghost
    # stepper_explicit.step(dt)
    stepper.step(dt)#, guess=stepper_explicit.grid_no_ghost)

    # plot(stepper, dt)

    time += min(dt, T - time)
    print(f"dt = {dt}, time = {time:.3f}/{T}")

# plotter.finalize()
