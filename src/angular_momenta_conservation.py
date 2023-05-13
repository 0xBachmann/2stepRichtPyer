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

resolution = np.array([80] * DIM.value)
Lx = 1
Ly = Lx
# stepper = Richtmeyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-8)
stepper = Richtmeyer2step(F, np.array([Lx, Ly]), resolution)

center = np.array([Lx / 2, Ly / 2])

M = 0.1
t = 1.
stepper.initial_cond(lambda x: gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi * Lx / 1))

avg_coords = [avg_x(coord) for coord in stepper.coords]
X, Y = np.meshgrid(*avg_coords, indexing='ij')
XY = np.empty((*resolution, 2))
XY[..., 0] = X
XY[..., 1] = Y
# X -= center[0]
# Y -= center[1]

plotter = Plotter(1, action="show", writeout=100, dim=stepper.dim, filename="gresho_vortex_iml_100_hybr_eps1e-9.mp4")


def plot(stepper, dt):
    return plotter.write(F.angular_momenta(stepper.grid_no_ghost, XY - center)[..., np.newaxis], dt)


stepper.step_for(t, fact=1, callback=plot)

plotter.finalize()
