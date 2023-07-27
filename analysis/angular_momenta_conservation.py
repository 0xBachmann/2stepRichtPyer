from src.plotting_setup import *
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
# stepper = Richtmyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-8)
stepper = Richtmyer2step(F, np.array([Lx, Ly]), resolution)

center = np.array([Lx / 2, Ly / 2])

M = 1.e-1
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

j = []
times = []

def plot(stepper, dt):
    global j, times
    j.append(np.sum(F.angular_momenta(stepper.grid_no_ghost, XY - center)))
    times.append(dt)


stepper.step_for(t, fact=1, callback=plot)

plt.plot(np.cumsum(times), np.array(j) - j[0])
plt.show()
