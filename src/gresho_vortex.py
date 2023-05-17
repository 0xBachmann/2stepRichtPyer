from PDE_Types import *
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import Dimension, log, avg_x
from intitial import gresho_vortex

import copy
import numpy as np

log("definition of variables")

DIM = Dimension.twoD
resolution = np.array([80] * DIM.value)
Lx = 1
Ly = Lx
h = [Lx / resolution[0], Ly / resolution[1]]
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=0., hx=h[0], hy=h[1], add_viscosity=False)

log("calculate initial conditions")

# stepper = Richtmeyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-8)
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


M = 0.1
t = 1.
stepper.initial_cond(lambda x: gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi * Lx / 1))

plotter = Plotter(1, action="show", writeout=100, dim=stepper.dim, filename="gresho_vortex_iml_100_hybr_eps1e-9.mp4")

plot_visc = True


def plot(stepper, dt, plot_mach=False, plot_curl=True):
    if plotter.ncomp == 1:
        if plot_mach:
            plotter.write(F.mach(stepper.grid_no_ghost)[..., np.newaxis], dt)
        elif plot_curl:
            plotter.write(curl(stepper.grid[..., 1:3], DIM, h)[..., np.newaxis], dt)
        else:
            plotter.write(u_phi(stepper.grid_no_ghost)[..., np.newaxis], dt)
    elif plotter.ncomp == 2:
        to_plot = np.empty((*stepper.grid_no_ghost.shape[:-1], 2))
        to_plot[..., 0] = F.mach(stepper.grid_no_ghost)
        to_plot[..., 1] = F.eta(stepper.grid, stepper.dxyz[0], stepper.dxyz[1])[..., 0]
        plotter.write(to_plot, dt)
    elif plot_visc:
        to_plot = np.empty((*stepper.grid_no_ghost.shape[:-1], 6))
        to_plot[..., 0:4] = F.viscosity(stepper.grid_no_ghost).reshape((*stepper.grid_no_ghost.shape[:-1], 4))
        to_plot[..., 4] = F.mach(stepper.grid_no_ghost)
        plotter.write(to_plot, dt)
    else:
        # plotter.write(F.conserved_to_primitive(stepper.grid_no_ghost), dt)
        plotter.write(stepper.grid_no_ghost, dt)


stepper.step_for(t, fact=1, callback=plot)

plotter.finalize()
