from PDE_Types import *
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import Dimension, log, avg_x
from intitial import gresho_vortex

import copy
import numpy as np

log("definition of variables")

DIM = Dimension.twoD
domain = np.array([[-3, 3], [-3, 3]])
resolution = np.array([160] * DIM.value)

h = ((domain[:, 1] - domain[:, 0]) / resolution).ravel()
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=0., hx=h[0], hy=h[1], add_viscosity=False)

log("calculate initial conditions")

# stepper = Richtmeyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-8)
stepper = Richtmeyer2step(F, domain, resolution, lerp=False)

center = np.array([1, 1])
avg_coords = [avg_x(coord) for coord in stepper.coords]

M = 0.1
t = 1.


def gresho_bomb(x):
    result = gresho_vortex(x, center, F, M, qr=0.4 * np.pi, primitives=True)
    midx = resolution[0] // 2
    midy = resolution[1] // 2
    result[midx - 3:midx + 4, midy - 3:midy + 4, -1] *= 2
    return F.primitive_to_conserved(result)


stepper.initial_cond(gresho_bomb)

plotter = Plotter(1, action="show", writeout=1, dim=stepper.dim, filename="gresho_bomb.mp4")


def plot(stepper, dt, plot_mach=True, plot_curl=False, plot_eta=False, plot_visc=True):
    if plotter.ncomp == 1:
        if plot_mach:
            plotter.write(F.mach(stepper.grid_no_ghost)[..., np.newaxis], dt)
        elif plot_curl:
            plotter.write(curl(stepper.grid[..., 1:3], DIM, h)[..., np.newaxis], dt)
        elif plot_eta:
            plotter.write(F.eta(stepper.grid, h[0], h[1])[..., np.newaxis], dt)
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
