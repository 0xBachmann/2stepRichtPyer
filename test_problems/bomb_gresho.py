from src.PDE_Types import *
from src.plotter import Plotter
from src.richtmyer_two_step_scheme import Richtmyer2step, Richtmyer2stepImplicit
from src.two_step_richtmyer_util import Dimension, log, avg_x
from src.intitial import gresho_vortex, isentropic_vortices

import copy
import numpy as np

log("definition of variables")

DIM = Dimension.twoD
l = 10
domain = np.array([[-l, l], [-l, l]])
resolution = np.array([160] * DIM.value)

h = ((domain[:, 1] - domain[:, 0]) / resolution).ravel()
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=0., hx=h[0], hy=h[1], add_viscosity=False)

log("calculate initial conditions")

# stepper = Richtmyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-8)
stepper = Richtmyer2step(F, domain, resolution, lerp=False)

center = np.array([1, 1])
avg_coords = [avg_x(coord) for coord in stepper.coords]

M = 0.1
t = 6.


def gresho_bomb(x):
    result = gresho_vortex(x, center, F, M, qr=0.4 * np.pi, primitives=True)
    midx = resolution[0] // 2
    midy = resolution[1] // 2
    result[midx - 3:midx + 4, midy - 3:midy + 4, -1] *= 2
    return F.primitive_to_conserved(result)

def vortex_bomb(x, bcenter=np.array([0, 0]), n_v=3):
    # centers = [np.array([np.sin(i * 2 * np.pi / n_v), np.cos(i * 2 * np.pi / n_v)]) for i in range(n_v)]
    # vortices = gresho_vortex(x, centers[0], F, M, qr=0.4 * np.pi, primitives=True)
    # for i in range(1, n_v):
    #     vortices += gresho_vortex(x, centers[i], F, M, qr=0.4 * np.pi, primitives=True, background=False)
    vortices = isentropic_vortices(x,
        [l / 2 * np.array([np.sin(i * 2 * np.pi / n_v), np.cos(i * 2 * np.pi / n_v)]) for i in range(n_v)], beta=5, F=F, primitives=True)
    X = x[..., 0] - bcenter[0]
    Y = x[..., 1] - bcenter[1]
    vortices[..., -1] += 2 * np.exp(-3 * (X**2 + Y**2))
    return F.primitive_to_conserved(vortices)


stepper.initial_cond(vortex_bomb)

plotter = Plotter(1, action="show", writeout=1, dim=stepper.dim, filename="gresho_bomb.mp4", lims={0: [0, 1.1]})


def plot(stepper, dt, plot_mach=False, plot_curl=False, plot_eta=False, plot_visc=False):
    if plotter.ncomp == 1:
        if plot_mach:
            plotter.write(F.mach(stepper.grid_no_ghost)[..., np.newaxis], dt)
        elif plot_curl:
            plotter.write(curl(stepper.grid[..., 1:3], DIM, h)[..., np.newaxis], dt)
        elif plot_eta:
            plotter.write(F.eta(stepper.grid, h[0], h[1])[..., np.newaxis], dt)
        else:
            plotter.write(stepper.grid_no_ghost[..., 0][..., np.newaxis], dt)
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
