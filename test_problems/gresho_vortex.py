from src.PDE_Types import *
from src.plotter import Plotter
from src.richtmyer_two_step_scheme import Richtmyer2step, Richtmyer2stepImplicit
from src.two_step_richtmyer_util import Dimension, log, avg_x
from src.intitial import gresho_vortex
import sys
import time
import numpy as np

log("definition of variables")

DIM = Dimension.twoD
resolution = np.array([40] * DIM.value)
Lx = 1
Ly = Lx
h = [Lx / resolution[0], Ly / resolution[1]]
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=0., hx=h[0], hy=h[1])

log("calculate initial conditions")

stepper = Richtmyer2stepImplicit(F, np.array([Lx, Ly]), resolution)  # , lerp=-1)  # works: 1, 3. doesn't: 0, 2
domain = np.array([Lx, Ly])
# stepper = Richtmyer2step(F, domain, resolution, lerp=False)

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


M = 1.e-2
t = 1.
stepper.initial_cond(lambda x: gresho_vortex(x, center, F, M, qr=0.4 * np.pi * Lx))

plotter = Plotter(1, action="show", writeout=100, dim=stepper.dim, filename="gresho_vortex_eta_rel.mp4")

plot_visc = False


def plot(stepper, dt, plot_mach=True, plot_curl=False, plot_eta=False):
    if plotter.ncomp == 1:
        if plot_mach:
            plotter.write(F.mach(stepper.grid_no_ghost)[..., np.newaxis], dt)
        elif plot_curl:
            plotter.write(curl(stepper.grid[..., 1:3], DIM, h)[..., np.newaxis], dt)
        elif plot_eta:
            plotter.write(F.eta(stepper.grid, h[0], h[1])[..., np.newaxis], dt)
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


if isinstance(stepper, Richtmyer2step):
    fact = 1.
else:
    fact = 1. / M
if len(sys.argv) >= 2:
    fact = float(sys.argv[1])
start = time.time()
stepper.step_for(t, fact=fact, callback=plot)
end = time.time()

print(f"finished, fact={fact}")
print(f"took {end - start}s")

plotter.finalize()
