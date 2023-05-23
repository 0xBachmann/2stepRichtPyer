from PDE_Types import EulerScalarAdvect, Euler
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import *
from intitial import kelvin_helmholtz

import copy
import numpy as np

log("definition of variables")

DIM = Dimension.twoD

domain = np.array([[0, 1], [0, 1]])
resolution = np.array([128] * DIM.value)

h = ((domain[:, 1] - domain[:, 0]) / resolution).ravel()
F = EulerScalarAdvect(5. / 3, dim=DIM, c1=1., c2=1., hx=h[0], hy=h[1], add_viscosity=True)

log("calculate initial conditions")

# stepper = Richtmeyer2stepImplicit(F, domain, resolution, eps=1e-9)
stepper = Richtmeyer2step(F, domain, resolution, lerp=False)

center = np.array([0.5, 0.5])


def kh_with_scalar(x: np.ndarray, F, Mr, pr=2.5, rhor=1., primitives=False):
    primitive = np.empty((*x.shape[:-1], 5))
    primitive[..., :-1] = kelvin_helmholtz(x, F, Mr=Mr, pr=pr, rhor=rhor, primitives=True)

    Y = x[..., 1]
    primitive[(Y < 0.25) | (0.75 <= Y), -1] = 0
    primitive[(0.25 <= Y) & (Y < 0.75), -1] = 1

    if primitives:
        return primitive
    else:
        return F.primitive_to_conserved(primitive)


M = 0.001
stepper.initial_cond(lambda x: kh_with_scalar(x, F, Mr=M))


plot_visc = True
plot_curl = False
plot_eta = False
plot_mach = False

ncomps = F

if plot_mach:
    ncomps = 1
elif plot_curl:
    ncomps = 2
elif plot_visc:
    ncomps = 4
elif plot_eta:
    ncomps = 6

plotter = Plotter(ncomps, action="show", writeout=40, dim=stepper.dim, filename="kelvin_helmholz_eta_rel.mp4"
                  , lims={0: [-.4, .4], 3: [-.4, .4]}
                  )


def plot(dt):
    if plot_mach:
        Mach = F.mach(stepper.grid_no_ghost)[..., np.newaxis]
        plotter.write(Mach, dt)
    elif plotter.ncomp == 6:
        to_plot = np.empty((*stepper.grid_no_ghost.shape[:-1], 6))
        to_plot[..., :5] = F.conserved_to_primitive(stepper.grid_no_ghost)
        to_plot[..., 5] = F.eta_(stepper.grid, stepper.dxyz[0], stepper.dxyz[1])[..., 0] \
                          + np.float_(F.eta(avg_x(avg_y(stepper.grid)), stepper.dxyz[0], stepper.dxyz[1]))
        plotter.write(to_plot, dt)
    elif plot_curl:
        to_plot = np.empty((*stepper.grid_no_ghost.shape[:-1], 2))
        to_plot[..., 0] = curl(stepper.grid[..., 1:3], DIM, h)
        to_plot[..., 1] = stepper.grid_no_ghost[..., -1]
        plotter.write(to_plot, dt)
    elif plot_eta:
        # to_plot = np.empty((*stepper.grid_no_ghost.shape[:-1], 2))
        # to_plot[..., 0] = F.eta(stepper.grid, stepper.dxyz[0], stepper.dxyz[1])
        # to_plot[..., 1] = stepper.grid_no_ghost[..., -1]
        plotter.write(F.eta(stepper.grid, stepper.dxyz[0], stepper.dxyz[1])[..., np.newaxis], dt)
    elif plot_visc:
        to_plot = np.empty((*stepper.grid_no_ghost.shape[:-1], 5))
        to_plot[..., :4] = F.viscosity(stepper.grid_no_ghost).reshape((*stepper.grid_no_ghost.shape[:-1], 4))
        to_plot[..., 4] = F.mach(stepper.grid_no_ghost)
        plotter.write(to_plot, dt)
    else:
        plotter.write(F.conserved_to_primitive(stepper.grid_no_ghost), dt)
        # plotter.write(stepper.grid_no_ghost, dt)


plot(0)

fact = 1
T = 3
time = 0.
while time < T:
    dt = stepper.cfl() * fact
    stepper.step(dt)

    plot(dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finalize()
