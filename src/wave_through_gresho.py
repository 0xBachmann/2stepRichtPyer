from PDE_Types import Euler
from plotter import Plotter
from richtmyer_two_step_scheme import Richtmyer2step, Richtmyer2stepImplicit
from two_step_richtmyer_util import Dimension, log
from intitial import gresho_vortex, sound_wave_packet

from plotting_setup import *

import numpy as np

log("definition of variables")

DIM = Dimension.twoD
c = 1.
h = 1./40
F = Euler(5. / 3, dim=DIM, c1=c/10, c2=c, hx=h, hy=h, add_viscosity=0, mu=1e-4)

log("calculate initial conditions")

domain = np.array([[-4, 3], [0, 1]])
res_per_unit = 40
resolution = np.array([res_per_unit * (domain[0, 1] - domain[0, 0]), res_per_unit])
stepper = Richtmyer2stepImplicit(F, domain, resolution)  # , lerp=3)

center = np.array([0.5, 0.5])

M = 1e-3
t = 1
stepper.initial_cond(lambda x: F.primitive_to_conserved(
    gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi, primitives=True)
    + sound_wave_packet(x, F, -0.5, Mmax=M, alpha=0.1**-2, primitives=True, qr=0.4 * np.pi)))

lims = {0: [-4, -2]}
plotter = Plotter(1, action="show", writeout=1, dim=stepper.dim, lims=lims, filename="wave_through_gresho.mp4")


def plot(dt):
    if plotter.ncomp == 1:
        log10M = np.log10(F.mach(stepper.grid_no_ghost))[..., np.newaxis]
        log10M[log10M < -4] = -4
        plotter.write(log10M[(-1 - domain[0, 0]) * res_per_unit:-(domain[0, 1] - 1) * res_per_unit], dt)
        # plotter.write(F.pres(stepper.grid_no_ghost)[..., np.newaxis], dt)
    else:
        plotter.write(F.conserved_to_primitive(stepper.grid_no_ghost[(-1 - domain[0, 0]) * res_per_unit:-(domain[0, 1] - 1) * res_per_unit]), dt)
        # plotter.write(stepper.grid_no_ghost, dt)


plot(0)

fact = 1
# T = 7.5 * 1e-4
T = 3.2 * 1e-3
T = 0.3
time = 0.
while time < T:
    dt = stepper.cfl() * fact
    stepper.step(dt)

    plot(dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finalize()
