from PDE_Types import Euler
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import Dimension, log
from intitial import kelvin_helmholtz

import copy
import numpy as np

log("definition of variables")

DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)

log("calculate initial conditions")

domain = np.array([[0, 1], [0, 1]])
resolution = np.array([1024] * DIM.value)
stepper = Richtmeyer2step(F, domain, resolution)

center = np.array([0.5, 0.5])

M = 0.001
t = 1
stepper.initial_cond(lambda x: kelvin_helmholtz(x, F, Mr=0.01))

plotter = Plotter(F, action="save", writeout=100, dim=stepper.dim, filename="kelvin_helmholz.mp4")


def plot(dt):
    if plotter.ncomp == 1:
        Mach = F.mach(stepper.grid_no_ghost)[..., np.newaxis]
        plotter.write(Mach, dt)
        # plotter.write(F.pres(stepper.grid_no_ghost)[..., np.newaxis], dt)
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