from PDE_Types import EulerScalarAdvect, Euler
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import Dimension, log
from intitial import kelvin_helmholtz

import copy
import numpy as np

log("definition of variables")

DIM = Dimension.twoD
F = EulerScalarAdvect(5. / 3, dim=DIM)

log("calculate initial conditions")

domain = np.array([[0, 1], [0, 1]])
resolution = np.array([32] * DIM.value)
stepper = Richtmeyer2stepImplicit(F, domain, resolution, method="hybr")

center = np.array([0.5, 0.5])


def kh_with_scalar(x: np.ndarray, F, Mr, pr=2.5, rhor=1., primitives=False):
    primitive = np.empty((*x.shape[:-1], 5))
    primitive[..., :-1] = kelvin_helmholtz(x, F, Mr=Mr, pr=pr, rhor=rhor, primitives=True)

    Y = x[..., 1]
    primitive[(Y < 0.25) | (0.75 <= Y), -1] = 0
    primitive[(0.25 <= Y) & (Y < 0.75), -1] = 1
    print(primitive.shape)

    if primitives:
        return primitive
    else:
        return F.primitive_to_conserved(primitive)


M = 0.1
t = 1
stepper.initial_cond(lambda x: kh_with_scalar(x, F, Mr=0.01))

plotter = Plotter(F, action="show", writeout=1, dim=stepper.dim, filename="kelvin_helmholz.mp4")


def plot(dt):
    if plotter.ncomp == 1:
        # Mach = F.mach(stepper.grid_no_ghost)[..., np.newaxis]
        # plotter.write(Mach, dt)
        plotter.write(stepper.grid_no_ghost[..., -1][..., np.newaxis], dt)
    else:
        plotter.write(F.conserved_to_primitive(stepper.grid_no_ghost), dt)
        # plotter.write(stepper.grid_no_ghost, dt)


plot(0)

fact = 100
T = 3
time = 0.
while time < T:
    dt = stepper.cfl() * fact
    stepper.step(dt)

    plot(dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finalize()
