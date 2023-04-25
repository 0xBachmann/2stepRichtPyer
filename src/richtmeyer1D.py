from PDE_Types import *
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2stepImplicit, Richtmeyer2step
from two_step_richtmeyer_util import Dimension, log
import numpy as np

log("definition of variables")

Type = PDE_Type.Euler
DIM = Dimension.oneD

if Type == PDE_Type.Linear_advection:
    F = LinearAdvection(np.array([0.5]), dim=DIM)
elif Type == PDE_Type.Burgers_equation:
    F = BurgersEq(dim=DIM)
elif Type == PDE_Type.Euler:
    F = Euler(5. / 3, dim=DIM)
else:
    raise "PDE_Type not known"

log("calculate initial conditions")

L = 1
stepper = Richtmeyer2step(F, np.array([L]), np.array([100]))#, eps=1e-16, method="root")


# TODO: initial values
def f(x):
    # return np.cos(2 * np.pi / L * x)
    # return np.exp(-(x - 3)**2)
    # return np.array(list(map(lambda x: 1 if 1 < x < 2 else 0, x)))
    # func = F.waves(1, np.array([1, 1, 1]), amp=1e-3)
    # return func(x / L)
    if isinstance(F, Euler):
        result = np.empty((*x.shape[:-1], 3))
        result[:25, 0] = 1
        result[25:75, 0] = 1/8
        result[75:, 0] = 1
        result[..., 1] = 0
        result[:25, 2] = 1
        result[25:75, 2] = 0.3
        result[75:, 2] = 1
        return F.primitive_to_conserved(result)
    if isinstance(F, LinearAdvection):
        # result = np.ones((*x.shape[:-1], 1))
        # result[:, 0] += 0.001 * np.sin(2 * np.pi / L * x[..., 0])
        return np.sin(2 * np.pi / L * x)


stepper.initial_cond(f)

plotter = Plotter(F, action="save", writeout=1, dim=stepper.dim,
                  coords=[stepper.coords[i][:-1] for i in range(stepper.dim.value)], filename="Euler_1D_impl.mp4")

T = 5
time = 0.
while time < T:
    dt = stepper.cfl()
    stepper.step(dt)

    plotter.write(stepper.grid_no_ghost, dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finalize()
