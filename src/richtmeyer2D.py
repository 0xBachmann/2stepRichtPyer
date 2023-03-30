from PDE_Types import *
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step
from two_step_richtmeyer_util import Dimension, log

log("definition of variables")

Type = PDE_Type.Euler
DIM = Dimension.twoD

if Type == PDE_Type.Linear_advection:
    F = LinearAdvection(np.array([3, -15]), dim=DIM)
elif Type == PDE_Type.Burgers_equation:
    F = BurgersEq(dim=DIM)
elif Type == PDE_Type.Euler:
    F = Euler(5. / 3, dim=DIM)
else:
    raise "PDE_Type not known"

log("calculate initial conditions")

Lx = 1
Ly = 1
stepper = Richtmeyer2step(F, np.array([Lx, Ly]), np.array([50, 50]))


# TODO: initial values
def f(x):
    if Type == PDE_Type.Euler:
        if True:
            func = F.waves(1, np.array([1, 1, 1]), amp=1e-3, alpha=np.pi / 4)
            # TODO not normalized in case of Lx/Ly != 1
            return func(x)

        if False:
            rho = 0.1
            delta = 0.05
            init = np.empty(F.ncomp)
            init[0] = 1
            if x[1] < Ly / 2:
                init[1] = np.tanh(2 * np.pi / rho * (x[1] - 0.25))
            else:
                init[1] = np.tanh(2 * np.pi / rho * (0.75 - x[1]))
            init[2] = delta * np.sin(2 * np.pi * x[0])
            init[3] = 1
            return init

        if False:
            init = np.empty(F.ncomp)
            init[0] = 1 + 0.01 * np.sin(2 * np.pi / Lx * x[0]) * np.sin(2 * np.pi / Ly * x[1])
            init[1] = 1
            init[2] = 1
            init[3] = 2
            return init

    else:
        return np.exp(-np.linalg.norm((x - np.array([0.3, 0.3])))**2 / 0.01)


stepper.initial_cond(f)

plotter = Plotter(F, action="show", writeout=1, dim=stepper.dim,
                  coords=[stepper.coords[i][:-1] for i in range(stepper.dim.value)])

T = 1
time = 0.
while time < T:
    dt = stepper.cfl()
    stepper.step(dt)

    plotter.write(stepper.grid_no_ghost, dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finalize()
