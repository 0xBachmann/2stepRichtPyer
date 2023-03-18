from two_step_richtmeyer_util import *
from PDE_Types import *
from plotter import *
from richtmeyer_two_step_scheme import Richtmeyer2step

log("definition of variables")

Type = PDE_Type.Euler
DIM = Dimension.oneD

if Type == PDE_Type.Linear_advection:
    F = LinearAdvection(np.array([3]), dim=DIM)
elif Type == PDE_Type.Burgers_equation:
    F = BurgersEq(dim=DIM)
elif Type == PDE_Type.Euler:
    F = Euler(5. / 3, dim=DIM)
else:
    raise "PDE_Type not known"

log("calculate initial conditions")

L = 1
stepper = Richtmeyer2step(F, np.array([L]), np.array([100]))


# TODO: initial values
def f(x):
    # return np.array([1. + 0.1*np.sin(2 * np.pi / L * x), 1, 1])
    # return np.cos(2 * np.pi / L * x)
    # return np.exp(-(x - 3)**2)
    # return np.array(list(map(lambda x: 1 if 1 < x < 2 else 0, x)))
    func = F.waves(0, np.array([1, 1, 1]), amp=1e-3)
    return func(x / L, 0)


stepper.initial_cond(f)

plotter = Plotter(F, action="save", writeout=1, dim=stepper.dim,
                  coords=[stepper.coords[i][:-1] for i in range(stepper.dim.value)])

T = 1
time = 0.
traj = []
while time < T:
    dt = stepper.cfl()
    stepper.step(dt)

    plotter.write(stepper.grid_no_ghost, dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finlaize()
