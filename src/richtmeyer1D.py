import itertools

from two_step_richtmeyer_util import *
from PDE_Types import *
from plotter import *

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
L = 10
dx = L / 1000
ncells = int(L / dx)

grid = np.zeros((ncells + 2, F.ncomp))  # pad with zero



# TODO: initial values
def f(x):
    # return np.array([1. + 0.1*np.sin(2 * np.pi / L * x), 1, 1])
    # return np.cos(2 * np.pi / L * x)
    # return np.exp(-(x - 3)**2)
    # return np.array(list(map(lambda x: 1 if 1 < x < 2 else 0, x)))
    func = F.waves(0, np.array([1, 1, 1]), 1e-3)
    return func(x, 0)


# initial conditions
domain = np.linspace(0, L, ncells + 1)
for i in range(ncells):
    grid[i + 1, :] = f((domain[i] + domain[i + 1])/2) #quadrature(f, domain[i], domain[i + 1])[0] / dx

plotter = Plotter(F, grid[1:-1, :], action="show", writeout=1, dim=DIM, coords=[domain[:-1]])

pbc(grid, dim=DIM)

T = 100
# TODO: derive CFL
Fprime = F.derivative(grid)
a = np.max(np.abs(Fprime))  # TODO maybe every step?

dt = dx / (2 * a) # damp?
nsteps = int(T / dt)

c = dt / dx




traj = []
for _ in range(nsteps):
    # TODO: Boundary conditions?
    pbc(grid, dim=DIM)
    staggerd_grid = avg_x(grid)  # get average

    Fgrid, = F(grid)
    staggerd_grid -= c / 2 * del_x(Fgrid)

    Fstaggered_grid, = F(staggerd_grid)
    grid[1:-1, :] -= c * del_x(Fstaggered_grid)

    plotter.write(grid[1:-1, :], dt)

plotter.finlaize()


