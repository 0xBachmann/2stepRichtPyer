import itertools

from two_step_richtmeyer_util import *
from PDE_Types import *
from plotter import *

log("definition of variables")

Type = PDE_Type.Linear_advection
DIM = Dimension.twoD

if Type == PDE_Type.Linear_advection:
    F = LinearAdvection(np.array([10, 10]), dim=DIM)
elif Type == PDE_Type.Burgers_equation:
    F = BurgersEq(dim=DIM)
elif Type == PDE_Type.Euler:
    F = Euler(5. / 3, dim=DIM)
else:
    raise "PDE_Type not known"

Lx = 10
Ly = 10
dx = Lx / 100
dy = Ly / 100
ncellsx = int(Lx / dx)
ncellsy = int(Ly / dy)

grid = np.zeros((ncellsx + 2, ncellsy + 2, F.ncomp))  # pad with zero


# TODO: better solution for init
def f(x):
    # return np.array([1. + 0.001*np.sin(2 * np.pi / Lx * x[0]) * np.sin(2 * np.pi / Ly * x[1]), 1, 1, 2])
    # return np.cos(2 * np.pi / L * x)
    return np.exp(-np.linalg.norm(x - np.array([3, 3])) ** 2)
    # return np.array(list(map(lambda x: 1 if 1 < x < 2 else 0, x)))


log("calculate initial conditions")

# initial conditions
coords_x = np.linspace(0, Lx, ncellsx + 1)
coords_y = np.linspace(0, Ly, ncellsy + 1)
X, Y = np.meshgrid(coords_x[:-1], coords_y[:-1])
# TODO takes too long
for i, j in itertools.product(range(ncellsx), range(ncellsy)):
    grid[i + 1, j + 1, :] = f(np.array([coords_x[i] + coords_x[i + 1], coords_y[j] + coords_y[j + 1]]) / 2)

plotter = Plotter(F, grid[1:-1, 1:-1, :], action="show", writeout=1, dim=DIM, coords=[coords_x, coords_y])

log("start time evaluation")

T = 1
traj: list[np.ndarray] = []
time = 0
while time < T:
    Fprime = F.derivative(grid[1:-1, 1:-1, ...])
    ax = np.max(np.abs(Fprime[..., 0]))
    ay = np.max(np.abs(Fprime[..., 1]))
    dt = dx / (2 * max(ax, ay))  # damp?
    time += dt
    print(f"dt = {dt}, time = {time:.3f}/{T}")

    c = dt / dx

    # TODO: Boundary conditions?
    pbc(grid, dim=DIM)

    staggered_grid = avg_x(avg_y(grid))  # get average

    Fgrid, Ggrid = F(grid)
    staggered_grid -= c / 2 * (del_x(avg_y(Fgrid)) + del_y(avg_x(Ggrid)))

    Fstaggered_grid, Gstaggered_grid = F(staggered_grid)
    grid[1:-1, 1:-1, :] -= c * (del_x(avg_y(Fstaggered_grid)) + del_y(avg_x(Gstaggered_grid)))

    plotter.write(grid[1:-1, 1:-1, :], dt)

log("generating plots")

plotter.finlaize()

log("finished")
