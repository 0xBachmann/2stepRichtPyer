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
dx = L / 100
ncells = int(L / dx)

grid = np.zeros((ncells + 2, F.ncomp))  # pad with zero



# TODO: initial values
def f(x):
    # return np.array([1. + 0.1*np.sin(2 * np.pi / L * x), 1, 1])
    # return np.cos(2 * np.pi / L * x)
    # return np.exp(-(x - 3)**2)
    # return np.array(list(map(lambda x: 1 if 1 < x < 2 else 0, x)))
    func = F.waves(1, np.array([1, 1, 1]), 1e-3)
    return func(x/L, 0)


# initial conditions
domain = np.linspace(0, L, ncells + 1)
grid[1:-1, :] = f(avg_x(domain))
# for i in range(ncells):
#     grid[i + 1, :] = f((domain[i] + domain[i + 1])/2) #quadrature(f, domain[i], domain[i + 1])[0] / dx

plotter = Plotter(F, grid[1:-1, :], action="show", writeout=1, dim=DIM, coords=[domain[:-1]])





T = 100
time = 0
traj = []
while time < T:
    # TODO: Boundary conditions?
    pbc(grid, dim=DIM)
    Fprime = F.derivative(grid[1:-1, :])
    a = np.max(np.abs(Fprime))

    dt = dx / (2 * a)  # damp?
    c = dt / dx
    print(f"dt = {dt}, time = {time:.3f}/{T}")


    staggered_grid = avg_x(grid)  # get average

    Fgrid, = F(grid)
    staggered_grid -= c / 2 * del_x(Fgrid)

    Fstaggered_grid, = F(staggered_grid)
    grid[1:-1, :] -= c * del_x(Fstaggered_grid)


    plotter.write(grid[1:-1, :], dt)

    time += dt

plotter.finlaize()


