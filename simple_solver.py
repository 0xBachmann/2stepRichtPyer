import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from enum import Enum
from scipy.integrate import quadrature
from copy import deepcopy


class PDE_Type(Enum):
    Linear_advection = 1
    Burgers_equation = 2
    Euler = 3


Type = PDE_Type.Linear_advection


class LinearAdvection:
    def __init__(self, a):
        self.a = a

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.a * v

    def derivative(self, v: np.ndarray) -> np.ndarray:
        return np.full(v.shape, self.a)


class BurgersEq:
    def __call__(self, v: np.ndarray) -> np.ndarray:
        return 0.5 * np.square(v)

    def derivative(self, v: np.ndarray) -> np.ndarray:
        return v


class Euler:
    def __call__(self, v: np.ndarray) -> np.ndarray:
        # define p and E
        p = 0
        E = 0
        res = np.ndarray(v.shape)
        res[0, :] = np.multiply(v[0, :], v[1, :])
        res[1, :] = np.multiply(v[0, :], np.square(v[1, :])) + p
        res[2, :] = np.multiply(E + p, v[1, :])
        return res

    # TODO
    def derivative(self, v: np.ndarray) -> np.ndarray:
        return np.zeros(v.shape)


if Type == PDE_Type.Linear_advection:
    F = LinearAdvection(2)
elif Type == PDE_Type.Burgers_equation:
    F = BurgersEq()
elif Type == PDE_Type.Euler:
    F = Euler()
else:
    raise "PDE_Type not known"

L = 10
dx = 0.001
ncells = int(L / dx)

grid = np.zeros(ncells + 2)  # pad with zero


# TODO: initial values
def f(x):
    # return np.sin(2 * np.pi / L * x)
    return np.exp(-(x - 3) ** 2)


# initial conditions
domain = np.linspace(0, L, ncells + 1)
for i in range(ncells):
    grid[i + 1] = quadrature(f, domain[i], domain[i + 1])[0] / dx

staggerd_grid = (grid[1:] - grid[:-1]) / 2

T = 1
# TODO: derive CFL
Fprime = F.derivative(grid)
a = np.max(np.abs(Fprime))  # TODO maybe every step?
dt = dx / (2 * a) * 0.8  # damp?
nsteps = int(T / dt)

c = dt / dx

fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
frame, = ax.plot([], [])


def pbc(grid: np.ndarray):
    grid[0] = grid[-2]
    grid[-1] = grid[1]


traj = []
for _ in range(nsteps):
    # TODO: Boundary conditions?
    pbc(grid)
    staggerd_grid -= c / 2 * (F(grid[1:]) - F(grid[:-1]))
    grid[1:-1] -= c * (F(staggerd_grid[1:]) - F(staggerd_grid[:-1]))
    # grid[1:-1] -= c * (F(grid[2:]) - F(grid[:-2]))
    traj.append(deepcopy(grid[1:-1]))

stride = 100


def step(time):
    frame.set_data(domain[:-1], traj[time * stride])
    ax.set_title("%.4f s" % (time * dt))
    return frame,


ani = FuncAnimation(fig, step, frames=int(nsteps / stride), blit=False)
plt.show()
