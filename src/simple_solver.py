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


Type = PDE_Type.Burgers_equation


class LinearAdvection:
    def __init__(self, a):
        self.a = a
        self.ncomp = 1

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.a * v

    def derivative(self, v: np.ndarray) -> np.ndarray:
        return np.full(v.shape, self.a)


class BurgersEq:
    def __init__(self):
        self.ncomp = 1
    def __call__(self, v: np.ndarray) -> np.ndarray:
        return 0.5 * np.square(v)

    def derivative(self, v: np.ndarray) -> np.ndarray:
        return v


class Euler:
    def __init__(self, gamma):
        self.gamma = gamma
        self.ncomp = 3
    def pres(self, v):
        dens = v[...,0]
        momx = v[...,1]
        Etot = v[...,2]
        eint = Etot - 0.5*momx**2/dens
        return eint*(self.gamma -1.)
    def csnd(self, v):
        p = self.pres(v)
        dens = v[...,0]
        return np.sqrt(self.gamma*p/dens)
    def __call__(self, v: np.ndarray) -> np.ndarray:
        # define p and E
        p = self.pres(v)
        res = np.ndarray(v.shape)
        res[...,0] = v[...,1]
        res[...,1] = v[...,1]**2/v[...,0] + p
        res[...,2] = (v[...,2] + p)*v[...,1]/v[...,0]
        return res

    # TODO
    def derivative(self, v: np.ndarray) -> np.ndarray:
        velx = v[...,1]/v[...,0]
        csnd = self.csnd(v)
        return np.abs(velx) + csnd


if Type == PDE_Type.Linear_advection:
    F = LinearAdvection(10)
elif Type == PDE_Type.Burgers_equation:
    F = BurgersEq()
elif Type == PDE_Type.Euler:
    F = Euler(5./3)
else:
    raise "PDE_Type not known"

L = 10
dx = L / 1000
ncells = int(L / dx)

grid = np.zeros((ncells + 2, F.ncomp))  # pad with zero


# TODO: initial values
def f(x):
    # return np.array([1. + 0.1*np.sin(2 * np.pi / L * x), 1, 1])
    # return np.cos(2 * np.pi / L * x)
    return np.exp(-(x - 3)**2)
    # return np.array(list(map(lambda x: 1 if 1 < x < 2 else 0, x)))


# initial conditions
domain = np.linspace(0, L, ncells + 1)
for i in range(ncells):
    grid[i + 1, :] = f((domain[i] + domain[i + 1])/2) #quadrature(f, domain[i], domain[i + 1])[0] / dx


def pbc(grid: np.ndarray):
    grid[0,:] = grid[-2,:]
    grid[-1, :] = grid[1,:]

pbc(grid)

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
    pbc(grid)
    staggerd_grid = (grid[1:] + grid[:-1]) / 2

    staggerd_grid -= c / 2 * (F(grid[1:]) - F(grid[:-1]))
    grid[1:-1] -= c * (F(staggerd_grid[1:]) - F(staggerd_grid[:-1]))
    # grid[1:-1] -= c * (F(grid[2:]) - F(grid[:-2]))
    traj.append(deepcopy(grid[1:-1, 0]))

# Plotting
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)
frame, = ax.plot([], [])

stride = 10


def step(time):
    frame.set_data(domain[:-1], traj[time * stride])
    ax.set_title("%.4f s" % (time * dt))
    return frame,


ani = FuncAnimation(fig, step, frames=int(nsteps / stride), blit=False)
plt.show()
