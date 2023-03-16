import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from enum import Enum
from tqdm import tqdm
import subprocess
from copy import deepcopy
from two_step_richtmeyer_util import *
from PDE_Types import *




class plotting_Type(Enum):
    show = 1
    save = 2


Plotting = plotting_Type.save
Type = PDE_Type.Linear_advection
DIM = Dimension.twoD


# TODO: there is flux F and flux G -> solvers d/dt + d/dx F + d/dy G = 0




if Type == PDE_Type.Linear_advection:
    F = LinearAdvection(np.array([10, 10]), dim=DIM)
elif Type == PDE_Type.Burgers_equation:
    F = BurgersEq(dim=DIM)
elif Type == PDE_Type.Euler:
    F = Euler(5. / 3, dim=DIM)
else:
    raise "PDE_Type not known"

fig, ax = plt.subplots(F.ncomp, subplot_kw={"projection": "3d"})
axs = [ax] if F.ncomp == 1 else ax

L = 10
dx = L / 1000
dy = L / 1000
ncellsx = int(L / dx)
ncellsy = int(L / dy)

grid = np.zeros((ncellsx + 2, ncellsy + 2, F.ncomp))  # pad with zero


# TODO: initial values
def f(x):
    # return np.array([1. + 0.1*np.sin(2 * np.pi / L * x), 1, 1])
    # return np.cos(2 * np.pi / L * x)
    return np.exp(-np.linalg.norm(x - np.array([3, 3]))**2)
    # return np.array(list(map(lambda x: 1 if 1 < x < 2 else 0, x)))


# initial conditions
coords_x = np.linspace(0, L, ncellsx + 1)
coords_y = np.linspace(0, L, ncellsy + 1)
X, Y = np.meshgrid(coords_x[:-1], coords_y[:-1])
for i, j in itertools.product(range(ncellsx), range(ncellsy)):
    grid[i + 1, j + 1, :] = f((
                                      np.array([coords_x[i], coords_y[j]])
                                      + np.array([coords_x[i + 1], coords_y[j]])
                                      + np.array([coords_x[i], coords_y[j + 1]])
                                      + np.array([coords_x[i + 1], coords_y[j + 1]])
                              ) / 4)


def Plot(vals: np.ndarray, s, ncomps):
    frames = []
    for i in range(ncomps):
        axs[i].clear()
        axs[i].set_title(f"step {s}")
        frame = axs[i].plot_surface(X, Y, vals[..., i], cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
        frames.append(frame)
    return frames


pbc(grid, dim=DIM)

T = 1

traj: list[np.ndarray] = []
time = 0
steps = 0
writeout = 10
while time < T:
    Fprime = F.derivative(grid)
    ax = np.max(np.abs(Fprime[..., 0]))
    ay = np.max(np.abs(Fprime[..., 1]))
    dt = dx / (2 * max(ax, ay))  # damp?
    time += dt
    print(f"step: {steps}, dt = {dt}, time = {time:.3f}/{T}")

    c = dt / dx

    # TODO: Boundary conditions?
    pbc(grid, dim=DIM)

    staggerd_grid = avg_x(avg_y(grid))

    Fgrid, Ggrid = F(grid)

    staggerd_grid -= c / 2 * (del_x(avg_y(Fgrid)) + del_y(avg_x(Ggrid)))

    Fstaggered_grid, Gstaggered_grid = F(staggerd_grid)

    grid[1:-1, 1:-1, :] -= c * (del_x(avg_y(Fstaggered_grid)) + del_y(avg_x(Gstaggered_grid)))

    if steps % writeout == 0:
        if Plotting == plotting_Type.show:
            traj.append(deepcopy(grid[1:-1, 1:-1, :]))
        elif Plotting == plotting_Type.save:
            Plot(grid[1:-1, 1:-1, :], steps, F.ncomp)
            plt.savefig(f"movie/{Type}_{steps}.png")
    steps += 1


def step(s):
    Plot(traj[s], s, F.ncomp)


if Plotting == plotting_Type.show:
    ani = FuncAnimation(fig, step, frames=int(steps / writeout), blit=False)
    plt.show()
elif Plotting == plotting_Type.save:
    # TODO wtf
    # subprocess.run(["ffmpeg", "-framerate", "30",  "-pattern_type", "glob", "-i", f"'~/bachelor-thesis/src/movie/{Type}_*.png'", f"movie/{Type}.mp4"])
    # subprocess.run(["rm", f"/home/jonas/bachelor-thesis/src/movie/{Type}_*.png"])
    pass
