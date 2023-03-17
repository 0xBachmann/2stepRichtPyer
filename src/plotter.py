import matplotlib.pyplot as plt
from PDE_Types import PDE
import numpy as np
from enum import Enum
from copy import deepcopy
from matplotlib.animation import FuncAnimation
import subprocess
from two_step_richtmeyer_util import Dimension


class plotting_Type(Enum):
    show = 1
    save = 2


class Plotter:
    def __init__(self, F: PDE, vals: np.ndarray, action, writeout, dim: Dimension, coords: list[np.ndarray, ...],
                 filename=None):
        self.ncomps = F.ncomp
        self.comp_names = F.comp_names
        self.dim = dim
        self.fig, self.ax = plt.subplots(F.ncomp)
        self.axs = [self.ax] if F.ncomp == 1 else self.ax

        if self.dim >= Dimension.oneD:
            self.x_coords = coords[0]
        if self.dim >= Dimension.twoD:
            self.y_coords = coords[1]

        if self.dim == Dimension.oneD:
            self.x_coords = coords[0]
            self.ims = [self.axs[i].plot(self.x_coords, vals[..., i])[0] for i in range(self.ncomps)]
        elif self.dim == Dimension.twoD:
            self.ims = [self.axs[i].imshow(vals[..., i]) for i in range(self.ncomps)]
        else:
            raise NotImplementedError

        self.traj: list[tuple[np.ndarray, float], ...] = []

        if self.dim >= Dimension.twoD:
            for i in range(self.ncomps):
                self.fig.colorbar(self.ims[i])

        assert action in ["show", "save"]
        self.action = action
        self.step = 0
        self.writeout = writeout
        self.PDE_type = f"{F.Type}"
        self.filename = filename if filename is not None else "{self.PDE_type}.mp4"
        self.time = 0

    def plot(self, vals: np.ndarray, time):
        for i in range(self.ncomps):
            self.axs[i].set_title(f"{self.comp_names[i]}\ntime: {time:.5}")
            if self.dim == Dimension.oneD:
                self.ims[i].set_data(self.x_coords, vals[..., i])
            if self.dim == Dimension.twoD:
                self.ims[i].set_data(vals[..., i])

    def write(self, vals: np.ndarray, dt):
        self.time += dt
        if self.step % self.writeout == 0:
            if self.action == "show":
                self.traj.append((deepcopy(vals), self.time))
            elif self.action == "save":
                self.plot(vals, self.time)
                plt.savefig(f"movie/{self.PDE_type}_{int(self.step / self.writeout)}.png")

        self.step += 1


    def finlaize(self):
        if self.action == "show":
            ani = FuncAnimation(self.fig, lambda s: self.plot(*self.traj[s]), frames=int(self.step / self.writeout),
                                blit=False)
            plt.show()
        elif self.action == "save":
            subprocess.run(
                ["ffmpeg", "-framerate", "30", "-i", f"'.'movie/{self.PDE_type}_%d.png'", f"movie/{self.filename}"])
            subprocess.run(["rm", f"/home/jonas/bachelor-thesis/src/movie/{self.PDE_type}_*.png"])
