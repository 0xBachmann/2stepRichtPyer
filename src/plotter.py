import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import os

from PDE_Types import PDE
from copy import deepcopy
from two_step_richtmeyer_util import Dimension


class Plotter:
    def __init__(self, F: PDE, action, writeout, dim: Dimension, coords: list[np.ndarray, ...],
                 filename=None):
        self.ncomp = F.ncomp
        self.comp_names = F.comp_names
        self.dim = dim
        self.init = False
        # Will be set the first time plotting happens
        self.fig = None
        self.ax = None
        self.axs = None
        self.ims = None

        if self.dim == Dimension.oneD:
            self.x_coords = coords[0]
        elif self.dim == Dimension.twoD:
            pass
        else:
            raise NotImplementedError

        self.traj: list[tuple[np.ndarray, float], ...] = []

        assert action in ["show", "save"]
        self.action = action
        self.step = 0
        self.writeout = writeout
        self.PDE_type = f"{F.Type}"
        self.filename = filename if filename is not None else f"{self.PDE_type}.mp4"
        self.time = 0.

    def init_plots(self, vals: np.ndarray, time):
        if self.init:
            return

        self.init = True
        self.fig, self.ax = plt.subplots(self.ncomp)
        self.axs = [self.ax] if self.ncomp == 1 else [ax for ax in self.ax]

        if self.dim == Dimension.oneD:
            self.ims = [self.axs[i].plot(self.x_coords, vals[..., i])[0] for i in range(self.ncomp)]
        if self.dim == Dimension.twoD:
            self.ims = [self.axs[i].imshow(vals[..., i]) for i in range(self.ncomp)]
            for i in range(self.ncomp):
                self.fig.colorbar(self.ims[i])

        for i in range(self.ncomp):
            self.axs[i].set_title(f"{self.comp_names[i]}\ntime: {time:.5}")

    def plot(self, vals: np.ndarray, time):
        if not self.init:
            self.init_plots(vals, time)
        else:
            for i in range(self.ncomp):
                self.axs[i].set_title(f"{self.comp_names[i]}\ntime: {time:.5}")
                if self.dim == Dimension.oneD:
                    self.ims[i].set_data(self.x_coords, vals[..., i])
                if self.dim == Dimension.twoD:
                    self.ims[i].set_data(vals[..., i])

    def write(self, vals: np.ndarray, dt):
        if self.step % self.writeout == 0:
            if self.action == "show":
                self.traj.append((deepcopy(vals), self.time))
            elif self.action == "save":
                self.plot(vals, self.time)
                plt.savefig(f"movie/{self.PDE_type}_{int(self.step / self.writeout)}.png")

        self.time += dt
        self.step += 1

    def finlaize(self):
        if self.action == "show":
            self.init_plots(*self.traj[0])
            ani = FuncAnimation(self.fig, lambda s: self.plot(*self.traj[s]), frames=int(self.step / self.writeout),
                                blit=False)
            plt.show()
        elif self.action == "save":
            os.system(f"ffmpeg -framerate 30 -i 'movie/{self.PDE_type}_%d.png' movie/{self.filename}")
            os.system(f"rm movie/{self.PDE_type}_*.png")
