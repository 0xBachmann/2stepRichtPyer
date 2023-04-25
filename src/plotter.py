import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import os

from PDE_Types import PDE
from copy import deepcopy
from two_step_richtmeyer_util import Dimension


class Plotter:
    def __init__(self, F: PDE | int, action, writeout, dim: Dimension, coords=None,
                 filename=None, lims: dict = {}):
        self.ncomp = F.ncomp if isinstance(F, PDE) else F
        self.comp_names = F.comp_names if isinstance(F, PDE) else None
        self.dim = dim
        self.init = False
        # Will be set the first time plotting happens
        self.fig = None
        self.ax = None
        self.axs = None
        self.ims = None

        if self.dim == Dimension.oneD:
            self.x_coords = coords[0]
            self.lims = lims
            for i in range(self.ncomp):
                if i not in self.lims:
                    self.lims[i] = []
        elif self.dim == Dimension.twoD:
            self.lims = dict()
            for i in range(self.ncomp):
                if i in lims:
                    self.lims[i] = {"vmin": lims[i][0], "vmax": lims[i][1]}
                else:
                    self.lims[i] = {}
            pass
        else:
            raise NotImplementedError

        self.traj: list[tuple[np.ndarray, float], ...] = []

        assert action in ["show", "save"]
        self.action = action
        self.step = 0
        self.writeout = writeout
        self.PDE_type = f"{F.Type}" if isinstance(F, PDE) else ""
        self.filename = filename if filename is not None else f"{self.PDE_type}_{self.dim.value}D.mp4"
        self.time = 0.

    def init_plots(self, vals: np.ndarray, time):
        if self.init:
            return

        self.init = True
        layout = {
            1: (1,),
            2: (2,),
            3: (3,),
            4: (2, 2),
            5: (2, 3),
        }
        self.fig, self.ax = plt.subplots(*layout[self.ncomp])
        self.axs = [self.ax] if self.ncomp == 1 else self.ax.flatten()

        if self.dim == Dimension.oneD:
            self.ims = [self.axs[i].plot(self.x_coords, vals[..., i])[0] for i in range(self.ncomp)]
            for i in range(self.ncomp):
                self.axs[i].set_ylim(*self.lims[i])
        if self.dim == Dimension.twoD:
            self.ims = [self.axs[i].imshow(vals[..., i].T, origin="lower", **self.lims[i]) for i in range(self.ncomp)]
            self.cbar = [self.fig.colorbar(self.ims[i]) for i in range(self.ncomp)]

        if self.comp_names is not None:
            for i in range(self.ncomp):
                self.axs[i].set_title(f"{self.comp_names[i]}")
        self.fig.suptitle(f"time: {time:.5}")

    def plot(self, vals: np.ndarray, time):
        if not self.init:
            self.init_plots(vals, time)
        else:
            for i in range(self.ncomp):
                if self.comp_names is not None:
                    self.axs[i].set_title(f"{self.comp_names[i]}")
                if self.dim == Dimension.oneD:
                    self.ims[i].set_data(self.x_coords, vals[..., i])
                if self.dim == Dimension.twoD:
                    self.ims[i].set_data(vals[..., i].T)
            self.fig.suptitle(f"time: {time:.5}")

    def write(self, vals: np.ndarray, dt):
        if self.step % self.writeout == 0:
            if self.action == "show":
                self.traj.append((deepcopy(vals), self.time))
            elif self.action == "save":
                self.plot(vals, self.time)
                plt.tight_layout()
                plt.savefig(f"movie/{self.PDE_type}_{int(self.step / self.writeout)}.png")

        self.time += dt
        self.step += 1

    def finalize(self):
        if self.action == "show":
            self.init_plots(*self.traj[0])
            ani = FuncAnimation(self.fig, lambda s: self.plot(*self.traj[s]), frames=int(self.step / self.writeout),
                                blit=False)
            plt.show()
        elif self.action == "save":
            os.system(f"ffmpeg -framerate 30 -i 'movie/{self.PDE_type}_%d.png' movie/{self.filename}")
            os.system(f"rm movie/{self.PDE_type}_*.png")
