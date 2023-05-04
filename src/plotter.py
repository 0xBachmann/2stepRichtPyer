import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

import numpy as np
import os
from pathlib import Path

from PDE_Types import PDE
from copy import deepcopy
from two_step_richtmeyer_util import Dimension

import uuid


class Plotter:
    def __init__(self, F: PDE | int, action, writeout, dim: Dimension, coords=None,
                 filename=None, lims=None, savedir="movie"):
        self.ncomp = F.ncomp if isinstance(F, PDE) else F
        self.comp_names = F.comp_names if isinstance(F, PDE) else None
        self.dim = dim
        self.savedir = savedir

        self.uuid = uuid.uuid4()

        self.init = False
        # Will be set the first time plotting happens
        self.fig = None
        self.ax = None
        self.axs = None
        self.ims = None

        if lims is None:
            lims = {}

        if self.dim == Dimension.oneD:
            self.x_coords = coords[0]
            self.lims = lims
            for i in range(self.ncomp):
                if i not in self.lims:
                    self.lims[i] = []
        elif self.dim == Dimension.twoD:
            self.lims = {}
            for i in range(self.ncomp):
                if i in lims:
                    self.lims[i] = {"vmin": lims[i][0], "vmax": lims[i][1]}
                else:
                    self.lims[i] = {}
            pass
        else:
            raise NotImplementedError

        self.traj: list[tuple[np.ndarray, float] | float, ...] = []

        assert action in ["show", "save", "saveb"]
        self.action = action
        if self.action == "save" or self.action == "saveb":
            print("=====================" + "=" * 36)
            print(f"The Plotters UUID is {self.uuid}")
            print("=====================" + "=" * 36)

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
            1: [{"shape": (1, 1), "loc": (i, 0)} for i in range(1)],
            2: [{"shape": (2, 1), "loc": (i, 0)} for i in range(2)],
            3: [{"shape": (3, 1), "loc": (i, 0)} for i in range(3)],
            4: [{"shape": (2, 2), "loc": (i, j)} for i in range(2) for j in range(2)],
            5: [{"shape": (2, 6), "loc": loc, "colspan": 2} for loc in [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3)]],
        }
        # self.fig, self.ax = plt.subplots(*layout[self.ncomp])
        # self.axs = [self.ax] if self.ncomp == 1 else self.ax.flatten()
        self.axs = [plt.subplot2grid(**(layout[self.ncomp][i])) for i in range(self.ncomp)]

        self.fig = plt.gcf()
        self.fig.tight_layout()

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
                plt.savefig(Path(self.savedir, f"{self.PDE_type}_{int(self.step / self.writeout)}_{self.uuid}"))
            elif self.action == "saveb":
                np.save(str(Path(self.savedir, f"{self.PDE_type}_{int(self.step / self.writeout)}_{self.uuid}.npy")), vals)
                self.traj.append(self.time)

        self.time += dt
        self.step += 1

    def finalize(self):
        if self.action == "show":
            self.init_plots(*self.traj[0])
            anim = FuncAnimation(self.fig, lambda s: self.plot(*self.traj[s]), frames=int(self.step / self.writeout),
                                 blit=False)
            plt.show()

        elif self.action == "saveb":
            for i, t in tqdm(enumerate(self.traj), desc="Generating Movie", total=len(self.traj)):
                vals = np.load(str(Path(self.savedir, f"{self.PDE_type}_{i}_{self.uuid}")))
                os.remove(Path(self.savedir, f"{self.PDE_type}_{i}_{self.uuid}.npy"))
                self.plot(vals, t)
                plt.savefig(Path(self.savedir, f"{self.PDE_type}_{i}_{self.uuid}.png"))

        if self.action == "save" or self.action == "saveb":
            frames = Path(self.savedir, f"{self.PDE_type}_%d_{self.uuid}.png")
            output = Path(self.savedir, self.filename)
            os.system(f"ffmpeg -framerate 30 -i '{frames}' {output}")

            for item in os.listdir(self.savedir):
                if item.endswith(f"{self.uuid}.png"):
                    os.remove(os.path.join(self.savedir, item))
