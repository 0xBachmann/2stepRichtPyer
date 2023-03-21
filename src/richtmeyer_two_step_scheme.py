import numpy as np

from two_step_richtmeyer_util import *
from PDE_Types import PDE
import itertools


class Richtmeyer2step:
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray):
        self.pde = pde
        self.dim = pde.dim
        self.domain = domain
        self.ncellsxyz = resolutions
        self.coords = [np.linspace(0, self.domain[i], self.ncellsxyz[i] + 1) for i in range(self.dim.value)]
        self.dxyz = self.domain / self.ncellsxyz
        self.grid = np.empty((*(self.ncellsxyz + 2), self.pde.ncomp))
        self.no_ghost = tuple(slice(1, -1) for _ in range(self.dim.value))

    @property
    def grid_no_ghost(self):
        return self.grid[self.no_ghost]

    @grid_no_ghost.setter
    def grid_no_ghost(self, v: np.ndarray):
        self.grid[self.no_ghost] = v

    def initial_cond(self, f):
        # TODO make more efficient
        for indices in itertools.product(*[range(n) for n in self.ncellsxyz]):
            self.grid[self.no_ghost][indices] = f(
                np.array([self.coords[i][j] + self.coords[i][j] for i, j in zip(range(self.dim.value), indices)]) / 2)

    def step(self, dt):
        def div_fluxes(source: np.ndarray):
            source_fluxes = self.pde(source)
            if self.dim == Dimension.oneD:
                return c[0] * del_x(source_fluxes[0])
            if self.dim == Dimension.twoD:
                return c[0] * del_x(avg_y(source_fluxes[0])) + c[1] * del_y(avg_x(source_fluxes[1]))
            if self.dim == Dimension.threeD:
                return c[0] * del_x(avg_y(avg_z(source_fluxes[0]))) + c[1] * del_y(avg_x(avg_z(source_fluxes[1]))) \
                        + c[2] * del_z(avg_x(avg_y(source_fluxes[2])))

        # TODO correct in case of D > 1?
        # TODO INCORRECT!
        c = dt / self.dxyz
        # TODO other bd cond?
        pbc(self.grid, self.dim)

        staggered = avg_x(self.grid)
        if self.dim == Dimension.twoD:
            staggered = avg_y(staggered)
        if self.dim == Dimension.threeD:
            staggered = avg_z(staggered)

        staggered -= 0.5 * div_fluxes(self.grid)
        self.grid_no_ghost -= div_fluxes(staggered)

    def cfl(self):
        prime = self.pde.derivative(self.grid[self.no_ghost])
        a = np.max(np.reshape(np.abs(prime), (np.product(self.ncellsxyz), self.dim.value)), axis=0)
        dts = self.dxyz / (2 * a)
        return np.max(dts)
