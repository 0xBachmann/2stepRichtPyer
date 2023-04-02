from two_step_richtmeyer_util import *
from PDE_Types import PDE
import numpy as np
from copy import deepcopy
import itertools as it
import sys


class Solver:
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray, bdc=None):
        self.pde = pde
        self.dim = pde.dim
        self.domain = domain
        self.ncellsxyz = resolutions
        self.coords = [np.linspace(0, self.domain[i], self.ncellsxyz[i] + 1) for i in range(self.dim.value)]
        self.dxyz = self.domain / self.ncellsxyz
        self.grid = np.empty((*(self.ncellsxyz + 2), self.pde.ncomp))
        self.no_ghost = tuple(slice(1, -1) for _ in range(self.dim.value))
        self.bdc = bdc if bdc is not None else lambda grid: pbc(grid, self.dim)

    @property
    def grid_no_ghost(self):
        return self.grid[self.no_ghost]

    @grid_no_ghost.setter
    def grid_no_ghost(self, v: np.ndarray):
        self.grid[self.no_ghost] = v

    def initial_cond(self, f):
        avg_coords = [avg_x(coord) for coord in self.coords]
        XYZ = np.stack(np.meshgrid(*avg_coords, indexing='ij'), axis=-1)

        self.grid_no_ghost = f(XYZ)
        self.bdc(self.grid)

    def cfl(self):
        prime = self.pde.derivative(self.grid[self.no_ghost])
        a = np.max(np.reshape(np.abs(prime), (np.product(self.ncellsxyz), self.dim.value)), axis=0)
        dts = self.dxyz / (2 * a)  # TODO (2 bc half step)
        return np.min(dts)


class Richtmeyer2step(Solver):
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray, bdc=None):
        super().__init__(pde, domain, resolutions, bdc)

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

        # TODO correct in case of D > 1? yes should be
        c = dt / self.dxyz
        # TODO other bd cond?
        self.bdc(self.grid)

        staggered = avg_x(self.grid)
        if self.dim == Dimension.twoD:
            staggered = avg_y(staggered)
        if self.dim == Dimension.threeD:
            staggered = avg_z(staggered)

        staggered -= 0.5 * div_fluxes(self.grid)
        self.grid_no_ghost -= div_fluxes(staggered)


class Richtmeyer2stepImplicit(Solver):
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray, bdc=None, eps=sys.float_info.epsilon):
        super().__init__(pde, domain, resolutions, bdc)
        self.eps = eps

    # TODO correct?
    def del_x(self, grid_vals: np.ndarray) -> np.ndarray:
        return (grid_vals[2:, ...] - grid_vals[:-2, ...]) / 2

    def del_y(self, grid_vals: np.ndarray) -> np.ndarray:
        return (grid_vals[:, 2:, ...] - grid_vals[:, :-2, ...]) / 2

    def avg_x(self, grid_vals: np.ndarray) -> np.ndarray:
        return grid_vals[1:-1, ...]
        # return (grid_vals[2:, ...] + grid_vals[1:-1, ...] + grid_vals[:-2, ...]) / 4

    def avg_y(self, grid_vals: np.ndarray) -> np.ndarray:
        return grid_vals[:, 1:-1, ...]
        # return (grid_vals[:, 2:, ...] + grid_vals[:, 1:-1, ...] + grid_vals[:, :-2, ...]) / 4

    def step(self, dt):
        c = dt / self.dxyz

        grid_old = deepcopy(self.grid)

        if self.dim == Dimension.oneD:
            def FJ() -> tuple[np.ndarray, np.ndarray]:
                avg_t = 0.5 * (self.grid + grid_old)
                fluxes = self.pde(avg_t)
                F = self.grid_no_ghost - grid_old[self.no_ghost] + c[0] * self.del_x(fluxes[0])
                jacobians = self.pde.jacobian(avg_t)
                J = np.eye(self.pde.ncomp, self.pde.ncomp) + c[0] / 2 * self.del_x(jacobians[0])
                return F, J

        elif self.dim == Dimension.twoD:
            def FJ() -> tuple[np.ndarray, np.ndarray]:
                avg_t = 0.5 * (self.grid + grid_old)
                fluxes = self.pde(avg_t)
                F = self.grid_no_ghost - grid_old[self.no_ghost] + c[0] * self.del_x(self.avg_y(fluxes[0])) \
                                                                 + c[1] * self.del_y(self.avg_x(fluxes[1]))
                jacobians = self.pde.jacobian(avg_t)
                J = np.eye(self.pde.ncomp, self.pde.ncomp) + c[0] / 2 * self.del_x(self.avg_y(jacobians[0])) \
                                                           + c[1] / 2 * self.del_y(self.avg_x(jacobians[1]))
                return F, J
        else:
            def FJ(grid_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                raise NotImplementedError("not implemented for 3D")

        F_value, J_value = FJ()
        F_norm = np.linalg.norm(F_value)
        while abs(F_norm) > self.eps * np.product(self.ncellsxyz):
            for index in it.product(*[range(n) for n in self.ncellsxyz]):
                self.grid_no_ghost[index] -= np.linalg.solve(J_value[index], F_value[index])
            self.bdc(self.grid)
            F_value, J_value = FJ()
            F_norm = np.linalg.norm(F_value)
