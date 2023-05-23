from two_step_richtmeyer_util import *
from PDE_Types import PDE, Euler
import numpy as np
from copy import deepcopy
import sys
from scipy.optimize import root
from scipy.linalg import block_diag
from scipy import sparse

from typing import Callable, Union


class Solver:
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray, bdc: Union[str, Callable] = "periodic"):
        self.pde = pde
        self.dim = pde.dim
        if len(domain.shape) == 2:
            self.domain = domain
        else:
            self.domain = np.zeros((domain.shape[0], 2))
            self.domain[:, 1] = domain
        self.ncellsxyz = resolutions
        self.coords = [np.linspace(self.domain[i, 0], self.domain[i, 1], self.ncellsxyz[i] + 1) for i in
                       range(self.dim.value)]
        self.dxyz = ((self.domain[:, 1] - self.domain[:, 0]) / self.ncellsxyz).ravel()
        self.grid = np.empty((*(self.ncellsxyz + 2), self.pde.ncomp))
        self.no_ghost = tuple(slice(1, -1) for _ in range(self.dim.value))

        self.is_periodic = False
        if isinstance(bdc, str):
            if bdc == "periodic":
                self.is_periodic = True
                self.bdc = lambda grid: pbc(grid, self.dim)
            elif bdc == "zero":
                self.bdc = lambda grid: zero_bd(grid, self.dim)
            else:
                raise ValueError(f"{bdc} boundary condition type not known")
        elif callable(bdc):
            self.bdc = bdc
        else:
            raise TypeError(f"Expected string or Callable, got {type(bdc)} instead")

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
        prime = self.pde.max_speed(self.grid[self.no_ghost])
        a = np.max(np.reshape(np.abs(prime), (np.product(self.ncellsxyz), self.dim.value)), axis=0)
        dts = self.dxyz / (2 * a)  # TODO (2 bc half step)
        # TODO correct for diagonal?
        return np.min(dts)

    def get_coords(self, stacked=False):
        avg_coords = [avg_x(coord) for coord in self.coords]
        XYZ = np.meshgrid(*avg_coords, indexing='ij')
        if stacked:
            XYZ = np.stack(XYZ, axis=-1)
        return XYZ

    def step(self, dt):
        raise RuntimeError(f"{self.__class__} is only an abstract base class")

    def step_for(self, T, fact=1., callback=None, log_step=True):
        time = 0.
        if callable(callback):
            callback(self, 0)
        while time < T:
            dt = self.cfl() * fact
            self.step(dt)

            if callable(callback):
                callback(self, dt)

            time += min(dt, T - time)
            if log_step:
                print(f"dt = {dt}, time = {time:.3f}/{T}")


class Richtmeyer2step(Solver):
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray, bdc: Union[str, Callable] = "periodic",
                 lerp=False):
        super().__init__(pde, domain, resolutions, bdc)
        self.lerp = lerp

    def step(self, dt):
        assert isinstance(self.pde, Euler)

        def div_fluxes(fluxes: tuple[np.ndarray]) -> np.ndarray:
            if self.dim == Dimension.oneD:
                return c[0] * del_x(fluxes[0])
            elif self.dim == Dimension.twoD:
                return c[0] * del_x(avg_y(fluxes[0])) \
                    + c[1] * del_y(avg_x(fluxes[1]))
            else:
                return c[0] * del_x(avg_y(avg_z(fluxes[0]))) \
                    + c[1] * del_y(avg_x(avg_z(fluxes[1]))) \
                    + c[2] * del_z(avg_x(avg_y(fluxes[2])))

        def order1() -> tuple[np.ndarray, np.ndarray]:
            # A = self.pde.jacobian(avg_x(avg_y(v)))  # (self.pde(avg_x(staggered)), self.pde(avg_y(staggered)))  # TODO
            # flux_x = self.pde(avg_y(v))[0]
            # flux_y = self.pde(avg_x(v))[1]
            # return (avg_x(flux_x) - 0.5 * np.einsum("...ij,...j->...i", A[0], del_x(avg_y(v))),
            #         avg_y(flux_y) - 0.5 * np.einsum("...ij,...j->...i", A[1], del_y(avg_x(v))))
            fluxes = self.pde(self.grid)
            stagg = avg_x(avg_y(self.grid)) + div_fluxes(fluxes)
            fluxes = self.pde(stagg)
            return avg_y(avg_x(stagg)) + div_fluxes(fluxes)

        c = dt / self.dxyz
        self.bdc(self.grid)

        staggered = avg_x(avg_y(self.grid))
        staggered -= 0.5 * div_fluxes(self.pde(self.grid, False))  # no viscosity for predictor

        if self.lerp:
            # eta = self.pde.eta(staggered, self.dxyz[0], self.dxyz[1])[..., np.newaxis].astype(float)
            eta = self.pde.eta_(self.grid, self.dxyz[0], self.dxyz[1])
            # eta = np.minimum(eta, 1)
            # TODO or replace order1 by viscosity
            # self.grid_no_ghost = (1. - eta) * (self.grid_no_ghost - div_fluxes(self.pde(staggered))) + eta * order1()
            self.grid_no_ghost -= (1. - eta) * div_fluxes(self.pde(staggered, False)) + eta * div_fluxes(self.pde(staggered, True, other=self.grid_no_ghost))
        else:
            self.grid_no_ghost -= div_fluxes(self.pde(staggered, other=self.grid_no_ghost))


class Richtmeyer2stepImplicit(Solver):
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray, bdc: Union[str, Callable] = "periodic",
                 eps=sys.float_info.epsilon, method="krylov", manual_jacobian=False, use_sparse=False):
        super().__init__(pde, domain, resolutions, bdc)
        self.eps = eps
        root_methods = ["hybr", "lm", "broyden1", "broyden2", "anderson", "linearmixing", "diagbroyden",
                        "excitingmixing", "krylov", "df-sane"]
        assert method in root_methods + ["newton"]
        if manual_jacobian:
            raise NotImplementedError("Jacobian for new implicit scheme not implemented yet")
        self.use_root = method in root_methods
        self.manual_jacobian = True if method not in root_methods else manual_jacobian
        self.method = method
        self.use_sparse = use_sparse
        self.nfevs = []

    def step(self, dt, guess=None):
        c = dt / self.dxyz

        grid_old = deepcopy(self.grid)

        if self.dim == Dimension.oneD:
            def FJ(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                self.grid_no_ghost = v.reshape(self.grid_no_ghost.shape)
                self.bdc(self.grid)
                avg_t = 0.5 * (self.grid + grid_old)
                fluxes = self.pde(avg_t)
                F = (self.grid_no_ghost - grid_old[self.no_ghost]
                     + c[0] * self.del_x(fluxes[0])).ravel()

                JF, = self.pde.jacobian(avg_t[self.no_ghost])

                nx = self.ncellsxyz[0]
                ncomp = self.pde.ncomp

                JF = block_diag(*JF.reshape((nx, ncomp, ncomp)))
                # JF = block_diag(*[np.full((4, 4), i+1) for i in range(ncells)])
                # JG = block_diag(*[np.full((4, 4), i+101) for i in range(ncells)])

                J = np.eye(nx * ncomp, nx * ncomp)

                # dx JF
                J += np.roll(JF, shift=-nx * ncomp, axis=0) * c[0] / 2
                J -= np.roll(JF, shift=nx * ncomp, axis=0) * c[0] / 2

                # J[nx * ncomp:(nx + 1) * ncomp, (nx - 1) * ncomp:nx * ncomp, ...] = 0
                # J[(nx - 1) * ncomp:nx * ncomp, nx * ncomp:(nx + 1) * ncomp, ...] = 0
                if self.is_periodic:
                    J[(nx - 1) * ncomp:nx * ncomp, :ncomp, ...] = \
                        JF[:ncomp, :ncomp, ...]
                    J[:ncomp, (nx - 1) * ncomp:nx * ncomp, ...] = \
                        -JF[(nx - 1) * ncomp:nx * ncomp, (nx - 1) * ncomp:nx * ncomp, ...]
                return F, J

            def F(v: np.ndarray) -> np.ndarray:
                self.grid_no_ghost = v.reshape(self.grid_no_ghost.shape)
                self.bdc(self.grid)
                avg_t = 0.5 * (self.grid + grid_old)
                fluxes = self.pde(avg_x(avg_t))
                return (self.grid_no_ghost - grid_old[self.no_ghost] + c[0] * del_x(fluxes[0])).ravel()

            def J(v: np.ndarray) -> np.ndarray:
                self.grid_no_ghost = v.reshape(self.grid_no_ghost.shape)
                self.bdc(self.grid)
                avg_t = 0.5 * (self.grid + grid_old)
                jacobians = self.pde.jacobian(avg_t)
                jacobian = (np.diag(jacobians[0][:-3].ravel(), k=-1) - np.diag(jacobians[0][3:].ravel(), k=1)) / 2
                jacobian[0, -1] = jacobians[0][0] / 2
                jacobian[-1, 0] = -jacobians[0][-1] / 2

                raise NotImplementedError

        elif self.dim == Dimension.twoD:
            def FJ(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                self.grid_no_ghost = v.reshape(self.grid_no_ghost.shape)
                self.bdc(self.grid)
                avg_t = 0.5 * (self.grid + grid_old)
                fluxes = self.pde(avg_t)
                F = (self.grid_no_ghost - grid_old[self.no_ghost]
                     + c[0] * self.del_x(self.avg_y(fluxes[0]))
                     + c[1] * self.del_y(self.avg_x(fluxes[1]))).ravel()

                JF, JG = self.pde.jacobian(avg_t[self.no_ghost])

                ncells = np.product(self.ncellsxyz)
                ncomp = self.pde.ncomp

                JF = block_diag(*JF.reshape((ncells, ncomp, ncomp)))
                JG = block_diag(*JG.reshape((ncells, ncomp, ncomp)))
                # JF = block_diag(*[np.full((4, 4), i+1) for i in range(ncells)])
                # JG = block_diag(*[np.full((4, 4), i+101) for i in range(ncells)])

                J = np.eye(ncells * ncomp, ncells * ncomp)
                # dy JG
                J[:-ncomp, ...] += JG[ncomp:, ...] * c[1] / 2
                J[ncomp:, ...] -= JG[:-ncomp, ...] * c[1] / 2

                nx = self.ncellsxyz[0]
                ny = self.ncellsxyz[1]

                # dx JF
                J += np.roll(JF, shift=-nx * ncomp, axis=0) * c[0] / 2
                J -= np.roll(JF, shift=nx * ncomp, axis=0) * c[0] / 2

                for i in range(1, ny):
                    J[i * nx * ncomp:(i * nx + 1) * ncomp, (i * nx - 1) * ncomp:i * nx * ncomp, ...] = 0
                    J[(i * nx - 1) * ncomp:i * nx * ncomp, i * nx * ncomp:(i * nx + 1) * ncomp, ...] = 0
                if self.is_periodic:
                    for i in range(ny):
                        J[((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp, i * nx * ncomp:(i * nx + 1) * ncomp, ...] = \
                            JF[i * nx * ncomp:(i * nx + 1) * ncomp, i * nx * ncomp:(i * nx + 1) * ncomp, ...]
                        J[i * nx * ncomp:(i * nx + 1) * ncomp, ((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp, ...] = \
                            -JF[((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp,
                             ((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp, ...]
                if not self.is_periodic:
                    shift = (ny - 1) * nx * ncomp
                    for i in range(nx):
                        J[shift + i * ncomp:shift + (i + 1) * ncomp, i * ncomp:(i + 1) * ncomp, ...] = 0
                        J[i * ncomp:(i + 1) * ncomp, shift + i * ncomp:shift + (i + 1) * ncomp, ...] = 0
                return F, J

            def F(v: np.ndarray) -> np.ndarray:
                self.grid_no_ghost = v.reshape(self.grid_no_ghost.shape)
                self.bdc(self.grid)
                avg_t = 0.5 * (self.grid + grid_old)
                fluxes = self.pde(avg_x(avg_y(avg_t)))
                return (self.grid_no_ghost - grid_old[self.no_ghost] + c[0] * del_x(avg_y(fluxes[0]))
                        + c[1] * del_y(avg_x(fluxes[1]))).ravel()

            def J(v: np.ndarray) -> np.ndarray:
                grid = np.empty(self.grid.shape)
                grid[self.no_ghost] = v.reshape(self.grid_no_ghost.shape)
                self.bdc(grid)
                avg_t = 0.5 * (grid + grid_old)

                JF, JG = self.pde.jacobian(avg_t[self.no_ghost])

                ncells = np.product(self.ncellsxyz)
                ncomp = self.pde.ncomp

                JF = block_diag(*JF.reshape((ncells, ncomp, ncomp)))
                JG = block_diag(*JG.reshape((ncells, ncomp, ncomp)))

                J = np.eye(ncells * ncomp, ncells * ncomp)
                # dy JG
                J[:-ncomp, ...] += JG[ncomp:, ...] * c[0] / 2
                J[ncomp:, ...] -= JG[:-ncomp, ...] * c[0] / 2

                nx = self.ncellsxyz[0]
                ny = self.ncellsxyz[1]

                # dx JF
                J += np.roll(JF, shift=-nx * ncomp, axis=0) * c[1] / 2
                J -= np.roll(JF, shift=nx * ncomp, axis=0) * c[1] / 2
                for i in range(1, ny):
                    J[i * nx * ncomp:(i * nx + 1) * ncomp, (i * nx - 1) * ncomp:i * nx * ncomp, ...] = 0
                    J[(i * nx - 1) * ncomp:i * nx * ncomp, i * nx * ncomp:(i * nx + 1) * ncomp, ...] = 0
                if self.is_periodic:
                    for i in range(ny):
                        J[((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp, i * nx * ncomp:(i * nx + 1) * ncomp, ...] = \
                            JF[i * nx * ncomp:(i * nx + 1) * ncomp, i * nx * ncomp:(i * nx + 1) * ncomp, ...]
                        J[i * nx * ncomp:(i * nx + 1) * ncomp, ((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp, ...] = \
                            -JF[((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp,
                             ((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp, ...]
                if not self.is_periodic:
                    shift = (ny - 1) * nx * ncomp
                    for i in range(nx):
                        J[shift + i * ncomp:shift + (i + 1) * ncomp, i * ncomp:(i + 1) * ncomp, ...] = 0
                        J[i * ncomp:(i + 1) * ncomp, shift + i * ncomp:shift + (i + 1) * ncomp, ...] = 0
                return J

        else:
            raise NotImplementedError("Jacobians not implemented for 3D")

        if self.use_root:
            if guess is None:
                guess = self.grid_no_ghost
            sol = root(F, guess.ravel(), tol=self.eps * np.product(self.ncellsxyz),
                       jac=J if self.manual_jacobian else False, method=self.method)
            # self.nfevs.append(sol.nfev)
            self.grid_no_ghost = sol.x.reshape(self.grid_no_ghost.shape)
            self.bdc(self.grid)
        else:
            F_value, J_value = FJ(self.grid_no_ghost.ravel())
            F_norm = np.linalg.norm(F_value)
            while self.eps * np.product(self.ncellsxyz) < F_norm:
                if self.use_sparse:
                    J_value = sparse.csr_matrix(J_value)  # , blocksize=(self.pde.ncomp, self.pde.ncomp))
                    self.grid_no_ghost -= sparse.linalg.spsolve(J_value, F_value).reshape(self.grid_no_ghost.shape)
                else:
                    self.grid_no_ghost -= np.linalg.solve(J_value, F_value).reshape(self.grid_no_ghost.shape)

                self.bdc(self.grid)
                F_value, J_value = FJ(self.grid_no_ghost.ravel())
                F_norm = np.linalg.norm(F_value)

