from two_step_richtmeyer_util import *
from PDE_Types import PDE
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
        self.coords = [np.linspace(self.domain[i, 0], self.domain[i, 1], self.ncellsxyz[i] + 1) for i in range(self.dim.value)]
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
            raise RuntimeError(f"Expected string or Callable, got {type(bdc)} instead")

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


class Richtmeyer2step(Solver):
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray, bdc: Union[str, Callable] = "periodic"):
        super().__init__(pde, domain, resolutions, bdc)

    def step(self, dt):
        def div_fluxes(source: np.ndarray) -> np.ndarray:
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
    def __init__(self, pde: PDE, domain: np.ndarray, resolutions: np.ndarray, bdc: Union[str, Callable] = "periodic", eps=sys.float_info.epsilon,
                 method="krylov", manual_jacobian=False, use_sparse=False):
        super().__init__(pde, domain, resolutions, bdc)
        self.eps = eps
        root_methods = ["hybr", "lm", "broyden1", "broyden2", "anderson", "linearmixing", "diagbroyden",
                        "excitingmixing", "krylov", "df-sane"]
        assert method in root_methods + ["newton"]
        self.use_root = method in root_methods
        self.manual_jacobian = True if method not in root_methods else manual_jacobian
        self.method = method
        self.use_sparse = use_sparse
        self.nfevs = []

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
                fluxes = self.pde(avg_t)
                return (self.grid_no_ghost - grid_old[self.no_ghost] + c[0] * self.del_x(fluxes[0])).ravel()

            def J(v: np.ndarray) -> np.ndarray:
                self.grid_no_ghost = v.reshape(self.grid_no_ghost.shape)
                self.bdc(self.grid)
                avg_t = 0.5 * (self.grid + grid_old)
                jacobians = self.pde.jacobian(avg_t)
                # TODO: grad(del_x(...)) != del_x(grad(...))?
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
                            -JF[((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp, ((i + 1) * nx - 1) * ncomp:(i + 1) * nx * ncomp, ...]
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
                fluxes = self.pde(avg_t)
                return (self.grid_no_ghost - grid_old[self.no_ghost] + c[0] * self.del_x(self.avg_y(fluxes[0]))
                        + c[1] * self.del_y(self.avg_x(fluxes[1]))).ravel()

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
            sol = root(F, self.grid_no_ghost.ravel(), tol=self.eps * np.product(self.ncellsxyz),
                       jac=J if self.manual_jacobian else False, method=self.method)
            # self.nfevs.append(sol.nfev)
            self.grid_no_ghost = sol.x.reshape(self.grid_no_ghost.shape)
            self.bdc(self.grid)
        else:
            F_value, J_value = FJ(self.grid_no_ghost.ravel())
            F_norm = np.linalg.norm(F_value)
            while self.eps * np.product(self.ncellsxyz) < F_norm:
                # for _ in range(2):
                #     for index in it.product(*[range(n) for n in self.ncellsxyz]):
                #         self.grid_no_ghost[index] -= np.linalg.solve(J_value[index], F_value[index])
                if self.use_sparse:
                    J_value = sparse.csr_matrix(J_value)  # , blocksize=(self.pde.ncomp, self.pde.ncomp))
                    self.grid_no_ghost -= sparse.linalg.spsolve(J_value, F_value).reshape(self.grid_no_ghost.shape)
                else:
                    self.grid_no_ghost -= np.linalg.solve(J_value, F_value).reshape(self.grid_no_ghost.shape)

                self.bdc(self.grid)
                F_value, J_value = FJ(self.grid_no_ghost.ravel())
                F_norm = np.linalg.norm(F_value)
