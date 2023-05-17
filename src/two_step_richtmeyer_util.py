import numpy as np
from enum import Enum
from functools import total_ordering


@total_ordering
class Dimension(Enum):
    oneD = 1
    twoD = 2
    threeD = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


def pbc(grid: np.ndarray, dim: Dimension):
    if dim >= Dimension.oneD:
        # x dimension
        grid[0, ...] = grid[-2, ...]
        grid[-1, ...] = grid[1, ...]
    if dim >= Dimension.twoD:
        # y dimension
        grid[:, 0, ...] = grid[:, -2, ...]
        grid[:, -1, ...] = grid[:, 1, ...]
    if dim >= Dimension.threeD:
        # z dimension
        grid[:, :, 0, ...] = grid[:, :, -2, ...]
        grid[:, :, -1, ...] = grid[:, :, 1, ...]


def del_x(grid_vals: np.ndarray) -> np.ndarray:
    return grid_vals[1:, ...] - grid_vals[:-1, ...]


def del_y(grid_vals: np.ndarray) -> np.ndarray:
    return grid_vals[:, 1:, ...] - grid_vals[:, :-1, ...]


def del_z(grid_vals: np.ndarray) -> np.ndarray:
    return grid_vals[:, :, 1:, ...] - grid_vals[:, :, :-1, ...]


def avg_x(grid_vals: np.ndarray) -> np.ndarray:
    return (grid_vals[1:, ...] + grid_vals[:-1, ...]) / 2


def avg_y(grid_vals: np.ndarray) -> np.ndarray:
    return (grid_vals[:, 1:, ...] + grid_vals[:, :-1, ...]) / 2


def avg_z(grid_vals: np.ndarray) -> np.ndarray:
    return (grid_vals[:, :, 1:, ...] + grid_vals[:, :, :-1, ...]) / 2


def log(msg):
    print("=" * 10, f"{msg:^30s}", "=" * 10)


def zero_bd(grid: np.ndarray, dim: Dimension):
    if dim >= Dimension.oneD:
        # x dimension
        grid[0, ...] = 0
        grid[-1, ...] = 0
    if dim >= Dimension.twoD:
        # y dimension
        grid[:, 0, ...] = 0
        grid[:, -1, ...] = 0
    if dim >= Dimension.threeD:
        # z dimension
        grid[:, :, 0, ...] = 0
        grid[:, :, -1, ...] = 0


def dirichlet_bd(grid: np.ndarray, dim: Dimension, XYZ: np.ndarray, f):
    if dim >= Dimension.oneD:
        grid[0, ...] = f(XYZ[0, ...])
        grid[-1, ...] = f(XYZ[-1, ...])
    if dim >= Dimension.twoD:
        grid[:, 0, ...] = f(XYZ[:, 0, ...])
        grid[:, -1, ...] = f(XYZ[:, -1, ...])
    if dim >= Dimension.threeD:
        grid[:, :, 0, ...] = f(XYZ[:, :, 0, ...])
        grid[:, :, -1, ...] = f(XYZ[:, :, -1, ...])


def curl(vels: np.ndarray, dim: Dimension, h) -> np.ndarray:
    if dim == dim.oneD:
        raise RuntimeError("curl is not defined in 1D")

    if dim == dim.twoD:
        vels = avg_x(avg_y(vels))
        return del_x(avg_y(vels[..., 1])) / h[0] - del_y(avg_x(vels[..., 0])) / h[1]

    if dim == dim.threeD:
        vels = avg_x(avg_y(avg_z(vels)))
        res = np.empty_like(vels)
        res[..., 0] = del_y(avg_x(avg_z(vels[..., 2]))) / h[1] - del_z(avg_x(avg_y(vels[..., 1]))) / h[2]
        res[..., 1] = del_z(avg_x(avg_y(vels[..., 0]))) / h[2] - del_x(avg_y(avg_z(vels[..., 2]))) / h[0]
        res[..., 2] = del_x(avg_y(avg_z(vels[..., 1]))) / h[0] - del_y(avg_x(avg_z(vels[..., 0]))) / h[1]
        return res



