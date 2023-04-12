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
    # TODO correct ?
    if dim >= Dimension.twoD:
        # y dimension
        grid[:, 0, ...] = grid[:, -2, ...]
        grid[:, -1, ...] = grid[:, 1, ...]
        # TODO necessary?
        # if dim == Dimension.twoD:
        #     grid[0, 0, ...] = grid[-2, -2, ...]
        #     grid[0, -1, ...] = grid[-2, 1, ...]
        #     grid[-1, 0, ...] = grid[1, -2, ...]
        #     grid[-1, -1, ...] = grid[1, 1, ...]
        #     # TODO same for 3D
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


# TODO implement these two
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



