from enum import Enum
from two_step_richtmeyer_util import Dimension
import numpy as np


class PDE_Type(Enum):
    Linear_advection = 1
    Burgers_equation = 2
    Euler = 3


class PDE:
    def __int__(self, *, dim: Dimension):
        self.dim = dim


class LinearAdvection(PDE):
    def __init__(self, a: np.ndarray, *, dim: Dimension):
        super().__init__(self, dim=dim)
        assert a.shape == (self.dim.value,)
        self.a = a
        self.ncomp = 1

    def __call__(self, v: np.ndarray) -> (np.ndarray, np.ndarray):
        return (self.a[i] * v for i in range(self.dim.value))

    def derivative(self, v: np.ndarray) -> np.ndarray:
        return np.full((*v.shape[0:self.dim.value], self.a.shape), self.a)


class BurgersEq(PDE):
    def __init__(self, *, dim: Dimension):
        super().__init__(self, dim=dim)
        self.ncomp = 1

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return 0.5 * np.square(v)

    def derivative(self, v: np.ndarray) -> np.ndarray:
        return v


class Euler(PDE):
    def __init__(self, gamma, dim: Dimension):
        super().__init__(self, dim=dim)
        self.gamma = gamma
        self.ncomp = self.dim.value + 2  # nr of dims velocities + density plus energy

    def pres(self, v):
        dens = v[..., 0]
        eint = v[..., -1]  # Etot

        # Ekin
        if self.dim >= Dimension.oneD:
            eint -= 0.5 * v[..., 1] ** 2 / dens
        if self.dim >= Dimension.twoD:
            eint = 0.5 * v[..., 1] ** 2 / dens
        if self.dim >= Dimension.threeD:
            eint = 0.5 * v[..., 1] ** 2 / dens

        return eint * (self.gamma - 1.)

    def csnd(self, v):
        p = self.pres(v)
        dens = v[..., 0]
        return np.sqrt(self.gamma * p / dens)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        # define p and E
        p = self.pres(v)
        res = np.ndarray(v.shape)
        res[..., 0] = v[..., 1]
        res[..., 1] = v[..., 1] ** 2 / v[..., 0] + p
        res[..., 2] = (v[..., 2] + p) * v[..., 1] / v[..., 0]
        return res

    # TODO
    def derivative(self, v: np.ndarray) -> np.ndarray:
        velx = v[..., 1] / v[..., 0]
        csnd = self.csnd(v)
        return np.abs(velx) + csnd
