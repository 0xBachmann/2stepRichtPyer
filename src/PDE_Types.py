from enum import Enum
from two_step_richtmeyer_util import Dimension
import numpy as np


class PDE_Type(Enum):
    BaseClass = 0
    Linear_advection = 1
    Burgers_equation = 2
    Euler = 3


class PDE(object):
    def __init__(self, dim: Dimension, ncomp, Type: PDE_Type):
        self.dim = dim
        self.ncomp = ncomp
        self.comp_names = None
        self.Type = Type


class LinearAdvection(PDE):
    def __init__(self, a: np.ndarray, dim: Dimension):
        super().__init__(dim=dim, ncomp=1, Type=PDE_Type.Linear_advection)
        assert a.shape == (self.dim.value,)
        self.a = a
        self.comp_names = ["u"]

    def __call__(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(self.a[i] * v for i in range(self.dim.value))

    def derivative(self, v: np.ndarray) -> np.ndarray:
        return np.full((*v.shape[0:self.dim.value], *self.a.shape), self.a)

    def initial_cond(self):  # TODO
        def gaussian(x):
            pass


class BurgersEq(PDE):
    def __init__(self, *, dim: Dimension):
        super().__init__(dim=dim, ncomp=1, Type=PDE_Type.Burgers_equation)
        self.comp_names = ["u"]
        self.Type = PDE_Type.Burgers_equation

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return 0.5 * np.square(v)

    def derivative(self, v: np.ndarray) -> np.ndarray:
        return v


class Euler(PDE):
    def __init__(self, gamma, dim: Dimension):
        super().__init__(dim=dim, ncomp=dim.value + 2,
                         Type=PDE_Type.Euler)  # nr of dims velocities + density plus energy
        self.gamma = gamma
        self.comp_names = ["density", *[f"momenta_{d}" for d in "xyz"[:self.dim.value + 1]], "Energy"]

    def pres(self, v):
        dens = v[..., 0]
        eint = v[..., -1]  # Etot

        # Ekin
        for i in range(self.dim.value):
            eint -= 0.5 * v[..., i + 1]**2 / dens

        return eint * (self.gamma - 1.)

    def csnd(self, v):
        p = self.pres(v)
        dens = v[..., 0]
        return np.sqrt(self.gamma * p / dens)

    def __call__(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        if self.dim == Dimension.threeD:
            raise NotImplementedError

        # define p and E
        p = self.pres(v)
        dens = v[..., 0]
        vels = v[..., 1:self.dim.value + 1] / dens[..., np.newaxis]
        E = v[..., -1]
        result = np.empty((*v.shape, self.dim.value))

        # TODO error probably here somewhere
        result[..., 0, :] = v[..., 1:self.dim.value + 1]
        result[..., 1:self.dim.value + 1, :] = np.einsum("...i,...j->...ij", vels, vels) * dens[..., np.newaxis, np.newaxis] + np.einsum("...i,...jk->...ijk", p, np.identity(self.dim.value))
        result[..., -1, :] = np.einsum("...,...i->...i", E + p, vels)

        return tuple(result[..., i] for i in range(self.dim.value))

    # TODO
    def derivative(self, v: np.ndarray) -> np.ndarray:
        vels = v[..., 1:self.dim.value + 1] / v[..., 0]
        csnd = self.csnd(v)
        return np.abs(vels) + csnd

    def waves(self, k, w0, amp):
        if self.dim != Dimension.oneD:
            raise NotImplementedError

        dens = w0[0]
        v = w0[1]
        p = w0[2]
        conserved_w = np.array([dens, dens * v, p / (self.gamma - 1) + dens / 2 * v ** 2])
        a = np.sqrt(self.gamma * p / dens)
        eigen_vectors = np.array([[2, dens / a, -dens / a],
                                  [0, 1, 1],
                                  [0, dens * a, -dens * a]]) / 2
        eigen_vals = np.array([v, v + a, v - a])

        def wave(x, t):
            """x needs to be normalized to [0, 1]"""
            w = w0 + amp * np.einsum("i,...j->...ji", eigen_vectors[:, k], np.sin(2 * np.pi * (x - eigen_vals[k] * t)))
            return self.primitive_to_conserved(w)

        return wave

    def conserved_to_primitive(self, v):
        dens = v[..., 0]
        mom = v[..., 1:self.dim.value + 1]
        E = v[..., -1]
        primitives = np.empty(w.shape)
        primitives[..., 0] = dens
        for i in range(self.dim.value):
            primitives[..., i] = mom[..., i] / dens
        primitives[..., -1] = (self.gamma - 1) * (E - 0.5 * np.sum(mom**2, axis=-1)) / dens
        return primitives

    def primitive_to_conserved(self, w):
        dens = w[..., 0]
        v = w[..., 1:self.dim.value + 1]
        p = w[..., -1]
        conserved_w = np.empty(w.shape)
        conserved_w[..., 0] = dens
        for i in range(self.dim.value):
            conserved_w[..., i + 1] = dens * v[..., i]
        conserved_w[..., -1] = p / (self.gamma - 1) + 0.5 * dens * np.sum(v**2, axis=-1)
        return conserved_w
