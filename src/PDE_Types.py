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

    def __call__(self, v: np.ndarray):
        raise NotImplementedError

    def derivative(self, v: np.ndarray):
        raise NotImplementedError

    def jacobian(self, v: np.ndarray):
        raise NotImplementedError


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

    def initial_cond(self, type):  # TODO
        def gaussian(x):
            pass

        def sin(x):
            pass

        def cos(x):
            pass

        if type == "gaussian":
            return gaussian
        if type == "cos":
            return cos
        if type == "sin":
            return sin

        raise NotImplementedError


class BurgersEq(PDE):
    def __init__(self, *, dim: Dimension):
        super().__init__(dim=dim, ncomp=dim.value, Type=PDE_Type.Burgers_equation)
        self.comp_names = [f"{vel}" for vel in "uvw"]

    def __call__(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        result = np.einsum("...i,...j->...ij", v, v) / 2
        return tuple(result[..., i] for i in range(self.dim.value))

    def derivative(self, v: np.ndarray) -> np.ndarray:
        if self.dim != Dimension.oneD:
            raise NotImplementedError
        return v


class Euler(PDE):
    def __init__(self, gamma, dim: Dimension):
        super().__init__(dim=dim, ncomp=dim.value + 2,
                         Type=PDE_Type.Euler)  # nr of dims velocities + density plus energy
        self.gamma = gamma
        self.comp_names = ["density", *[f"momenta_{dim}" for dim in "xyz"[:self.dim.value]], "Energy"]

    def pres(self, v):
        dens = v[..., 0]
        Etot = v[..., -1]
        Ekin = 0.5 * np.sum(v[..., 1:self.dim.value + 1] ** 2, axis=-1) / dens
        eint = Etot - Ekin
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
        Etot = v[..., -1]
        result = np.empty((*v.shape, self.dim.value))

        result[..., 0, :] = v[..., 1:self.dim.value + 1]
        result[..., 1:self.dim.value + 1, :] = np.einsum("...i,...j->...ij", vels, vels) * dens[
            ..., np.newaxis, np.newaxis] + np.einsum("...i,...jk->...ijk", p, np.identity(self.dim.value))
        result[..., -1, :] = np.einsum("...,...i->...i", Etot + p, vels)

        return tuple(result[..., i] for i in range(self.dim.value))

    # TODO
    def derivative(self, v: np.ndarray) -> np.ndarray:
        vels = v[..., 1:self.dim.value + 1] / v[..., 0][..., np.newaxis]
        csnd = self.csnd(v)
        return np.abs(vels) + csnd[..., np.newaxis]

    def jacobian(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        p = self.pres(v)
        dens = v[..., 0]
        Etot = v[..., -1]

        if self.dim == Dimension.oneD:
            vel = v[..., 1] / dens
            vel2 = vel ** 2
            
            J = np.empty((*v.shape, v.shape[-1]))

            J[..., 0, :] = np.array([0, 1, 0])

            J[..., 1, 0] = vel2 * (self.gamma - 3) / 2
            J[..., 1, 1] = vel * (3 - self.gamma)
            J[..., 1, 2] = self.gamma - 1

            J[..., 2, 0] = vel * (vel2[..., 0] * (self.gamma - 1) - Etot / dens)
            J[..., 2, 1] = vel2 * 1.5 * (1 - self.gamma) + self.gamma * Etot / dens
            J[..., 2, 2] = vel * self.gamma

            return J,

        if self.dim == Dimension.twoD:
            velx = v[..., 1] / dens
            vely = v[..., 2] / dens
            velx2 = velx ** 2
            vely2 = vely ** 2
            
            J = np.empty((self.dim.value, *v.shape, v.shape[-1]))

            # Df
            J[0, ..., 0, :] = np.array([0, 1, 0, 0])

            J[0, ..., 1, 0] = velx2 * ((self.gamma - 3) / 2) + vely2 * ((self.gamma - 1) / 2)
            J[0, ..., 1, 1] = velx * (3 - self.gamma)
            J[0, ..., 1, 2] = vely * (1 - self.gamma)
            J[0, ..., 1, 3] = self.gamma - 1

            J[0, ..., 2, 0] = velx * vely
            J[0, ..., 2, 1] = vely
            J[0, ..., 2, 2] = velx
            J[0, ..., 2, 3] = 0

            J[0, ..., 3, 0] = velx * ((velx2 + vely2) * ((self.gamma - 1) / 2) - self.gamma * Etot / dens)
            J[0, ..., 3, 1] = (3 * velx2 + vely2) * ((self.gamma - 1) / 2) + self.gamma * Etot / dens
            J[0, ..., 3, 2] = velx * vely * (1 - self.gamma)
            J[0, ..., 3, 3] = self.gamma * velx

            # Dg
            J[1, ..., 0, :] = np.array([0, 0, 1, 0])

            J[1, ..., 2, 0] = -velx * vely
            J[1, ..., 2, 1] = vely
            J[1, ..., 2, 2] = velx
            J[1, ..., 2, 3] = 0

            J[1, ..., 1, 0] = velx2 * ((self.gamma - 1) / 1) + vely2 * ((self.gamma - 3) / 2)
            J[1, ..., 1, 1] = velx * (1 - self.gamma)
            J[1, ..., 1, 2] = vely * (3 - self.gamma)
            J[1, ..., 1, 3] = self.gamma - 1

            J[1, ..., 3, 0] = vely * ((velx2 + vely2) * ((self.gamma - 1) / 2) - self.gamma * Etot / dens)
            J[1, ..., 3, 1] = velx * vely * (1 - self.gamma)
            J[1, ..., 3, 2] = (velx2 + 3 * vely2) * ((self.gamma - 1) / 2) + self.gamma * Etot / dens
            J[1, ..., 3, 3] = self.gamma * vely

            return J[0, ...], J[1, ...]

    def waves(self, k, w0, amp, alpha=None):
        if self.dim == Dimension.twoD:
            assert alpha is not None
        if self.dim == Dimension.threeD:
            raise NotImplementedError

        dens = w0[0]
        v = w0[1]
        p = w0[2]
        # conserved_w = np.array([dens, dens * v, p / (self.gamma - 1) + dens / 2 * v ** 2])
        a = np.sqrt(self.gamma * p / dens)
        eigen_vectors = np.array([[2, dens / a, -dens / a],
                                  [0, 1, 1],
                                  [0, dens * a, -dens * a]]) / 2
        eigen_vals = np.array([v, v + a, v - a])

        if self.dim == Dimension.twoD:
            cos_alpha = np.cos(alpha)
            Rinv = np.array([[cos_alpha, np.sin(alpha)],
                             [-np.sin(alpha), cos_alpha]])
            R = np.array([[cos_alpha, -np.sin(alpha)],
                          [np.sin(alpha), cos_alpha]])

        if self.dim == Dimension.oneD:
            def wave(x, t=0):
                """x needs to be normalized to [0, 1]"""
                w = w0 + amp * np.einsum("i,...j->...ji", eigen_vectors[:, k],
                                         np.sin(2 * np.pi * (x[..., 0] - eigen_vals[k] * t)))
                return self.primitive_to_conserved(w)
        elif self.dim == Dimension.twoD:
            def wave(x, t=0):
                """x needs to be normalized to [0, 1]"""
                rotx = x @ R
                # NOTE: whole input * 2 * cos_alpha in case of pi/4
                w = w0 + amp * np.einsum("i,...j->...ji", eigen_vectors[:, k],
                                         np.sin(2 * np.pi * (rotx[..., 0] - eigen_vals[k] * t)))
                # first rotate then transform
                # TODO correct??
                vxy = np.zeros((*w.shape[:-1], self.dim.value))
                vxy[..., 0] = w[..., 1]
                rotv = vxy @ Rinv

                w2d = np.empty((*w.shape[:-1], self.ncomp))
                w2d[..., 0] = w[..., 0]
                w2d[..., 1:self.dim.value + 1] = rotv
                w2d[..., 3] = w[..., 2]
                return self.primitive_to_conserved(w2d)
        else:
            raise NotImplementedError("Wave not implemented for 3D")

        return wave

    def conserved_to_primitive(self, v):
        dens = v[..., 0]
        mom = v[..., 1:self.dim.value + 1]
        E = v[..., -1]
        primitives = np.empty(v.shape)
        primitives[..., 0] = dens
        for i in range(self.dim.value):
            primitives[..., i + 1] = mom[..., i] / dens
        primitives[..., -1] = (self.gamma - 1) * (E - 0.5 * np.sum(mom ** 2, axis=-1)) / dens
        return primitives

    def primitive_to_conserved(self, w):
        dens = w[..., 0]
        v = w[..., 1:self.dim.value + 1]
        p = w[..., -1]
        conserved_w = np.empty(w.shape)
        conserved_w[..., 0] = dens
        for i in range(self.dim.value):
            conserved_w[..., i + 1] = dens * v[..., i]
        conserved_w[..., -1] = p / (self.gamma - 1) + 0.5 * dens * np.sum(v ** 2, axis=-1)
        return conserved_w
