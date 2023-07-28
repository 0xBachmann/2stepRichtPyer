from enum import Enum
from .two_step_richtmyer_util import *
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

    def max_speed(self, v: np.ndarray):
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

    def max_speed(self, v: np.ndarray) -> np.ndarray:
        return np.full((*v.shape[0:self.dim.value], *self.a.shape), self.a)

    def jacobian(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(np.full(np.product(*v.shape[:self.dim.value]), self.a[i]) for i in range(self.dim.value))


class BurgersEq(PDE):
    def __init__(self, *, dim: Dimension):
        super().__init__(dim=dim, ncomp=dim.value, Type=PDE_Type.Burgers_equation)
        self.comp_names = [f"{vel}" for vel in "uvw"]

    def __call__(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        result = np.einsum("...i,...j->...ij", v, v) / 2
        return tuple(result[..., i] for i in range(self.dim.value))

    def max_speed(self, v: np.ndarray) -> np.ndarray:
        if self.dim != Dimension.oneD:
            raise NotImplementedError
        return v

    def jacobian(self, v: np.ndarray):
        raise NotImplementedError


class Euler(PDE):
    def __init__(self, gamma, dim: Dimension, extra_comp=0, add_viscosity=-1, c1=0., c2=0., hx=0., hy=0., mu=1.):
        super().__init__(dim=dim, ncomp=dim.value + 2 + extra_comp,
                         Type=PDE_Type.Euler)  # nr of dims velocities + density plus energy_new
        self.gamma = gamma
        self.comp_names = ["density", *[f"momenta_{dim}" for dim in "xyz"[:self.dim.value]], "Energy"]
        self.add_viscosity = add_viscosity
        self.c1 = c1
        self.c2 = c2
        self.hx = hx
        self.hy = hy
        self.lz = np.minimum(hx, hy)
        self.mu = mu

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

    def eta(self, primary: np.ndarray, staggered: np.ndarray, dx, dy, which=-1) -> np.ndarray:
        if which == 0:
            return self.eta_post(staggered, dx, dy)[..., np.newaxis].astype(float)
        elif which == 1:
            return self.eta_(primary, dx, dy)
        elif which == 2:
            return self.eta_entropy(primary, dx, dy)[..., np.newaxis]
        elif which == 3:
            return self.eta_st(primary, kind="csnd")[..., np.newaxis]
        elif which == 4:
            return self.eta_st(primary, kind="entropy")[..., np.newaxis]
        else:
            raise ValueError

    def eta_post(self, v: np.ndarray, dx, dy) -> np.ndarray:
        """
        post-processing shock detection
        """
        assert self.dim == Dimension.twoD

        csnd = self.csnd(v)
        p = self.pres(v)

        def div(v: np.ndarray) -> np.ndarray:
            return del_x(avg_y(v[..., 0])) / dx + del_y(avg_x(v[..., 1])) / dy

        grad_p = np.empty((*[i - 1 for i in v.shape[:-1]], 2))
        grad_p[..., 0] = del_x(avg_y(p)) / dx
        grad_p[..., 1] = del_y(avg_x(p)) / dy

        mom = v[..., 1:3]
        vels = mom / v[..., 0:1]
        avg_vels = avg_x(avg_y(vels))

        eta = (1. - self.gamma) * (div((v[..., -1] + p)[..., np.newaxis] * vels)
                                   + 0.5 * np.sum(avg_vels ** 2, axis=-1) * div(mom)
                                   + np.sum(avg_vels * div(np.einsum("...i,...j->...ij", mom, vels)
                                                           + np.einsum("...i,jk->...ijk", p,
                                                                       np.identity(self.dim.value))), axis=-1)
                                   + np.sum(avg_vels * grad_p, axis=-1))

        norm_grad_p = np.linalg.norm(grad_p, axis=-1)
        tol = 0.0001
        # for test case 4
        # eta = np.clip(np.abs(eta) / (avg_x(avg_y(csnd)) * norm_grad_p * 0.3 + 1e-24), a_min=0, a_max=1)
        # eta[norm_grad_p < tol * np.max(norm_grad_p)] = 0.
        # return eta
        eta = np.clip(np.abs(eta) / (avg_x(avg_y(csnd)) * norm_grad_p * 1. + 1e-24), a_min=0, a_max=1)
        eta[norm_grad_p < tol * np.max(norm_grad_p)] = 0.
        return eta

        return (np.abs(eta) >= avg_x(avg_y(csnd)) * norm_grad_p * 1.) \
            & (norm_grad_p >= tol * np.max(norm_grad_p))  # & (norm_grad_p >= 0.5)
        # & (eta >= 0.5)
        # return np.clip(eta / avg_x(avg_y(csnd)) * norm_grad_p, a_min=0, a_max=1)
        eta[np.logical_not((np.abs(eta) >= avg_x(avg_y(csnd)) * norm_grad_p)
                           & (norm_grad_p >= tol * np.max(norm_grad_p))
                           # & (eta >= 0.5)
                           )] = 0.
        eta = np.abs(eta)
        return np.clip(eta, a_min=0, a_max=1)

    def eta_st(self, v: np.ndarray, kind="csnd") -> np.ndarray:
        assert self.dim == Dimension.twoD

        if kind == "csnd":
            c = self.csnd(v)
        elif kind == "entropy":
            p = self.pres(v)
            c = p / np.power(v[..., 0], self.gamma)
        else:
            raise ValueError

        w = 0.5
        d = 1e-9
        thetax = np.abs((c[2:, ...] - 2 * c[1:-1, ...] + c[:-2, ...]) /
                        ((1. - w) * (np.abs(c[2:, ...] - c[1:-1, ...]) + np.abs(c[1:-1, ...] - c[:-2, ...]))
                         + w * (np.abs(c[2:, ...]) + 2 * np.abs(c[1:-1, ...]) + np.abs(c[:-2, ...])) + d))
        thetay = np.abs((c[:, 2:, ...] - 2 * c[:, 1:-1, ...] + c[:, :-2, ...]) /
                        ((1. - w) * (np.abs(c[:, 2:, ...] - c[:, 1:-1, ...]) + np.abs(c[:, 1:-1, ...] - c[:, :-2, ...]))
                         + w * (np.abs(c[:, 2:, ...]) + 2 * np.abs(c[:, 1:-1, ...]) + np.abs(c[:, :-2, ...])) + d))
        eta = (thetax[:, 1:-1, ...] + thetay[1:-1, ...]) * 2.5 # factor possible
        return np.clip(eta, a_min=0, a_max=1)

    def eta_entropy(self, v: np.ndarray, dx, dy) -> np.ndarray:
        assert self.dim == Dimension.twoD

        p = self.pres(v)
        rho = v[..., 0]

        K = p / rho**self.gamma
        eta = np.maximum(del_x(avg_x(K))[:, 1:-1, ...] / dx
                         , del_y(avg_y(K))[1:-1, ...] / dy
                         )
        return np.clip(eta, a_min=0, a_max=1)

    def eta_(self, v: np.ndarray, dx, dy) -> np.ndarray:
        """
        the one from Baslare et al
        """
        assert self.dim == Dimension.twoD

        csnd = self.csnd(v)
        min_c = np.minimum.reduce([
            csnd[1:-1, 1:-1],
            csnd[1:-1, 2:],
            csnd[1:-1, :-2],
            csnd[2:, 1:-1],
            csnd[2:, 2:],
            csnd[2:, :-2],
            csnd[:-2, 1:-1],
            csnd[:-2, 2:],
            csnd[:-2, :-2],
        ])

        k1 = 0.004
        vels = v[..., 1:3] / v[..., 0, np.newaxis]
        div = 2 * (avg_x(del_x(vels[..., 0]))[..., 1:-1] / dx + avg_y(del_y(vels[..., 1]))[1:-1, ...] / dy)
        eta = np.clip(-dx * div / (k1 * min_c) - 1, a_min=0, a_max=1)[..., np.newaxis]

        p = self.pres(v)[1:-1, 1:-1, np.newaxis]

        np.putmask(eta[1:], (eta > 0)[:-1] & np.isclose(eta[1:], 0) & (p[:-1] > p[1:]), eta[:-1])
        np.putmask(eta[:-1], (eta > 0)[1:] & np.isclose(eta[:-1], 0) & (p[:-1] < p[1:]), eta[1:])
        # eta[1:][(eta > 0)[:-1] & (eta[1:] == 0) & (p[:-1] > p[1:])] = eta[:-1]
        # eta[:-1][(eta > 0)[1:] & (eta[:-1] == 0) & (p[:-1] < p[1:])] = eta[1:]

        np.putmask(eta[:, 1:], (eta > 0)[:, :-1] & np.isclose(eta[:, 1:], 0) & (p[:, :-1] > p[:, 1:]), eta[:, :-1])
        np.putmask(eta[:, :-1], (eta > 0)[:, 1:] & np.isclose(eta[:, :-1], 0) & (p[:, :-1] < p[:, 1:]), eta[:, 1:])
        # eta[:, 1:][(eta > 0)[:, :-1] & (eta[:, 1:] == 0) & (p[:, :-1] > p[:, 1:])] = eta[:, :-1]
        # eta[:, -1][(eta > 0)[:, 1:] & (eta[:, :-1] == 0) & (p[:, :-1] < p[:, 1:])] = eta[:, 1:]
        return eta

    def viscosity_ns(self, v: np.ndarray, other: np.ndarray) -> np.ndarray:
        vels = np.empty(tuple([d + 2 for d in other.shape[:-1]] + [2]))
        vels[1:-1, 1:-1, ...] = other[..., 1:3] / other[..., 0, np.newaxis]

        vels[0, ...] = vels[-3, ...]
        vels[-1, ...] = vels[2, ...]
        vels[:, 0, ...] = vels[:, -3, ...]
        vels[:, -1, ...] = vels[:, 2, ...]

        vx = vels[..., 0]
        vy = vels[..., 1]

        def dx(v: np.ndarray) -> np.ndarray:
            return avg_y(del_x(v)) / self.hx

        def dy(v: np.ndarray) -> np.ndarray:
            return avg_x(del_y(v)) / self.hy

        tau = np.empty(tuple([d for d in v.shape[:-1]] + [2, 2]))
        tau[..., 0, 0] = 2 * dx(vx)
        tau[..., 0, 1] = (dx(vy) + dy(vx))
        tau[..., 1, 0] = (dx(vy) + dy(vx))
        tau[..., 1, 1] = 2 * dy(vy)

        tau -= 2. / 3 * np.einsum("...i,jk->...ijk", dx(vx) + dy(vy), np.eye(2, 2))

        return self.mu * tau


    def viscosity2(self, v: np.ndarray, other: np.ndarray) -> np.ndarray:
        csnd = self.csnd(v)

        vels = np.empty(tuple([d + 2 for d in other.shape[:-1]] + [2]))
        vels[1:-1, 1:-1, ...] = other[..., 1:3] / other[..., 0, np.newaxis]

        vels[0, ...] = vels[-3, ...]
        vels[-1, ...] = vels[2, ...]
        vels[:, 0, ...] = vels[:, -3, ...]
        vels[:, -1, ...] = vels[:, 2, ...]

        vx = vels[..., 0]
        vy = vels[..., 1]

        velocity_diff = np.maximum.reduce([
            np.linalg.norm(vels[1:, 1:, ...] - vels[1:, :-1, ...], axis=-1),
            np.linalg.norm(vels[1:, 1:, ...] - vels[:-1, 1:, ...], axis=-1),
            np.linalg.norm(vels[1:, 1:, ...] - vels[:-1, :-1, ...], axis=-1),
            np.linalg.norm(vels[:-1, :-1, ...] - vels[1:, :-1, ...], axis=-1),
            np.linalg.norm(vels[:-1, :-1, ...] - vels[:-1, 1:, ...], axis=-1),
            np.linalg.norm(vels[:-1, 1:, ...] - vels[1:, :-1, ...], axis=-1),
        ])

        q = v[..., 0] * (self.c1 * csnd + self.c2 * velocity_diff) * self.lz

        def dx(v: np.ndarray) -> np.ndarray:
            return avg_y(del_x(v)) / self.hx

        def dy(v: np.ndarray) -> np.ndarray:
            return avg_x(del_y(v)) / self.hy

        eps = np.empty(tuple([d for d in v.shape[:-1]] + [2, 2]))
        eps[..., 0, 0] = dx(vx)
        eps[..., 0, 1] = 0.5 * (dx(vy) + dy(vx))
        eps[..., 1, 0] = 0.5 * (dx(vy) + dy(vx))
        eps[..., 1, 1] = dy(vy)

        Q = eps - 1. / 3 * np.einsum("...i,jk->...ijk", dx(vx) + dy(vy), np.eye(2, 2))

        return q[..., np.newaxis, np.newaxis] * Q

    def viscosity(self, v: np.ndarray) -> np.ndarray:
        csnd = self.csnd(v)

        vels = np.empty(tuple([d + 2 for d in v.shape[:-1]] + [2]))
        vels[1:-1, 1:-1, ...] = v[..., 1:3] / v[..., 0, np.newaxis]

        vels[0, ...] = vels[-3, ...]
        vels[-1, ...] = vels[2, ...]
        vels[:, 0, ...] = vels[:, -3, ...]
        vels[:, -1, ...] = vels[:, 2, ...]

        vx = vels[..., 0]
        vy = vels[..., 1]

        velocity_diff = np.maximum.reduce([
            np.linalg.norm(vels[2:, 1:-1, ...] - vels[:-2, 1:-1, ...], axis=-1),
            np.linalg.norm(vels[1:-1, 2:, ...] - vels[1:-1, :-2, ...], axis=-1),
            np.linalg.norm(vels[2:, 2:, ...] - vels[:-2, :-2, ...], axis=-1),
            np.linalg.norm(vels[:-2, 2:, ...] - vels[2:, :-2, ...], axis=-1),
        ])

        q = v[..., 0] * (self.c1 * csnd + self.c2 * velocity_diff) * self.lz

        def dx(v: np.ndarray) -> np.ndarray:
            return (v[2:, ...] - v[:-2, ...])[:, 1:-1, ...] / (2 * self.hx)

        def dy(v: np.ndarray) -> np.ndarray:
            return (v[:, 2:, ...] - v[:, :-2, ...])[1:-1, ...] / (2 * self.hy)

        eps = np.empty(tuple([d for d in v.shape[:-1]] + [2, 2]))
        eps[..., 0, 0] = dx(vx)
        eps[..., 0, 1] = 0.5 * (dx(vy) + dy(vx))
        eps[..., 1, 0] = 0.5 * (dx(vy) + dy(vx))
        eps[..., 1, 1] = dy(vy)

        Q = eps - 1. / 3 * np.einsum("...i,jk->...ijk", dx(vx) + dy(vy), np.eye(2, 2))

        return q[..., np.newaxis, np.newaxis] * Q

    def __call__(self, v: np.ndarray, visc=True, other=None) -> tuple[np.ndarray, ...]:
        # define p and E
        p = self.pres(v)
        dens = v[..., 0]
        vels = v[..., 1:self.dim.value + 1] / dens[..., np.newaxis]
        Etot = v[..., -1]
        result = np.empty((*v.shape, self.dim.value))

        result[..., 0, :] = v[..., 1:self.dim.value + 1]
        result[..., 1:self.dim.value + 1, :] = np.einsum("...i,...j->...ij", v[..., 1:self.dim.value + 1], vels) \
                                               + np.einsum("...i,jk->...ijk", p, np.identity(self.dim.value))
        result[..., -1, :] = np.einsum("...,...i->...i", Etot + p, vels)

        if self.add_viscosity >= 0 and visc:
            if self.dim != Dimension.twoD:
                raise NotImplementedError("Viscosity only possible for 2D")
            if self.add_viscosity == 0:  # artificial
                # version 1: dx = mux delx
                if False:
                    result[..., 1:3, :] -= self.viscosity(v)
                # version 2: dx = muy delx
                else:
                    result[..., 1:3, :] -= self.viscosity2(v, avg_x(avg_y(v)) if other is None else other)
            elif self.add_viscosity == 1:
                tau = self.viscosity_ns(v, avg_x(avg_y(v)) if other is None else other)
                result[..., 1:3, :] -= tau
                result[..., -1, :] += np.einsum("...ij,...j->...i", tau, vels)
            else:
                raise ValueError
        return tuple(result[..., i] for i in range(self.dim.value))

    def max_speed(self, v: np.ndarray) -> np.ndarray:
        vels = v[..., 1:self.dim.value + 1] / v[..., 0][..., np.newaxis]
        csnd = self.csnd(v)
        return np.abs(vels) + csnd[..., np.newaxis]

    def mach(self, v: np.ndarray) -> np.ndarray:
        vels = v[..., 1:self.dim.value + 1] / v[..., 0][..., np.newaxis]
        M = np.linalg.norm(vels, axis=-1) / self.csnd(v)
        # M = np.sqrt(np.sum(vels ** 2, axis=-1)) / self.csnd(v)
        return M

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

        elif self.dim == Dimension.twoD:
            velx = v[..., 1] / dens
            vely = v[..., 2] / dens
            velx2 = velx ** 2
            vely2 = vely ** 2

            JF = np.empty((*v.shape, v.shape[-1]))
            JG = np.empty((*v.shape, v.shape[-1]))

            # Df
            JF[..., 0, :] = np.array([0, 1, 0, 0])

            JF[..., 1, 0] = velx2 * ((self.gamma - 3) / 2) + vely2 * ((self.gamma - 1) / 2)
            JF[..., 1, 1] = velx * (3 - self.gamma)
            JF[..., 1, 2] = vely * (1 - self.gamma)
            JF[..., 1, 3] = self.gamma - 1

            JF[..., 2, 0] = velx * vely
            JF[..., 2, 1] = vely
            JF[..., 2, 2] = velx
            JF[..., 2, 3] = 0

            JF[..., 3, 0] = velx * ((velx2 + vely2) * ((self.gamma - 1) / 2) - self.gamma * Etot / dens)
            JF[..., 3, 1] = (3 * velx2 + vely2) * ((self.gamma - 1) / 2) + self.gamma * Etot / dens
            JF[..., 3, 2] = velx * vely * (1 - self.gamma)
            JF[..., 3, 3] = self.gamma * velx

            # Dg
            JG[..., 0, :] = np.array([0, 0, 1, 0])

            JG[..., 2, 0] = -velx * vely
            JG[..., 2, 1] = vely
            JG[..., 2, 2] = velx
            JG[..., 2, 3] = 0

            JG[..., 1, 0] = velx2 * ((self.gamma - 1) / 2) + vely2 * ((self.gamma - 3) / 2)
            JG[..., 1, 1] = velx * (1 - self.gamma)
            JG[..., 1, 2] = vely * (3 - self.gamma)
            JG[..., 1, 3] = self.gamma - 1

            JG[..., 3, 0] = vely * ((velx2 + vely2) * ((self.gamma - 1) / 2) - self.gamma * Etot / dens)
            JG[..., 3, 1] = velx * vely * (1 - self.gamma)
            JG[..., 3, 2] = (velx2 + 3 * vely2) * ((self.gamma - 1) / 2) + self.gamma * Etot / dens
            JG[..., 3, 3] = self.gamma * vely

            return JF, JG
        else:
            raise NotImplementedError

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
                # TODO need * 2 * cos_alpha if rotated by 45°
                w = w0 + amp * np.einsum("i,...j->...ji", eigen_vectors[:, k],
                                         np.sin(2 * np.pi * (rotx[..., 0] - eigen_vals[k] * t)))
                # first rotate then transform
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
        primitives[..., -1] = (self.gamma - 1) * (E - 0.5 * np.sum(mom ** 2, axis=-1) / dens)
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

    def angular_momenta(self, v: np.ndarray, XYZ) -> np.ndarray:
        return np.cross(XYZ, v[..., 1:self.dim.value + 1])

    def derivatives(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        csnd = self.csnd(v)
        absvels = np.abs(v[..., 1:self.dim.value + 1] / v[..., 0, np.newaxis])
        if self.dim == Dimension.twoD:
            return np.maximum(absvels[1:, ..., 0], absvels[:-1, ..., 0]) + np.maximum(csnd[1:, ...], csnd[:-1, ...]), \
                   np.maximum(absvels[:, 1:, ..., 1], absvels[:, :-1, ..., 1]) + np.maximum(csnd[:, 1:, ...],
                                                                                            csnd[:, :-1, ...])
        else:
            raise NotImplementedError


class EulerScalarAdvect(Euler):
    def __init__(self, gamma, dim: Dimension, add_viscosity=-1, c1=0, c2=0, hx=0., hy=0., mu=1.):
        super().__init__(gamma, dim, extra_comp=1, add_viscosity=add_viscosity, c1=c1, c2=c2, hx=hx,
                         hy=hy, mu=mu)  # nr of dims velocities + density plus energy_new
        self.comp_names = ["density", *[f"momenta_{dim}" for dim in "xyz"[:dim.value]], "Energy", "X"]

    def pres(self, v):
        dens = v[..., 0]
        Etot = v[..., -2]
        Ekin = 0.5 * np.sum(v[..., 1:self.dim.value + 1] ** 2, axis=-1) / dens
        eint = Etot - Ekin
        return eint * (self.gamma - 1.)

    def __call__(self, v: np.ndarray, visc=True, other=None) -> tuple[np.ndarray, ...]:
        # define p and E
        p = self.pres(v)
        dens = v[..., 0]
        vels = v[..., 1:self.dim.value + 1] / dens[..., np.newaxis]
        Etot = v[..., -2]
        X = v[..., -1]
        result = np.empty((*v.shape, self.dim.value))

        result[..., 0, :] = v[..., 1:self.dim.value + 1]
        result[..., 1:self.dim.value + 1, :] = np.einsum("...i,...j->...ij", v[..., 1:self.dim.value + 1], vels) \
                                               + np.einsum("...i,jk->...ijk", p, np.identity(self.dim.value))
        result[..., -2, :] = np.einsum("...,...i->...i", Etot + p, vels)
        result[..., -1, :] = np.einsum("...,...i->...i", X, v[..., 1:self.dim.value + 1])

        if self.add_viscosity >= 0 and visc:
            if self.dim != Dimension.twoD:
                raise NotImplementedError("Viscosity only possible for 2D")
            if self.add_viscosity == 0:  # artificial
                # version 1: dx = mux delx
                if False:
                    result[..., 1:3, :] -= self.viscosity(v)
                # version 2: dx = muy delx
                else:
                    result[..., 1:3, :] -= self.viscosity2(v, avg_x(avg_y(v)) if other is None else other)
            elif self.add_viscosity == 1:
                tau = self.viscosity_ns(v, avg_x(avg_y(v)) if other is None else other)
                result[..., 1:3, :] -= tau
                result[..., -2, :] += np.einsum("...ij,...j->...i", tau, vels)
            else:
                raise ValueError
        return tuple(result[..., i] for i in range(self.dim.value))

    def jacobian(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        dens = v[..., 0]
        X = v[..., -1] / dens

        if self.dim == Dimension.oneD:
            vel = v[..., 1] / dens
            J = np.empty((*v.shape, v.shape[-1]))

            J[..., :-1, :-1] = super().jacobian(v[..., :-1])
            J[..., :-1, -1] = 0

            J[..., -1, 0] = -vel * X
            J[..., -1, 1] = X
            J[..., -1, 2] = 0
            J[..., -1, 3] = vel

            return J,

        elif self.dim == Dimension.twoD:
            velx = v[..., 1] / dens
            vely = v[..., 2] / dens

            JF = np.empty((*v.shape, v.shape[-1]))
            JG = np.empty((*v.shape, v.shape[-1]))

            JF[..., :-1, :-1], JG[..., :-1, :-1] = super().jacobian(v[..., :-1])
            JF[..., :-1, -1] = 0
            JG[..., :-1, -1] = 0

            # Df
            JF[..., -1, 0] = -velx * X
            JF[..., -1, 1] = X
            JF[..., -1, 2:4] = 0
            JF[..., -1, 4] = velx

            # Dg
            JG[..., -1, 0] = -vely * X
            JG[..., -1, 1] = 0
            JG[..., -1, 2] = X
            JG[..., -1, 3] = 0
            JG[..., -1, 4] = vely

            return JF, JG
        else:
            raise NotImplementedError

    def conserved_to_primitive(self, v):
        dens = v[..., 0]
        mom = v[..., 1:self.dim.value + 1]
        E = v[..., -2]
        primitives = np.empty(v.shape)
        primitives[..., 0] = dens
        for i in range(self.dim.value):
            primitives[..., i + 1] = mom[..., i] / dens
        primitives[..., -2] = (self.gamma - 1) * (E - 0.5 * np.sum(mom ** 2, axis=-1)) / dens
        primitives[..., -1] = v[..., -1] / dens
        return primitives

    def primitive_to_conserved(self, w):
        dens = w[..., 0]
        v = w[..., 1:self.dim.value + 1]
        p = w[..., -2]
        conserved_w = np.empty(w.shape)
        conserved_w[..., 0] = dens
        for i in range(self.dim.value):
            conserved_w[..., i + 1] = dens * v[..., i]
        conserved_w[..., -2] = p / (self.gamma - 1) + 0.5 * dens * np.sum(v ** 2, axis=-1)
        conserved_w[..., -1] = w[..., -1] * dens
        return conserved_w


class EulerNondimensional(Euler):
    def __init__(self, gamma, dim: Dimension, Mr):
        super().__init__(gamma, dim)
        self.mr = Mr

    def pres(self, v):
        dens = v[..., 0]
        Etot = v[..., -2]
        Ekin = 0.5 * np.sum(v[..., 1:self.dim.value + 1] ** 2, axis=-1) / dens
        eint = Etot - self.mr * Ekin
        return eint * (self.gamma - 1.)

    def __call__(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        # define p and E
        p = self.pres(v)
        dens = v[..., 0]
        vels = v[..., 1:self.dim.value + 1] / dens[..., np.newaxis]
        Etot = v[..., -2]
        X = v[..., -1]
        result = np.empty((*v.shape, self.dim.value))

        result[..., 0, :] = v[..., 1:self.dim.value + 1]
        result[..., 1:self.dim.value + 1, :] = np.einsum("...i,...j->...ij", v[..., 1:self.dim.value + 1], vels) \
                                               + np.einsum("...i,jk->...ijk", p / self.mr ** 2,
                                                           np.identity(self.dim.value))
        result[..., -2, :] = np.einsum("...,...i->...i", Etot + p, vels)
        result[..., -1, :] = np.einsum("...,...i->...i", X, v[..., 1:self.dim.value + 1])

        return tuple(result[..., i] for i in range(self.dim.value))

    def jacobian(self, v: np.ndarray) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def conserved_to_primitive(self, v):
        dens = v[..., 0]
        mom = v[..., 1:self.dim.value + 1]
        E = v[..., -1]
        primitives = np.empty(v.shape)
        primitives[..., 0] = dens
        for i in range(self.dim.value):
            primitives[..., i + 1] = mom[..., i] / dens
        primitives[..., -1] = (self.gamma - 1) * (E - self.mr * 0.5 * np.sum(mom ** 2, axis=-1)) / dens
        return primitives

    def primitive_to_conserved(self, w):
        dens = w[..., 0]
        v = w[..., 1:self.dim.value + 1]
        p = w[..., -1]
        conserved_w = np.empty(w.shape)
        conserved_w[..., 0] = dens
        for i in range(self.dim.value):
            conserved_w[..., i + 1] = dens * v[..., i]
        conserved_w[..., -1] = p / (self.gamma - 1) + self.mr * 0.5 * dens * np.sum(v ** 2, axis=-1)
        return conserved_w
