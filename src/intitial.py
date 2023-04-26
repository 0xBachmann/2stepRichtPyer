import numpy as np
from PDE_Types import Euler


def gresho_vortex(x: np.ndarray, center, F: Euler, Mmax=None, qr=1, primitives=False) -> np.ndarray:
    primitive = np.empty((*x.shape[:-1], 4))

    # offset domain
    X = x[..., 0] - center[0]
    Y = x[..., 1] - center[1]

    # get angle
    alpha = np.arctan2(Y, X)

    # calculate radius
    r2 = X ** 2 + Y ** 2
    r = np.sqrt(r2)

    # density
    rho = 1.
    primitive[..., 0] = rho

    # define u_phi
    u = np.zeros(x.shape[:-1])
    inner_ring = r < 0.2
    u[inner_ring] = 5. * r[inner_ring] * qr
    outer_ring = (0.2 <= r) & (r < 0.4)
    u[outer_ring] = (2. - 5. * r[outer_ring]) * qr

    # split u_phi int ux and uy
    primitive[..., 1] = u * np.sin(alpha)  # or Y/r
    primitive[..., 2] = u * -np.cos(alpha)  # or -x/r
    gamma = F.gamma

    # background pressure
    if Mmax is not None:
        p0 = rho / (gamma * Mmax ** 2) - 0.5
    else:
        p0 = 5.

    primitive[..., 3] = p0

    # pressure disturbance
    inner_ring = r < 0.4
    primitive[inner_ring, 3] += 25. / 2 * r2[inner_ring]
    primitive[outer_ring, 3] += 4. * (1. - 5. * r[outer_ring] - np.log(0.2) + np.log(r[outer_ring]))
    primitive[r >= 0.4, 3] += -2. + 4. * np.log(2)

    # convert to conserved variables
    if primitives:
        return primitive
    else:
        return F.primitive_to_conserved(primitive)


def sound_wave_packet(x: np.ndarray, F: Euler, x0, Mmax=None, alpha=100, primitives=False) -> np.ndarray:
    X = x[..., 0] - x0
    psi = np.exp(-alpha * X**2) * Mmax

    rho0 = 1.
    if Mmax is not None:
        p0 = rho0 / (F.gamma * Mmax ** 2) - 0.5
    else:
        p0 = 5.
        raise NotImplementedError
    c = np.sqrt(F.gamma * p0 / rho0)

    primitive = np.empty((*x.shape[:-1], 4))
    u = - 2 * alpha * X * psi
    p = - 2 * alpha * c / Mmax * X * psi
    rho = p / c**2

    primitive[..., 0] = rho
    primitive[..., 1] = u
    primitive[..., 2] = 0
    primitive[..., 3] = p

    if primitives:
        return primitive
    else:
        return F.primitive_to_conserved(primitive)


def kelvin_helmholtz(x: np.ndarray, F, Mr, pr=2.5, rhor=1., primitives=False, amp=1e-2) -> np.ndarray:
    primitive = np.empty((*x.shape[:-1], 4))
    X = x[..., 0]
    Y = x[..., 1]

    cr = np.sqrt(pr / rhor)
    ur = Mr * cr

    rho1 = 1.
    rho2 = 2.
    rhom = (rho2 - rho1) / 2

    u1 = 0.5
    u2 = -0.5
    um = (u2 - u1) / 2

    L = 0.025

    p = 2.5

    primitive[Y < 0.25, 0] = rho1 - rhom * np.exp((Y[Y < 0.25] - 0.25) / L)
    primitive[(0.25 <= Y) & (Y < 0.5), 0] = rho2 + rhom * np.exp((-Y[(0.25 <= Y) & (Y < 0.5)] + 0.25) / L)
    primitive[(0.5 <= Y) & (Y < 0.75), 0] = rho2 + rhom * np.exp((Y[(0.5 <= Y) & (Y < 0.75)] - 0.75) / L)
    primitive[0.75 <= Y, 0] = rho1 - rhom * np.exp((-Y[0.75 <= Y] + 0.75) / L)
    # primitive[..., 0] *= rhor

    primitive[Y < 0.25, 1] = u1 - um * np.exp((Y[Y < 0.25] - 0.25) / L)
    primitive[(0.25 <= Y) & (Y < 0.5), 1] = u2 + um * np.exp((-Y[(0.25 <= Y) & (Y < 0.5)] + 0.25) / L)
    primitive[(0.5 <= Y) & (Y < 0.75), 1] = u2 + um * np.exp((Y[(0.5 <= Y) & (Y < 0.75)] - 0.75) / L)
    primitive[0.75 <= Y, 1] = u1 - um * np.exp((-Y[0.75 <= Y] + 0.75) / L)

    primitive[..., 2] = amp * np.sin(2 * np.pi * X)
    # primitive[..., 1:3] *= ur

    primitive[..., 3] = p
    # primitive[..., 3] *= pr

    if primitives:
        return primitive
    else:
        return F.primitive_to_conserved(primitive)

