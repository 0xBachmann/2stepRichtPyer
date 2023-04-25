import numpy as np


def gresho_vortex(x: np.ndarray, center, F, Mmax=None, qr=1) -> np.ndarray:
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
    return F.primitive_to_conserved(primitive)




