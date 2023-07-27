import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from src.plotting_setup import *
from pathlib import Path


def GetPhaseSpaceAngles(dim, Np):
    """Compute dim-dimensional phase space angles."""
    # Phase angles in resp. direction
    if dim == 1:
        tx = np.linspace(-np.pi, +np.pi, Np)
        ty = np.array([0.])
        tz = np.array([0.])
    elif dim == 2:
        tx = np.linspace(-np.pi, +np.pi, Np)
        ty = np.linspace(-np.pi, +np.pi, Np)
        tz = np.array([0.])
    elif dim == 3:
        tx = np.linspace(-np.pi, +np.pi, Np)
        ty = np.linspace(-np.pi, +np.pi, Np)
        tz = np.linspace(-np.pi, +np.pi, Np)
    else:
        raise ValueError("Unexpected dimension dim = 1, 2, 3")
    [TX, TY, TZ] = np.meshgrid(tx, ty, tz, indexing='ij')
    return TX, TY, TZ


def StabilityAnalysis_diag(G, dim, c_range=[0., 1.2], Nc=101, Np=101):
    """Compute dim-dimensional amplification factor in a range of CFL
       numbers."""
    TX, TY, TZ = GetPhaseSpaceAngles(dim, Np)
    # Courant numbers in resp. direction
    c = np.linspace(c_range[0], c_range[1], Nc)
    Smax = np.zeros_like(c)
    for l in range(Nc):
        Smax[l] = np.max(np.abs(G(TX, TY, TZ, c[l], c[l], c[l])))
    return c, Smax


def MaxCFL(G, dim, c0=1.2, Np=101):
    """Determine dim-dimensional CFL condition."""
    TX, TY, TZ = GetPhaseSpaceAngles(dim, Np)
    f = lambda c: np.max(np.abs(G(TX, TY, TZ, c, c, c)) - 1.)
    CFL = fsolve(f, c0)
    return CFL


def G(tx, ty, tz, cx, cy, cz):
    """Compute amplification factor of two-step Richtmyer scheme in 1D, 2D and
       3D."""
    mu_x = np.cos(0.5*tx)
    mu_y = np.cos(0.5*ty)
    mu_z = np.cos(0.5*tz)
    d_x  = 2j*np.sin(0.5*tx)
    d_y  = 2j*np.sin(0.5*ty)
    d_z  = 2j*np.sin(0.5*tz)
    g = (  1.
         - cx*d_x*mu_x*mu_y**2*mu_z**2
         - cy*d_y*mu_y*mu_x**2*mu_z**2
         - cz*d_z*mu_z*mu_x**2*mu_y**2
         + 0.5*cx**2*d_x**2*mu_y**2*mu_z**2
         + 0.5*cy**2*d_y**2*mu_x**2*mu_z**2
         + 0.5*cz**2*d_z**2*mu_x**2*mu_y**2
         + cx*cy*d_x*mu_x*d_y*mu_y*mu_z**2
         + cx*cz*d_x*mu_x*d_z*mu_z*mu_y**2
         + cy*cz*d_y*mu_y*d_z*mu_z*mu_x**2)
    return g


if __name__ == "__main__":
    plt.figure(figsize=(7, 4.5))
    colors = ["palevioletred", "darkmagenta", "mediumpurple"]
    for dim in [1, 2, 3]:
        c, Smax = StabilityAnalysis_diag(G, dim)
        cmax = MaxCFL(G, dim)
        plt.plot(c, Smax, linestyle="-", color=colors[dim-1],
                 label=f"{dim} Dimension{'' if dim == 1 else 's'}")
        plt.axvline(cmax, linestyle="--", color=colors[dim-1])
    plt.xlim(0., 1.2)
    plt.xticks([0, 0.2, 0.4, 3**-0.5, 2**-0.5, 1, 1.2], labels=["0", "0.2", "0.4", r"$\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{2}}$", 1, 1.2])
    plt.xlabel(r"$c\ (c_{x} = c_{y} = c_{z})$")
    plt.ylabel(r"$\mathrm{Maximal\ amplification\ factor}$")
    # plt.title(r"$\mathrm{Explicit\ two}$-$\mathrm{step\ Richtmyer\ scheme}$")
    plt.legend(loc="upper left")
    plt.savefig(Path("ims", "stability_explicit_two_step_richtmyer.pdf"))
    plt.show()
