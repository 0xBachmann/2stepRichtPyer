from PDE_Types import *
from richtmeyer_two_step_scheme import Richtmeyer2step
from two_step_richtmeyer_util import Dimension
from plotting_setup import *
from matplotlib.ticker import FormatStrFormatter

from pathlib import Path


from RP1D_Euler import RP1D_Euler

import numpy as np

DIM = Dimension.twoD
n = 320
resolution = np.array([n, 80])
Lx = 1
Ly = Lx
h = [Lx / resolution[0], Ly / resolution[1]]


lerp = 3
visc = False
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=1., hx=h[0], hy=h[1], add_viscosity=visc)
stepper = Richtmeyer2step(F, np.array([Lx, Ly]), resolution, lerp=lerp)

ntest = 7
tests = [(1.0, 0.75, 1.0, 0.125, 0.0, 0.1),  # rusanov works
         (1.0, -2.0, 0.4, 1.0, 2.0, 0.4),  # not even with visc
         (1.0, 0.0, 1000.0, 1.0, 0.0, 0.01),  # also not too bad
         (5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950),  # also okayish
         (1.0, -19.59745, 1000.0, 1.0, -19.59745, 0.01),  # not even with visc
         (1.4, 0.0, 1.0, 1.0, 0.0, 1.0),  # works fine
         (1.4, 0.1, 1.0, 1.0, 0.1, 1.0)]  # densitiy oscilates

times = [0.06, 0.06, 0.0015, 0.005, 0.0015, 0.05, 0.05]

for i in range(ntest):
    rhol, ul, pl, rhor, ur, pr = tests[i]

    RP_exact_l = RP1D_Euler(5. / 3., rhol, ul, pl, rhor, ur, pr, xdiaph=0.25)
    RP_exact_r = RP1D_Euler(5. / 3., rhor, ur, pr, rhol, ul, pl, xdiaph=0.75)

    def test(x):
        quart = x.shape[0] // 4
        threq = 3 * quart

        result = np.empty((*x.shape[:-1], 4))

        result[:quart, ..., 0] = rhol
        result[quart:threq, ..., 0] = rhor
        result[threq:, ..., 0] = rhol

        result[:quart, ..., 1] = ul
        result[quart:threq, ..., 1] = ur
        result[threq:, ..., 1] = ul

        result[:, ..., 2] = 0

        result[:quart, ..., 3] = pl
        result[quart:threq, ..., 3] = pr
        result[threq:, ..., 3] = pl

        return F.primitive_to_conserved(result)

    stepper.initial_cond(test)

    T = times[i]
    cfl = stepper.cfl()
    stepper.step_for(T, const_dt=True)

    coords = np.linspace(0, 1, resolution[0])
    coords_exact = np.linspace(0, 0.5, 400)

    msz = 8
    plot_approx = {"linewidth": 0.7, "s": 10, "marker": "^", "facecolors": 'none', "edgecolors": 'darkmagenta', "zorder": 1}
    plot_exact = {"linewidth": 0.7, "linestyle": "solid", "color": "mediumpurple", "zorder": -1}  # "palevioletred", "darkmagenta", "mediumpurple"

    fig, axs = plt.subplots(2, 2)
    for j in range(2):
        for k in range(2):
            axs[j, k].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # exact
    exact_l = np.stack(RP_exact_l.sample(coords_exact, T), axis=-1)
    exact = exact_l
    axs[0, 0].plot(coords_exact, exact[..., 0], **plot_exact)
    axs[0, 0].set_ylabel(r"$\mathrm{Density}$")
    axs[0, 1].plot(coords_exact, exact[..., 1], **plot_exact)
    axs[0, 1].set_ylabel(r"$\mathrm{Velocity}$")
    axs[1, 0].plot(coords_exact, exact[..., 2], **plot_exact)
    axs[1, 0].set_ylabel(r"$\mathrm{Pressure}$")
    # todo, subtract kinetic smth wrong
    internal_energy_exact = exact[..., 2] / (F.gamma - 1) / exact[..., 0]
    axs[1, 1].plot(coords_exact, internal_energy_exact, **plot_exact)
    axs[1, 1].set_ylabel(r"$\mathrm{Internal\ energy}$")

    primitives = F.conserved_to_primitive(stepper.grid_no_ghost)[..., 0, (0, 1, 3)]
    # approx
    half = coords.shape[0] // 2
    axs[0, 0].scatter(coords[:half], primitives[:half, 0], **plot_approx)
    axs[0, 1].scatter(coords[:half], primitives[:half, 1], **plot_approx)
    axs[1, 0].scatter(coords[:half], primitives[:half, 2], **plot_approx)
    # todo, subtract kinetic smth wrong
    internal_energy = (stepper.grid_no_ghost[..., 0, -1] - 0.5 * primitives[..., 0] * np.power(primitives[..., 1], 2)) / primitives[..., 0]
    axs[1, 1].scatter(coords[:half], internal_energy[:half], **plot_approx)

    fig.tight_layout()
    plt.savefig(Path("sod_shock", f"sod_{lerp}_{int(visc)}_{i}.pdf"))

