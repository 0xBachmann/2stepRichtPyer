from mpl_toolkits.axes_grid1 import ImageGrid

from PDE_Types import EulerScalarAdvect
from richtmyer_two_step_scheme import Richtmyer2step
from two_step_richtmyer_util import *
from intitial import kelvin_helmholtz
from plotting_setup import *
from pathlib import Path

import numpy as np

log("definition of variables")

DIM = Dimension.twoD

domain = np.array([[0, 1], [0, 1]])
resolution = np.array([1024] * DIM.value)

h = ((domain[:, 1] - domain[:, 0]) / resolution).ravel()
F_visc = EulerScalarAdvect(5. / 3, dim=DIM, c1=.1, c2=1., hx=h[0], hy=h[1], add_viscosity=0, mu=0.001)
F_visc_1 = EulerScalarAdvect(5. / 3, dim=DIM, c1=.1, c2=1., hx=h[0], hy=h[1], add_viscosity=1, mu=0.001)
F = EulerScalarAdvect(5. / 3, dim=DIM)

log("calculate initial conditions")

# stepper = Richtmyer2stepImplicit(F, domain, resolution, eps=1e-9)
steppers = [Richtmyer2step(F, domain, resolution, lerp=-1),
            Richtmyer2step(F_visc, domain, resolution, lerp=-1),
            Richtmyer2step(F, domain, resolution, lerp=3),
            Richtmyer2step(F_visc, domain, resolution, lerp=3)]

stepper_names = ["default", "1", "2", "3"]


def kh_with_scalar(x: np.ndarray, F, Mr, pr=2.5, rhor=1., primitives=False):
    primitive = np.empty((*x.shape[:-1], 5))
    primitive[..., :-1] = kelvin_helmholtz(x, F, Mr=Mr, pr=pr, rhor=rhor, primitives=True)

    Y = x[..., 1]
    primitive[(Y < 0.25) | (0.75 <= Y), -1] = 0
    primitive[(0.25 <= Y) & (Y < 0.75), -1] = 1

    if primitives:
        return primitive
    else:
        return F.primitive_to_conserved(primitive)


M = 0.01
t = 3.
generate = True
if generate:
    for i, stepper in enumerate(steppers):
        if False:
            continue
        stepper.initial_cond(lambda x: kh_with_scalar(x, stepper.pde, Mr=M))

        stepper.step_for(3.)
        np.save(str(Path("traj", f"kh_{stepper_names[i]}.npy")), stepper.grid_no_ghost[..., -1])

plot = True
if plot:
    times_latex = [r"$t = 0.0$", r"$t = 7.0 \times 10^{-4}$", r"$t = 1.0$"]
    fig = plt.figure(figsize=(12, 12))  # Notice the equal aspect ratio
    # axs_mach = [fig.add_subplot(1, nM, i + 1) for i in range(nM)]

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(2, 2),
                     axes_pad=0.,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.,
                     )

    for j, ax in enumerate(grid):
        y = j // 2
        x = j % 2
        data = np.load(str(Path("traj", f"kh_{stepper_names[j]}.npy")))

        im = ax.imshow(data.T, origin="lower", vmin=0, vmax=1, extent=(0, 1, 0, 1))

        ax.set_yticks(np.linspace(0, 1, 5, endpoint=False))
        ax.set_xticks(np.linspace(0, 1, 5, endpoint=False))
        if x == 2:
            ax.set_xticks(np.linspace(0, 1, 6, endpoint=True))
        if y == 0:
            ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))
        ax.spines['bottom'].set_color('grey')
        ax.spines['top'].set_color('grey')
        ax.spines['right'].set_color('grey')
        ax.spines['left'].set_color('grey')
        ax.tick_params(color='grey', direction="in", top=True, right=True)
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$x$")

    # Colorbar
    cbar = ax.cax.colorbar(im, label=r"$\mathrm{Passive\ Scalar\ }X$")
    cbar.outline.set_edgecolor('grey')
    cbar.ax.tick_params(color='gray', direction="in")
    ax.cax.toggle_label(True)

    plt.savefig(Path("ims", f"kelvin_helmholtz_high_res.pdf"), dpi=200)





