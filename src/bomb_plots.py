from pathlib import Path

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid

from PDE_Types import *
from plotter import Plotter
from richtmyer_two_step_scheme import Richtmyer2step, Richtmyer2stepImplicit
from two_step_richtmyer_util import Dimension, log, avg_x
from intitial import gresho_vortex, isentropic_vortices

from plotting_setup import *


import copy
import numpy as np

colors = ["darkmagenta", "mediumpurple"]  # , "teal", "darkkhaki"]
cycler = plt.cycler(color=colors)  # + plt.cycler(lw=[1, 1, 1, 1])  # + plt.cycler(markersize=[6, 3, 2, 1])
plt.rc('axes', prop_cycle=cycler)
mpl.rc('image', cmap='magma_r')


log("definition of variables")

DIM = Dimension.twoD
l = 10
domain = np.array([[-l, l], [-l, l]])
resolution = np.array([400] * DIM.value)

h = ((domain[:, 1] - domain[:, 0]) / resolution).ravel()
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=1., hx=h[0], hy=h[1], add_viscosity=-1)

log("calculate initial conditions")

# stepper = Richtmyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-8)
steppers = [Richtmyer2step(F, domain, resolution, lerp=-1), Richtmyer2stepImplicit(F, domain, resolution, eps=1e-16)]
nS = len(steppers)
stepper_names = ["expl", "impl"]

M = 0.1
t = 6.


def vortex_bomb(x, bcenter=np.array([0, 0]), n_v=3):
    # centers = [np.array([np.sin(i * 2 * np.pi / n_v), np.cos(i * 2 * np.pi / n_v)]) for i in range(n_v)]
    # vortices = gresho_vortex(x, centers[0], F, M, qr=0.4 * np.pi, primitives=True)
    # for i in range(1, n_v):
    #     vortices += gresho_vortex(x, centers[i], F, M, qr=0.4 * np.pi, primitives=True, background=False)
    vortices = isentropic_vortices(x,
        [l / 2 * np.array([np.sin(i * 2 * np.pi / n_v), np.cos(i * 2 * np.pi / n_v)]) for i in range(n_v)], beta=5, F=F, primitives=True)
    X = x[..., 0] - bcenter[0]
    Y = x[..., 1] - bcenter[1]
    vortices[..., -1] += np.exp(-3 * (X**2 + Y**2))
    return F.primitive_to_conserved(vortices)


generate = False
if generate:
    for i, stepper in enumerate(steppers):
        avg_coords = [avg_x(coord) for coord in stepper.coords]
        X, Y = np.meshgrid(*avg_coords, indexing='ij')
        XY = np.empty((*resolution, 2))
        XY[..., 0] = X
        XY[..., 1] = Y
        j = []
        times = []

        def angular(stepper, dt):
            global j, times
            j.append(np.sum(F.angular_momenta(stepper.grid_no_ghost, XY)))
            times.append(dt)

        stepper.initial_cond(vortex_bomb)

        stepper.step_for(t, fact=1, callback=angular)
        np.save(str(Path("traj", f"bomb_{stepper_names[i]}.npy")), stepper.grid_no_ghost[..., 0])

        np.save(str(Path("ang", f"bomb_{stepper_names[i]}_j.npy")), np.array(j))
        np.save(str(Path("ang", f"bomb_{stepper_names[i]}_times.npy")), np.array(times))

plot = True
if plot:
    # density plot
    fig = plt.figure(figsize=(4 * nS + 1, 4))  # Notice the equal aspect ratio
    # axs_mach = [fig.add_subplot(1, nM, i + 1) for i in range(nM)]

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, nS),
                     axes_pad=0.,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.,
                     )

    for j, ax in enumerate(grid):
        data = np.load(str(Path("traj", f"bomb_{stepper_names[j]}.npy")))

        # print(np.mean(data.ravel()))
        im = ax.imshow(data.T, origin="lower", vmin=0.5, vmax=1.1, extent=(-l, l, -l, l))

        ax.set_yticks(np.linspace(-l, l, 9, endpoint=True))
        ax.set_xticks(np.linspace(-l, l, 8, endpoint=False))
        if j == nS - 1:
            ax.set_xticks(np.linspace(-l, l, 9, endpoint=True))
        ax.spines['bottom'].set_color('grey')
        ax.spines['top'].set_color('grey')
        ax.spines['right'].set_color('grey')
        ax.spines['left'].set_color('grey')
        ax.tick_params(color='grey', direction="in", top=True, right=True)
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$x$")
        # ax.text(-0.96, 0.04, times_latex[x], color="white", fontsize=13)

    # Colorbar
    cbar = ax.cax.colorbar(im, label=r"$\rho$")
    cbar.outline.set_edgecolor('grey')
    cbar.ax.tick_params(color='gray', direction="in")
    ax.cax.toggle_label(True)

    plt.savefig(Path("ims", f"bomb.pdf"), dpi=200)

    # angluar momentum
    fig_j, ax_j = plt.subplots(figsize=(8, 3))
    for stepper_name in stepper_names:
        j = np.load(str(Path("ang", f"bomb_{stepper_name}_j.npy")))
        times = np.load(str(Path("ang", f"bomb_{stepper_name}_times.npy")))
        ax_j.plot(np.cumsum(times), np.abs(j / j[0] - 1), label=f"{stepper_name}")
    ax_j.set_xlabel(r"$t$")
    ax_j.set_ylabel(r"$|J(t)/J(0) - 1|$")
    ax_j.ticklabel_format(useOffset=False)

    custom_labels = [Line2D([0], [0], color="darkmagenta", lw=1),
                     Line2D([0], [0], color="mediumpurple", lw=1)]

    ax_j.semilogy()

    ax_j.legend(custom_labels, [r"$\mathrm{explicit}$", r"$\mathrm{implicit}$"])
    fig_j.tight_layout()
    plt.savefig(Path("ims", "bomb_ang_mom.pdf"), dpi=200)

