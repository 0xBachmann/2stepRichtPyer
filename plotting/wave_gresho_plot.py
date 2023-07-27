from mpl_toolkits.axes_grid1 import ImageGrid

from PDE_Types import Euler
from plotter import Plotter
from richtmyer_two_step_scheme import Richtmyer2step, Richtmyer2stepImplicit
from two_step_richtmyer_util import Dimension, log
from intitial import gresho_vortex, sound_wave_packet

from plotting_setup import *
from pathlib import Path

import numpy as np

log("definition of variables")

DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)

log("calculate initial conditions")

domain = np.array([[-10, 9], [0, 1]])
res_per_unit = 40
resolution = np.array([res_per_unit * (domain[0, 1] - domain[0, 0]), res_per_unit])
steppers = [Richtmyer2step(F, domain, resolution), Richtmyer2stepImplicit(F, domain, resolution)]
nS = len(steppers)

center = np.array([0.5, 0.5])

M = 1e-3
t = 1
stepper_names = ["expl", "impl"]

times = [0, 7.0e-4, 1.2e-2]

generate = False
if generate:
    fact = 1

    for j, stepper in enumerate(steppers):
        plotted = [False, False, False, False]

        stepper.initial_cond(lambda x: F.primitive_to_conserved(
            gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi, primitives=True)
            + sound_wave_packet(x, F, -0.5, Mmax=M, alpha=100, primitives=True, qr=0.4 * np.pi)))

        time = 0.
        while time < times[-1]:
            dt = stepper.cfl()
            stepper.step(dt)

            for i in range(len(times)):
                if time >= times[i] and not plotted[i]:
                    log10M = np.log10(F.mach(stepper.grid_no_ghost))[..., np.newaxis]
                    log10M[log10M < -4] = -4
                    np.save(str(Path("traj", f"wave_gresho_{stepper_names[j]}_{times[i]:.5f}.npy")),
                            log10M[(-1 - domain[0, 0]) * res_per_unit:-(domain[0, 1] - 1) * res_per_unit])
                    plotted[i] = True

            print(f"dt = {dt:.8g}, time = {time:.5g}/{times[-1]:.5g}")
            time += dt

            for i in range(len(times)):
                if time >= times[i] and not plotted[i]:
                    log10M = np.log10(F.mach(stepper.grid_no_ghost))[..., np.newaxis]
                    log10M[log10M < -4] = -4
                    np.save(str(Path("traj", f"wave_gresho_{stepper_names[j]}_{times[i]:.5f}.npy")),
                            log10M[(-1 - domain[0, 0]) * res_per_unit:-(domain[0, 1] - 1) * res_per_unit])
                    plotted[i] = True

plot = True
if plot:
    times_latex = [r"$t = 0.0$", r"$t = 7.0 \times 10^{-4}$", r"$t = 1.0$"]
    fig = plt.figure(figsize=(6 * 3, 5 * nS))  # Notice the equal aspect ratio
    # axs_mach = [fig.add_subplot(1, nM, i + 1) for i in range(nM)]

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(nS, 3),
                     axes_pad=0.,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.,
                     )

    for j, ax in enumerate(grid):
        y = j // 3
        x = j % 3
        data = np.load(str(Path("traj", f"wave_gresho_{stepper_names[y]}_{times[x]:.5f}.npy")))

        # print(np.mean(data.ravel()))
        im = ax.imshow(data.T[0], origin="lower", vmin=-4, vmax=-2, extent=(-1, 1, 0, 1))

        ax.set_yticks(np.linspace(0, 1, 5, endpoint=False))
        ax.set_xticks(np.linspace(-1, 1, 4, endpoint=False))
        if x == 2:
            ax.set_xticks(np.linspace(-1, 1, 5, endpoint=True))
        if y == 0:
            ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))
        ax.spines['bottom'].set_color('grey')
        ax.spines['top'].set_color('grey')
        ax.spines['right'].set_color('grey')
        ax.spines['left'].set_color('grey')
        ax.tick_params(color='grey', direction="in", top=True, right=True)
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$x$")
        ax.text(-0.96, 0.04, times_latex[x], color="white", fontsize=13)

    # Colorbar
    cbar = ax.cax.colorbar(im, label=r"$\log_{10} M$")
    cbar.outline.set_edgecolor('grey')
    cbar.ax.tick_params(color='gray', direction="in")
    ax.cax.toggle_label(True)

    plt.savefig(Path("ims", f"wave_gresho.pdf"), dpi=200)



