from PDE_Types import *
from pathlib import Path
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import Dimension, log, avg_x
from intitial import gresho_vortex
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
mpl.rc('image', cmap='magma')

plt.rcParams['text.usetex'] = True
# cycler = plt.cycler(color=plt.cm.tab20c.colors)
colors = ["firebrick", "darkviolet", "teal", "darkkhaki"]
cycler = plt.cycler(color=colors) + plt.cycler(lw=[1, 0.8, 0.6, 0.3]) + plt.cycler(markersize=[6, 3, 2, 1])
plt.rc('axes', prop_cycle=plt.cycler(linestyle=["solid", "dotted", "dashed", "dashdot"]) * cycler)

DIM = Dimension.twoD
R = 40
resolution = np.array([R] * DIM.value)
Lx = 1
Ly = Lx
h = [Lx / resolution[0], Ly / resolution[1]]
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=0., hx=h[0], hy=h[1])

domain = np.array([Lx, Ly])
steppers = [Richtmeyer2step(F, domain, resolution, first_order=True),
            Richtmeyer2step(F, domain, resolution),
            Richtmeyer2stepImplicit(F, np.array([Lx, Ly]), resolution),
            Richtmeyer2stepImplicit(F, np.array([Lx, Ly]), resolution),
            Richtmeyer2stepImplicit(F, np.array([Lx, Ly]), resolution)]

stepper_names = {0: "first",
                 1: "expl",
                 2: "impl",
                 3: "impl_10",
                 4: "impl_100"}
stepper_names_fancy = {0: "Rusanov",
                       1: "two step",
                       2: r"implicit",
                       3: r"implicit, $dt \cdot 10$",
                       4: r"implicit, $dt \cdot 100$"}

stepper_fact = [1, 1, 1, 10, 100]

Ms = [np.power(10., -i) for i in range(1, 5)]
nM = len(Ms)

center = np.array([Lx / 2, Ly / 2])

energies = []
times = []


def E_kin(stepper, dt):
    global energies, times
    v = stepper.grid_no_ghost
    energies.append(np.sum(0.5 * np.sum(v[..., 1:DIM.value + 1] ** 2, axis=-1) / v[..., 0]))
    times.append(dt)


t = 1.
generate = False
if generate:
    for i, stepper in enumerate(steppers):
        if i != 1:
            continue
        for j, M in enumerate(Ms):
            energies = []
            times = []

            print(f"stepper: {i}, M = {M}")

            stepper.initial_cond(lambda x: gresho_vortex(x, center, F, M, qr=0.4 * np.pi * Lx / 1.))
            stepper.step_for(0.01, fact=stepper_fact[i], callback=E_kin)
            np.save(str(Path("traj", f"gresho_vortex_{stepper_names[i]}_1e{-(j+1)}_t1.npy")),
                    F.mach(stepper.grid_no_ghost)[..., np.newaxis] / M)
            stepper.step_for(t - 0.01, fact=stepper_fact[i], callback=E_kin)
            np.save(str(Path("traj", f"gresho_vortex_{stepper_names[i]}_1e{-(j+1)}_t2.npy")),
                    F.mach(stepper.grid_no_ghost)[..., np.newaxis] / M)

            np.save(str(Path("energy", f"en_{stepper_names[i]}_1e{-(j+1)}.npy")), np.array(energies))
            np.save(str(Path("energy", f"times_{stepper_names[i]}_1e{-(j+1)}.npy")), np.array(times))

plot = True
if plot:
    # energies
    fig_en, ax_en = plt.subplots()
    for i in range(len(steppers)):
        if i == 0:
            continue
        for j, M in enumerate(Ms):
            energies = np.load(str(Path("energy", f"en_{stepper_names[i]}_1e{-(j+1)}.npy")))
            times = np.load(str(Path("energy", f"times_{stepper_names[i]}_1e{-(j+1)}.npy")))
            ax_en.plot(np.cumsum(times), energies / energies[0], label=f"{stepper_names[i]}")
    ax_en.set_xlabel(r"$t$")
    ax_en.set_ylabel(r"$E_\mathrm{kin}(t)/E_\mathrm{kin}(0)$")

    # l = ax.legend([(p1, p2)], ['Two keys'], numpoints=1,
    #               handler_map={tuple: HandlerTuple(ndivide=None)})

    custom_labels = [Line2D([0], [0], color=colors[i], lw=1) for i in range(len(Ms))] + \
                    [Line2D([0], [0], color="grey", lw=1, linestyle=i) for i in ["solid", "dotted", "dashed", "dashdot"]]

    ax_en.legend(custom_labels, [fr"$M_\mathrm{{max}} = 10^{{{-i}}}$" for i in range(1, nM + 1)] + [*stepper_names_fancy.values()][1:], ncol=2)
    fig_en.tight_layout()
    plt.savefig(Path("ims", "energy_gresho_all.pdf"), dpi=200)

    # Mach number
    for i in range(len(steppers)):

        fig = plt.figure(figsize=(3 * nM, 3))  # Notice the equal aspect ratio
        # axs_mach = [fig.add_subplot(1, nM, i + 1) for i in range(nM)]

        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(1, nM),
                         axes_pad=0.,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.,
                         )

        for j, ax in enumerate(grid):
            data = np.load(str(Path("traj", f"gresho_vortex_{stepper_names[i]}_1e{-(j+1)}_t{1 if i == 0 else 2}.npy")))
            # print(np.mean(data.ravel()))
            im = ax.imshow(data.T[0], origin="lower", vmin=0, vmax=1, extent=(0, 1, 0, 1))

            ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))
            ax.set_xticks(np.linspace(0, 1, 5, endpoint=False))
            # ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.0])
            # ax.set_xticklabels([0., 0.2, 0.4, 0.6, 0.8])
            ax.spines['bottom'].set_color('grey')
            ax.spines['top'].set_color('grey')
            ax.spines['right'].set_color('grey')
            ax.spines['left'].set_color('grey')
            ax.tick_params(color='grey', direction="in", top=True, right=True)
            ax.set_ylabel(r"$y$")
            ax.set_xlabel(r"$x$")
            ax.text(0.04, 0.04, fr"$M_\mathrm{{max}} = 10^{{{-(j+1)}}}$", color="white", fontsize=13)

        # Colorbar
        cbar = ax.cax.colorbar(im, label=r"$M/M_\mathrm{max}$")
        cbar.outline.set_edgecolor('grey')
        cbar.ax.tick_params(color='gray', direction="in")
        ax.cax.toggle_label(True)

        plt.savefig(Path("ims", f"mach_{stepper_names[i]}.pdf"), dpi=200)


