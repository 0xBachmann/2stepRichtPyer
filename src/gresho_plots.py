from PDE_Types import *
from pathlib import Path
from richtmyer_two_step_scheme import Richtmyer2step, Richtmyer2stepImplicit
from two_step_richtmyer_util import Dimension, log, avg_x
from intitial import gresho_vortex
import numpy as np
import time
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
from plotting_setup import *
mpl.rc('image', cmap='magma')

# cycler = plt.cycler(color=plt.cm.tab20c.colors)
colors = ["firebrick", "darkviolet", "teal", "darkkhaki"]
cycler = plt.cycler(color=colors) + plt.cycler(lw=[1, 1, 1, 1])  # + plt.cycler(markersize=[6, 3, 2, 1])
plt.rc('axes', prop_cycle=plt.cycler(linestyle=["solid", "dotted"]) * cycler)

DIM = Dimension.twoD
R = 40
resolution = np.array([R] * DIM.value)
Lx = 1
Ly = Lx
h = [Lx / resolution[0], Ly / resolution[1]]
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=0., hx=h[0], hy=h[1])

tol = 3e-16
domain = np.array([Lx, Ly])
steppers = [Richtmyer2step(F, domain, resolution, first_order=True),
            Richtmyer2step(F, domain, resolution),
            Richtmyer2stepImplicit(F, domain, resolution),
            Richtmyer2stepImplicit(F, domain, resolution),
            Richtmyer2stepImplicit(F, domain, resolution)]

stepper_names = {0: "first",
                 1: "expl",
                 2: "impl",
                 3: "impl_10",
                 4: "impl_100",
                 5: "impl_M"}
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
            if j != 0:
                continue
            energies = []
            times = []

            print(f"stepper: {i}, M = {M}")

            stepper.initial_cond(lambda x: gresho_vortex(x, center, F, M, qr=1))#0.4 * np.pi * Lx / 1.))
            stepper.step_for(0.01, fact=stepper_fact[i], callback=E_kin)
            np.save(str(Path("traj", f"gresho_vortex_{stepper_names[i]}_1e{-(j+1)}_t1.npy")),
                    F.mach(stepper.grid_no_ghost)[..., np.newaxis] / M)
            start = time.time()
            stepper.step_for(t - 0.01, fact=stepper_fact[i], callback=E_kin)
            end = time.time()
            if i == 1:
                print(f"M = {M:.5f} took {end - start}s")
            np.save(str(Path("traj", f"gresho_vortex_{stepper_names[i]}_1e{-(j+1)}_t2.npy")),
                    F.mach(stepper.grid_no_ghost)[..., np.newaxis] / M)

            np.save(str(Path("energy", f"en_{stepper_names[i]}_1e{-(j+1)}.npy")), np.array(energies))
            np.save(str(Path("energy", f"times_{stepper_names[i]}_1e{-(j+1)}.npy")), np.array(times))

generate_timings_and_energy = True
if generate_timings_and_energy:
    for j, M in enumerate(Ms):
        energies = []
        times = []

        stepper = Richtmyer2stepImplicit(F, domain, resolution, eps=11e-16)
        stepper.initial_cond(lambda x: gresho_vortex(x, center, F, M, qr=1))#0.4 * np.pi * Lx / 1.))
        start = time.time()
        stepper.step_for(1., fact=1./M, callback=E_kin)
        end = time.time()
        print(f"M = {M:.5f} took {end-start}s")
        np.save(str(Path("traj", f"gresho_vortex_impl_M-1_1e{-(j + 1)}.npy")),
                F.mach(stepper.grid_no_ghost)[..., np.newaxis] / M)

        np.save(str(Path("energy", f"en_impl_M-1_1e{-(j + 1)}.npy")), np.array(energies))
        np.save(str(Path("energy", f"times_impl_M-1_1e{-(j + 1)}.npy")), np.array(times))

    # for j, M in enumerate(Ms):
    #     energies = []
    #     times = []
    #
    #     stepper = Richtmyer2step(F, domain, resolution)
    #     stepper.initial_cond(lambda x: gresho_vortex(x, center, F, M, qr=1))#0.4 * np.pi * Lx / 1.))
    #     start = time.time()
    #     stepper.step_for(1., callback=E_kin)
    #     end = time.time()
    #     print(f"M = {M:.5f} took {end-start}s")
    #     np.save(str(Path("traj", f"gresho_vortex_expl_1e{-(j + 1)}.npy")),
    #             F.mach(stepper.grid_no_ghost)[..., np.newaxis] / M)
    #
    #     np.save(str(Path("energy", f"en_expl_1e{-(j + 1)}.npy")), np.array(energies))
    #     np.save(str(Path("energy", f"times_expl_1e{-(j + 1)}.npy")), np.array(times))


plot = True
if plot:
    # energies
    if False:
        fig_en, ax_en = plt.subplots(figsize=(8, 4.5))
        for i in range(len(steppers)):
            if i == 0:
                continue
            for j, M in enumerate(Ms):
                energies = np.load(str(Path("energy", f"en_{stepper_names[i]}_1e{-(j+1)}.npy")))
                times = np.load(str(Path("energy", f"times_{stepper_names[i]}_1e{-(j+1)}.npy")))
                ax_en.plot(np.cumsum(times), energies / energies[0], label=f"{stepper_names[i]}")
        ax_en.set_xlabel(r"$t$")
        ax_en.set_ylabel(r"$E_\mathrm{kin}(t)/E_\mathrm{kin}(0)$")
        ax_en.ticklabel_format(useOffset=False)

        # l = ax.legend([(p1, p2)], ['Two keys'], numpoints=1,
        #               handler_map={tuple: HandlerTuple(ndivide=None)})

        custom_labels = [Line2D([0], [0], color=colors[i], lw=1) for i in range(len(Ms))] + \
                        [Line2D([0], [0], color="grey", lw=1, linestyle=i) for i in ["solid", "dotted", "dashed", "dashdot"]]

        ax_en.legend(custom_labels, [fr"$M_\mathrm{{max}} = 10^{{{-i}}}$" for i in range(1, nM + 1)] + [*stepper_names_fancy.values()][1:], ncol=2)
        fig_en.tight_layout()
        plt.savefig(Path("ims", "energy_gresho_all.pdf"), dpi=200)
    if True:
        fig_en, ax_en = plt.subplots(figsize=(8, 4.5))
        for stepper_name in ["expl", "impl_M-1"]:
            for j, M in enumerate(Ms):
                energies = np.load(str(Path("energy", f"en_{stepper_name}_1e{-(j + 1)}.npy")))
                times = np.load(str(Path("energy", f"times_{stepper_name}_1e{-(j + 1)}.npy")))
                ax_en.plot(np.cumsum(times), energies / energies[0], label=f"{stepper_name}_{M}")
        ax_en.set_xlabel(r"$t$")
        ax_en.set_ylabel(r"$E_\mathrm{kin}(t)/E_\mathrm{kin}(0)$")
        ax_en.ticklabel_format(useOffset=False)

        custom_labels = [Line2D([0], [0], color=colors[i], lw=1) for i in range(len(Ms))] + \
                        [Line2D([0], [0], color="grey", lw=1, linestyle=i) for i in
                         ["solid", "dotted"]] + [Line2D([0], [0], color="white", lw=0)]

        ax_en.legend(custom_labels,
                     [fr"$M_\mathrm{{max}} = 10^{{{-i}}}$" for i in range(1, nM + 1)] + [r"$\mathrm{explicit}$",
                                                                                         r"$\mathrm{implicit, d}t\cdot\frac{1}{M_{\mathrm{max}}}$", ""], ncol=2)
        fig_en.tight_layout()
        plt.savefig(Path("ims", "energy_gresho_all_new.pdf"), dpi=200)

    # Mach number
    for i in range(len(steppers) + 1):
        fig = plt.figure(figsize=(3 * nM, 3))  # Notice the equal aspect ratio
        # axs_mach = [fig.add_subplot(1, nM, i + 1) for i in range(nM)]

        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(1, nM),
                         axes_pad=0.,
                         share_all=False,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.,
                         )

        for j, ax in enumerate(grid):

            if i == len(steppers):
                data = np.load(str(Path("traj", f"gresho_vortex_impl_M-1_1e{-(j + 1)}.npy")))
            else:
                data = np.load(str(Path("traj", f"gresho_vortex_{stepper_names[i]}_1e{-(j+1)}_t{1 if i == 0 else 2}.npy")))

            # print(np.mean(data.ravel()))
            im = ax.imshow(data.T[0], origin="lower", vmin=0, vmax=1, extent=(0, 1, 0, 1))

            ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))
            ax.set_xticks(np.linspace(0, 1, 5, endpoint=False))
            if j == nM - 1:
                ax.set_xticks(np.linspace(0, 1, 6, endpoint=True))
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

