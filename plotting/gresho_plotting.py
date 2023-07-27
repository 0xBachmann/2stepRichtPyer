import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-muted")
plt.rcParams['text.usetex'] = True
mpl.rc('image', cmap='magma')

Ms = [np.power(10., -i) for i in range(1, 5)]
nM = len(Ms)

# Mach number
fig = plt.figure(figsize=(3 * nM, 3))

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
    data = np.load(f"traj/gresho_vortex_impl_1e{-(j+1)}_t2.npy")

    # print(np.mean(data.ravel()))
    im = ax.imshow(data.T[0], origin="lower", vmin=0, vmax=1, extent=(0, 1, 0, 1))
    ax.text(0.04, 0.04, fr"$M_\mathrm{{max}} = 10^{{{-(j+1)}}}$", color="white", fontsize=13)

    ax.spines['bottom'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['right'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.tick_params(color='grey', direction="in", top=True, right=True)

    ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))
    ax.set_xticks(np.linspace(0, 1, 5, endpoint=False))
    if j == nM - 1:
        ax.set_xticks(np.linspace(0, 1, 6, endpoint=True))
    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$x$")

# Colorbar
cbar = ax.cax.colorbar(im, label=r"$M/M_\mathrm{max}$")
cbar.outline.set_edgecolor('grey')
cbar.ax.tick_params(color='gray', direction="in")
ax.cax.toggle_label(True)

plt.savefig(f"ims/mach_impl_test.pdf", dpi=200)
