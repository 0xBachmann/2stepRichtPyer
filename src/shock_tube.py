from PDE_Types import *
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step
from two_step_richtmeyer_util import Dimension
from intitial import gresho_vortex

from RP1D_Euler import RP1D_Euler

import copy
import numpy as np

DIM = Dimension.twoD
n = 80
resolution = np.array([n] * DIM.value)
Lx = 1
Ly = Lx
h = [Lx / resolution[0], Ly / resolution[1]]
F = Euler(5. / 3, dim=DIM, c1=0.1, c2=0., hx=h[0], hy=h[1], add_viscosity=False)

stepper = Richtmeyer2step(F, np.array([Lx, Ly]), resolution, lerp=False)


tests = [(1.0, 0.75, 1.0, 0.125, 0.0, 0.1),
         (1.0, -2.0, 0.4, 1.0, 2.0, 0.4),
         (1.0, 0.0, 1000.0, 1.0, 0.0, 0.01),
         (5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950),
         (1.0, -19.59745, 1000.0, 1.0, -19.59745, 0.01),
         (1.4, 0.0, 1.0, 1.0, 0.0, 1.0),                           # works fine
         (1.4, 0.1, 1.0, 1.0, 0.1, 1.0)]

rhol, ul, pl, rhor, ur, pr = tests[0]
RP_exact = RP1D_Euler(5. / 3., rhol, ul, pl, rhor, ur, pr)

def test(x):
    quart = x.shape[0] // 4
    threq = 3 * quart

    result = np.empty((*x.shape[:-1], 4))

    result[:quart,      ..., 0] = rhol
    result[quart:threq, ..., 0] = rhor
    result[threq:,      ..., 0] = rhol

    result[:quart,      ..., 1] = ul
    result[quart:threq, ..., 1] = ur
    result[threq:,      ..., 1] = ul

    result[:,           ..., 2] = 0

    result[:quart,      ..., 3] = pl
    result[quart:threq, ..., 3] = pr
    result[threq:,      ..., 3] = pl

    return F.primitive_to_conserved(result)


M = 0.1
t = 1.
stepper.initial_cond(test)

coords = np.linspace(0, 1, resolution[0])
coords_exact = np.linspace(0, 1, 200)

plotter = Plotter(F, action="show", writeout=1, dim=Dimension.oneD, filename="sod_shock.mp4",
                  per_plot=2, coords=([coords_exact, coords]))

plot_visc = False

T = 0
def plot(stepper, dt, plot_mach=False, plot_curl=False, plot_eta=False):
    global T
    T += dt
    exact = RP_exact.sample(coords_exact, T)

    plotter.write((exact, stepper.grid_no_ghost[..., 1, :]), dt)


stepper.step_for(t, fact=1, callback=plot)

plotter.finalize()
