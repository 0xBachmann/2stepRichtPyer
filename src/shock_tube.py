from PDE_Types import *
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step
from two_step_richtmeyer_util import Dimension
from intitial import gresho_vortex

from RP1D_Euler import RP1D_Euler

import copy
import numpy as np

DIM = Dimension.twoD
n = 320
resolution = np.array([n, 80])
Lx = 1
Ly = Lx
h = [Lx / resolution[0], Ly / resolution[1]]
F = Euler(5. / 3, dim=DIM, c1=1., c2=1., hx=h[0], hy=h[1], add_viscosity=True)
F_novisc = Euler(5. / 3, dim=DIM, c1=1., c2=1., hx=h[0], hy=h[1], add_viscosity=False)

stepper_vanilla = Richtmeyer2step(F_novisc, np.array([Lx, Ly]), resolution, lerp=-1)
stepper_visc = Richtmeyer2step(F, np.array([Lx, Ly]), resolution, lerp=-1)
stepper_lerp = Richtmeyer2step(F_novisc, np.array([Lx, Ly]), resolution, lerp=3)
stepper_lerp_visc = Richtmeyer2step(F, np.array([Lx, Ly]), resolution, lerp=3, order1=False)

tests = [(1.0, 0.75, 1.0, 0.125, 0.0, 0.1),  # rusanov works
         (1.0, -2.0, 0.4, 1.0, 2.0, 0.4),  # not even with visc
         (1.0, 0.0, 1000.0, 1.0, 0.0, 0.01),  # also not too bad
         (5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950),  # also okayish
         (1.0, -19.59745, 1000.0, 1.0, -19.59745, 0.01),  # not even with visc
         (1.4, 0.0, 1.0, 1.0, 0.0, 1.0),  # works fine
         (1.4, 0.1, 1.0, 1.0, 0.1, 1.0)]  # densitiy oscilates

# TODO: not working well: 1, 3, 4, 5 (as pressure = const -> eta = 1 everywhere)
#           working well: 0, 2, 6 (also refine filtering of eta)
# detection eta:
# test nr   0: right side weird
#           1:
#           2: good, small osz
#           3:
#           4:
#           5: needs filter, p = const -> eta = 1
#           6: need more filter, shouldn't wash out

# st eta:
# test nr   0: nah, better with higher eta, * 10
#           1: pressure gets too low
#           2: eta bit too small?
#           3: only left side wrong
#           4: osz a bit high, right side more
#           5: good
#           6: small osz
which_test = 6

rhol, ul, pl, rhor, ur, pr = tests[which_test]
RP_exact_l = RP1D_Euler(5. / 3., rhol, ul, pl, rhor, ur, pr, xdiaph=0.25)
RP_exact_r = RP1D_Euler(5. / 3., rhor, ur, pr, rhol, ul, pl, xdiaph=0.75)

scalrho = 1
scalu = 1
scalp = 1
lims = {0: (min(rhol, rhor) - scalrho, max(rhol, rhor) + scalrho),
        1: (min(ul, ur) - scalu, max(ul, ur) + scalu),
        2: (min(pl, pr) - scalp, max(pl, pr) + scalp),
        3: (0, 1),
        4: (-100, 100)
        }


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


M = 0.1
t = 1.
stepper_vanilla.initial_cond(test)
stepper_lerp.initial_cond(test)
stepper_visc.initial_cond(test)
stepper_lerp_visc.initial_cond(test)

coords = np.linspace(0, 1, resolution[0])
coords_exact = np.linspace(0, 1, 400)


msz = 8
plot_args = [{"marker": "1", "markersize": msz, "linewidth": 0, "label": "vanilla"},
             {"marker": "2", "markersize": msz, "linewidth": 0, "label": "eta order 1"},
             {"marker": "3", "markersize": msz, "linewidth": 0, "label": "with visc"},
             {"marker": "4", "markersize": msz, "linewidth": 0, "label": "eta with visc"},
             {"linestyle": "solid", "label": "exact"}]

plotter = Plotter(5, action="show", writeout=1, dim=Dimension.oneD, filename="sod_shock.mp4",
                  per_plot=[5, 5, 5, 4, 4], coords=[[coords, coords, coords, coords, coords_exact]],
                  comp_names=["density", "vel-x", "pressure", "eta", "visc"],
                  plot_args=plot_args, lims=lims, interval=200, drop_last=1)

plot_visc = False


def plot(dt):
    half = coords_exact.shape[0] // 2
    exact_l = np.stack(RP_exact_l.sample(coords_exact[:half], time), axis=-1)
    exact_r = np.stack(RP_exact_r.sample(coords_exact[half:], time), axis=-1)
    exact = np.concatenate([exact_l, exact_r])

    plotter.write((np.column_stack(
        [F.conserved_to_primitive(stepper_vanilla.grid_no_ghost)[..., 0, (0, 1, 3)],
         F.eta(((stepper_vanilla.grid)), None, h[0], h[1], 3)[..., 0],
         F.viscosity2(stepper_vanilla.grid_no_ghost, avg_x(avg_y(stepper_vanilla.grid_no_ghost)))[..., 0, 0, 0]]),
                   np.column_stack(
                       [F.conserved_to_primitive(stepper_lerp.grid_no_ghost)[..., 0, (0, 1, 3)],
                        F.eta(((stepper_lerp.grid)), None, h[0], h[1], 3)[..., 0],
                        F.viscosity2(stepper_lerp.grid_no_ghost, avg_x(avg_y(stepper_lerp.grid_no_ghost)))[
                            ..., 0, 0, 0]]),
                   np.column_stack(
                       [F.conserved_to_primitive(stepper_visc.grid_no_ghost)[..., 0, (0, 1, 3)],
                        F.eta(((stepper_visc.grid)), None, h[0], h[1], 3)[..., 0],
                        F.viscosity2(stepper_visc.grid_no_ghost, avg_x(avg_y(stepper_visc.grid_no_ghost)))[
                            ..., 0, 0, 0]]),
                   np.column_stack(
                       [F.conserved_to_primitive(stepper_lerp_visc.grid_no_ghost)[..., 0, (0, 1, 3)],
                        F.eta(((stepper_lerp_visc.grid)), None, h[0], h[1], 3)[..., 0],
                        F.viscosity2(stepper_lerp_visc.grid_no_ghost, avg_x(avg_y(stepper_lerp_visc.grid_no_ghost)))[
                            ..., 0, 0, 0]]),
                   exact), dt)


time = 0.
plot(0)
T = 0.05
while time < T:
    cfls = [stepper_vanilla.cfl(), stepper_lerp.cfl(), stepper_visc.cfl(), stepper_lerp_visc.cfl()]
    dt = np.min(cfls)
    # dt = 0.0001

    if np.any(np.isnan([stepper_vanilla.cfl(), stepper_lerp.cfl(), stepper_visc.cfl(), stepper_lerp_visc.cfl()])):
        print(cfls)

    # stepper_vanilla.step(dt)
    stepper_lerp.step(dt)
    # stepper_visc.step(dt)
    # stepper_lerp_visc.step(dt)

    plot(dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

plotter.finalize()
