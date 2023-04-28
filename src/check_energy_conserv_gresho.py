from PDE_Types import Euler
from plotter import Plotter
from richtmeyer_two_step_scheme import Richtmeyer2step, Richtmeyer2stepImplicit
from two_step_richtmeyer_util import Dimension, log
from intitial import gresho_vortex

import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np

log("definition of variables")

DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)

log("calculate initial conditions")

resolution = np.array([40] * DIM.value)
Lx = 1
Ly = Lx
center = np.array([Lx / 2, Ly / 2])
stepper_impl = Richtmeyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-9, method="krylov")
stepper = Richtmeyer2step(F, np.array([Lx, Ly]), resolution)

M = 0.01
t = 0.5
stepper.initial_cond(lambda x: gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi * Lx / 1))
stepper_impl.initial_cond(lambda x: gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi * Lx / 1))


energies = []
energies_impl = []
times = []

energies.append(np.sum(stepper.grid_no_ghost[..., -1]))
energies_impl.append(np.sum(stepper_impl.grid_no_ghost[..., -1]))
times.append(0)

T = t
time = 0.
while time < T:
    dt = stepper.cfl()
    stepper.step(dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

    energies.append(np.sum(stepper.grid_no_ghost[..., -1]))
    energies_impl.append(np.sum(stepper_impl.grid_no_ghost[..., -1]))
    times.append(time)

plt.plot(times, energies, label="explicit")
plt.plot(times, energies_impl, label="implicit")
plt.xlabel("time")
plt.ylabel("Energy")
plt.legend()
plt.savefig(Path("ims", "energy_conserve_gresho.png"))

