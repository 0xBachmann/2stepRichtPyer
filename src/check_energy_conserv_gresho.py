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

resolution = np.array([20] * DIM.value)
Lx = 1
Ly = Lx
center = np.array([Lx / 2, Ly / 2])
stepper = Richtmeyer2step(F, np.array([Lx, Ly]), resolution)

M = 0.001
t = 1
stepper.initial_cond(lambda x: gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi * Lx / 1))


def E(v, Ekin=False, Eint=False):
    if Ekin:
        return np.sum(0.5 * np.sum(v[..., 1:DIM.value + 1] ** 2, axis=-1) / v[..., 0])
    elif Eint:
        return np.sum(v[..., -1] - 0.5 * np.sum(v[..., 1:DIM.value + 1] ** 2, axis=-1) / v[..., 0])
    else:
        return np.sum(v[..., -1])


energies = []
times = []

fig, (ax1, ax2) = plt.subplots(1, 2)

energies.append([E(stepper.grid_no_ghost), E(stepper.grid_no_ghost, Ekin=True)])
times.append(0)

T = t
time = 0.
while time < T:
    dt = stepper.cfl()
    stepper.step(dt)

    print(f"dt = {dt}, time = {time:.3f}/{T}")
    time += dt

    energies.append([E(stepper.grid_no_ghost), E(stepper.grid_no_ghost, Ekin=True)])
    times.append(time)


energies = np.array(energies)

np.save(f"energy/en_expl.npy", energies)
np.save(f"energy/times_expl.npy", np.array(times))

ax1.plot(times, energies[:, 0], label="explicit")
ax2.plot(times, energies[:, 1], label="explicit")

stepper = Richtmeyer2stepImplicit(F, np.array([Lx, Ly]), resolution, eps=1e-16, method="krylov")

for i in [1, 2, 4, 8, 10, 15, 20, 30, 50, 100]:
    print(f"\n{i}")
    stepper.initial_cond(lambda x: gresho_vortex(x, center, F, Mmax=M, qr=0.4 * np.pi * Lx / 1))

    energies = []
    times = []

    energies.append([E(stepper.grid_no_ghost), E(stepper.grid_no_ghost, Ekin=True)])
    times.append(0)

    time = 0.
    while time < T:
        dt = stepper.cfl() * i
        if T - time < dt:
            dt = T - time
        stepper.step(dt)

        print(f"dt = {dt}, time = {time:.3f}/{T}")
        time += dt

        energies.append([E(stepper.grid_no_ghost), E(stepper.grid_no_ghost, Ekin=True)])
        times.append(time)

    energies = np.array(energies)
    
    np.save(f"energy/en_impl_{i}.npy", energies)
    np.save(f"energy/times_impl_{i}.npy", np.array(times))
    
    ax1.plot(times, energies[:, 0], label=f"implicit, {i}*cfl")
    ax2.plot(times, energies[:, 1], label=f"implicit, {i}*cfl")

print()
plt.xlabel("time")
plt.ylabel("Energy")
ax1.set_title("Total Energy")
ax2.set_title("Kinetic Energy")
ax1.legend()
plt.tight_layout()
plt.savefig(Path("ims", "energy_gresho_krylov_M001_real_eps.png"), dpi=200)
