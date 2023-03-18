import numpy as np

from two_step_richtmeyer_util import *
from PDE_Types import *
from plotter import *
from richtmeyer_two_step_scheme import Richtmeyer2step


DIM = Dimension.oneD
F = Euler(5. / 3, dim=DIM)


L = 1
resolutions = np.array([10, 100, 1000, 10000, 100000])
w0 = np.array([1, 1, 1])
wave = F.waves(0, w0, amp=1e-3)

print("resolution\tL2 norm of w-wref")
print("="*40)
for r in range(2, 17):
    r = 2 ** r
    print(r, end="\t\t")

    coords_x = np.linspace(0, L, r + 1)

    stepper = Richtmeyer2step(F, np.array([L]), np.array([r]))
    stepper.initial_cond(lambda x: wave(x, 0))

    a = F.csnd(w0)

    T = 1 / (w0[1] + a)
    time = 0.
    while time < T:
        dt = stepper.cfl()
        stepper.step(dt)
        time += dt

    print(L / r * np.sum((wave(coords_x[:-1], time) - stepper.grid_no_ghost)**2))


