import numpy as np
from src.PDE_Types import Euler
from src.two_step_richtmyer_util import Dimension, avg_x
from src.richtmyer_two_step_scheme import Richtmyer2step, Richtmyer2stepImplicit


DIM = Dimension.oneD
F = Euler(5. / 3, dim=DIM)


L = 1
w0 = np.array([1, 1, 1])
waves = [F.waves(i, w0, amp=1e-5) for i in range(3)]

print("resolution\tL2 norm of w-wref")
print("="*40)
for r in range(2, 17):
    r = 2 ** r
    print(r, end="\t\t")

    coords_x = np.linspace(0, L, r + 1)[..., np.newaxis]

    a = F.csnd(F.primitive_to_conserved(w0))
    T = [1 / (w0[1]), 1 / (w0[1] + a), 1 / (w0[1] - a)]

    for i in range(3):
        stepper = Richtmyer2step(F, np.array([L]), np.array([r]))
        stepper.initial_cond(lambda x: waves[i](x))

        time = 0.
        while time < T[i]:
            dt = stepper.cfl()
            dt = min(dt, T[i] - time)
            stepper.step(dt)
            time += dt

        print((L / r * np.sqrt(np.sum((waves[i](avg_x(coords_x), 0) - stepper.grid_no_ghost)**2))), end="\t")

    print("")


