import numpy as np
from PDE_Types import Euler
from two_step_richtmeyer_util import Dimension, avg_x
from richtmeyer_two_step_scheme import Richtmeyer2step


DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)


L = 1
resolutions = np.array([[int(2**(i/2)), int(2**(i/2))] for i in range(4, 18)])
w0 = np.array([1, 1, 1])
w02d = np.array([1, 1, 0, 1])
waves = [F.waves(i, w0, amp=1e-3, alpha=0) for i in range(3)]

print("resolution\tL2 norm of w-wref")
print("="*40)
for r in resolutions:
    print(np.product(r), end="\t\t")

    coords_x = np.linspace(0, L, r[0] + 1)
    coords_y = np.linspace(0, L, r[1] + 1)
    X, Y = np.meshgrid(avg_x(coords_x), avg_x(coords_y))
    grid = np.stack([X, Y], axis=-1)

    a = F.csnd(F.primitive_to_conserved(w02d))
    T = 1 / (w0[1] + a)

    for i in range(3):
        stepper = Richtmeyer2step(F, np.array([L, L]), r)
        stepper.initial_cond(waves[i])

        time = 0.
        while time < T:
            dt = stepper.cfl()
            stepper.step(dt)
            time += dt

        ref = waves[i](grid, time)
        print((np.product(L / r) * np.sum((ref - stepper.grid_no_ghost)**2)), end="\t")
        # print(np.max(np.abs(ref - stepper.grid_no_ghost)), end="\t")

    print("")


