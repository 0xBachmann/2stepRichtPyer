import matplotlib.pyplot as plt
import numpy as np
from PDE_Types import Euler
from two_step_richtmeyer_util import Dimension, avg_x
from richtmeyer_two_step_scheme import Richtmeyer2step


DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)


L = 1
resolutions = np.array([[int(2**(i/2)), int(2**(i/2))] for i in range(4, 18)])
# resolutions = np.array([[50, 50]])
w0 = np.array([1, 1, 1])
w02d = np.array([1, 1, 0, 1])
waves = [F.waves(i, w0, amp=1e-5, alpha=1*np.pi/4) for i in range(3)]

print("resolution\tL2 norm of w-wref")
print("="*40)
for r in resolutions:
    print(np.product(r), end="\t\t")

    coords_x = np.linspace(0, L, r[0] + 1)
    coords_y = np.linspace(0, L, r[1] + 1)
    X, Y = np.meshgrid(avg_x(coords_x), avg_x(coords_y))
    grid = np.stack([X, Y][::-1], axis=-1)

    a = F.csnd(F.primitive_to_conserved(w02d))
    T = 1 / (w0[1] + a)

    for i in range(3):
        stepper = Richtmeyer2step(F, np.array([L, L]), r)
        stepper.initial_cond(waves[i])

        time = 0.
        while time < T:
            dt = stepper.cfl()
            if time + dt > T:
                dt = T - time
                time = T
            else:
                time += dt
            stepper.step(dt)

        ref = waves[i](grid, time)
        print((np.product(L / r) * np.sum((ref - stepper.grid_no_ghost)**2)), end="\t")
        # print(np.max(np.abs(ref - stepper.grid_no_ghost)), end="\t")

        # fig, ax = plt.subplots(2, 2)
        # for k in range(2):
        #     for j in range(2):
        #         im = ax[k][j].imshow((ref - stepper.grid_no_ghost)[..., k * 2 + j])
        #         fig.colorbar(im)
        # fig.suptitle(f"ref - stepper, time = {time}")
        #
        # fig, ax = plt.subplots(2, 2)
        # for k in range(2):
        #     for j in range(2):
        #         im = ax[k][j].imshow(ref[..., k * 2 + j])
        #         fig.colorbar(im)
        # fig.suptitle("ref")
        #
        # fig, ax = plt.subplots(2, 2)
        # for k in range(2):
        #     for j in range(2):
        #         im = ax[k][j].imshow(stepper.grid_no_ghost[..., k * 2 + j])
        #         fig.colorbar(im)
        # fig.suptitle("stepper")

    print("")

plt.show()


