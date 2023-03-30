import numpy as np
from PDE_Types import Euler
from two_step_richtmeyer_util import Dimension, avg_x
from plotter import Plotter


DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)


L = 1
resolutions = np.array([[int(2**(i/2)), int(2**(i/2))] for i in range(4, 18)])
w0 = np.array([1, 1, 1])
w02d = np.array([1, 1, 0, 1])
waves = [F.waves(i, w0, amp=1e-3, alpha=np.pi/4) for i in range(3)]

r = np.array([200, 200])
print(np.product(r), end="\t\t")

coords_x = np.linspace(0, L, r[0] + 1)
coords_y = np.linspace(0, L, r[1] + 1)
X, Y = np.meshgrid(avg_x(coords_x), avg_x(coords_y))

a = F.csnd(F.primitive_to_conserved(w02d))

T = 1 / (w0[1] + a)
dt = 0.001

wave = 0

plotter = Plotter(F, action="show", writeout=2, dim=DIM,
                  coords=[coords_x[:-1], coords_y[:-1]])

time = 0.
while time < T:
    ref = waves[wave](np.stack([X, Y], axis=-1), time)
    plotter.write(ref, dt)
    time += dt


plotter.finlaize()
