import numpy as np
from PDE_Types import Euler
from two_step_richtmeyer_util import Dimension, avg_x
from plotter import Plotter


DIM = Dimension.twoD
F = Euler(5. / 3, dim=DIM)


L = 1
w0 = np.array([1, 1, 1])
waves = [F.waves(i, w0, amp=1e-3, alpha=np.pi/4) for i in range(3)]

r = np.array([50 for i in range(DIM.value)])

coords = [np.linspace(0, L, r[i] + 1) for i in range(DIM.value)]
XY = np.meshgrid(*[avg_x(coord) for coord in coords])
grid = np.stack(XY, axis=-1)

T = 1
dt = 0.005

wave = 1

plotter = Plotter(F, action="show", writeout=1, dim=DIM,
                  coords=[coord[:-1] for coord in coords], filename=f"wave{wave}.mp4")

time = 0.
while time < T:
    ref = waves[wave](grid, time)
    plotter.write(ref, dt)
    time += dt


plotter.finalize()
