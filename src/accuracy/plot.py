import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams['text.usetex'] = True
# cycler = plt.cycler(color=plt.cm.Dark2.colors)
cycler = plt.cycler(color=["palevioletred", "darkmagenta", "mediumpurple"])
plt.rc('axes', prop_cycle=cycler * plt.cycler(lw=[2], marker="o"))

file = "euler2d_conv_45deg.txt"
data = np.loadtxt(file, skiprows=2)

xmax = np.max(data[:, 0])
xmin = np.min(data[:, 0])

ymax = np.max(data[:, 1:])
ymin = np.min(data[:, 1:])

slope = -2


def elerp(a, b, gamma):
    return np.exp(np.log(a) * (1 - gamma) + np.log(b) * gamma)


for i in range(3):
    plt.plot(data[:, 0], data[:, i + 1], label=f"wave k={i}")

offset = 0.7
scale = 0.2
x1 = elerp(xmin, xmax, offset)
x2 = elerp(xmin, xmax, offset + scale)
y1 = elerp(ymin, ymax, offset + scale)
y2 = np.exp(slope * (np.log(x2) - np.log(x1)) + np.log(y1))
p1 = Polygon(np.array([[x1, y1],
                       [x2, y1],
                       [x2, y2]]), fill=False)
plt.gca().add_patch(p1)
plt.text(elerp(xmin, xmax, offset), elerp(ymin, ymax, offset + scale + 0.01), f"s = {slope}")

plt.grid(True, ls="dashed")
plt.loglog()
plt.ylabel(f"$\|w -w_{{ref}}\|_{{L^2}}$")
plt.xlabel("$N_x$")
plt.legend()
plt.savefig(file.split(".")[0] + ".pdf")
plt.show()
