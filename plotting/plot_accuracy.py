import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams['text.usetex'] = True

file = "accuracy/euler2d_impl_conv_45deg.txt"
data = np.loadtxt(file, skiprows=2)

slopes = []
for i in range(3):
    slopes.append(np.polyfit(np.log(data[:, 0]), np.log(data[:, i + 1]), deg=1)[0])
    plt.plot(data[:, 0], data[:, i + 1], label=f"wave {i}: "r"$\|w -w_{{ref}}\|_{L^2}^2$")

xmax = np.max(data[:, 0])
xmin = np.min(data[:, 0])

ymax = np.max(data[:, 1:])
ymin = np.min(data[:, 1:])

slopes = np.unique([np.round(slope * 2, 0) / 2 for slope in slopes])


def elerp(a, b, gamma):
    return np.exp(np.log(a) * (1 - gamma) + np.log(b) * gamma)


assert slopes.size <= 2

for i, slope in enumerate(slopes):
    if i == 0:
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
        plt.text(elerp(xmin, xmax, offset),  elerp(ymin, ymax, offset + scale + 0.01), f"s = {slope}")
    else:
        offset = 0.2
        scale = 0.2
        x1 = elerp(xmin, xmax, offset)
        x2 = elerp(xmin, xmax, offset + scale)
        y2 = elerp(ymin, ymax, offset)
        y1 = np.exp(-slope * (np.log(x2) - np.log(x1)) + np.log(y2))
        p2 = Polygon(np.array([[x1, y1],
                               [x1, y2],
                               [x2, y2]]), fill=False)
        plt.gca().add_patch(p2)
        plt.text(elerp(xmin, xmax, offset),  elerp(ymin, ymax, offset - 0.02), f"s = {slope}")


plt.grid(True, ls="dashed")
plt.loglog()
plt.ylabel("error")
plt.xlabel("1/dx")
plt.legend()
plt.show()
