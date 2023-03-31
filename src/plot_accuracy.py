import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

file = "accuracy/euler2d_conv_45deg.txt"
data = np.loadtxt(file, skiprows=2)

for i in range(3):
    slope = np.polyfit(np.log(data[:, 0]), np.log(data[:, i + 1]), deg=1)[0]
    plt.loglog(data[:, 0], data[:, i + 1], label=f"wave {i}: "r"$\|w -w_{{ref}}\|_{L^2}^2$"f", slope={slope}")
plt.ylabel("error")
plt.xlabel("1/dx")
plt.legend()
plt.show()
