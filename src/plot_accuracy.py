import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

file = "accuracy/euler2d_conv_0deg.txt"
data = np.loadtxt(file, skiprows=2)

slope = np.polyfit(np.log(data[:, 0]), np.log(data[:, 1]), deg=1)[0]

plt.loglog(data[:, 0], data[:, 1], label=r"$\|w -w_{{ref}}\|_{L^2}$")
plt.ylabel("error")
plt.xlabel("1/dx")
plt.title(f"slope = {slope}")
plt.legend()
plt.show()
