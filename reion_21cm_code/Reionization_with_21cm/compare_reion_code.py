import numpy as np
import matplotlib.pyplot as plt
from experimenting_numba import running_code

parameters=np.array([67.70, 0.0223, 0.12, 0.81, 0.96, 0.0036, 0.0, 5.0])

reion=running_code(*parameters)

plt.plot(reion[0][20:], reion[4][20:])
plt.show()
