import sys
import matplotlib.pyplot as plt



sys.path.append('/home/atrideb/Dropbox/CosmoReionMC/Reionization_code')
import running_reion
from running_reion import Redshift_Evolution

import time

start=time.time()
Redshift, QHII, tau, dnlldz, gamma_PI, QHII5point3=Redshift_Evolution(67.70, 0.0223, 0.12, 0.81, 0.96, 5.00, 0.0, 0.0036).quntity_for_MCMC()
print('time_required', time.time()-start)

plt.plot(Redshift[20:], dnlldz[20:])
plt.show()
