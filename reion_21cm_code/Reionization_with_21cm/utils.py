import numpy as np
import matplotlib.pyplot as plt
import  normalization as norm
#from normalization import NORMALIZE

def setspline_sigma(rho_0, sigma_sourav): #sigma_sourav is a method in NORMALIZE
	len_R=100
	logM=np.linspace(0.2,20.0,int(20.0/0.2))
	Rcube=3.0*(10**logM)/(4.0*np.pi*rho_0)
	R=Rcube**(1/3.0)
	logsig=np.zeros(len_R)
	vfunc = np.vectorize(sigma_sourav)
	logsig=np.log10(vfunc(R))
	return logM,logsig

