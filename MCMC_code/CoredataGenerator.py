import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.integrate import solve_ivp
import sys

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

sys.path.append('/home/atridebchatterjee/reion_GPR/reion_21cm_code/Reionization_code') 

import running_reion
from running_reion import Redshift_Evolution

from warnings import filterwarnings
filterwarnings('ignore')
'''
Free_param_MAPPING_default={'epsilon_a0':0,
					       'epsilon_a1':1,
					       'epsilon_a2':2,
					       'epsilon_a3':3,
					       'epsilon_a4':4,
					       'epsilon_a5':5,
					       'lambda_0':6}

'''

Free_param_MAPPING_default={'epsilon_a0':0,
			'epsilon_a1':1,
			'epsilon_a2':2,
			'epsilon_a3':3,
			'lambda_0':4}


'''
You can add other free parameters in this dictionary containing
'''
zstart = 2.0
zend = 25.0
dz_step=0.2
dz=-0.2
n=int(abs(((zend-zstart)/dz_step)))+1

Z=np.linspace(zend,zstart,n)

'''
If you want to change range of Z. Please be careful, you also have to change it in start.py and running_reion.py (both of them are inside the folder Reionization_code)
'''

class CoreModule(object):
	"""
	Core Module for the delegation of the computation of the model products
	:param cosmodict: dict with planck cosmology
	:param GPR_mapping: dict mapping index of parameter vector to
		name used in Redshift_Evolution function of Reion modelling
	"""
	def __init__(self, cosmodict, Astro_dict, Free_param_mapping=Free_param_MAPPING_default):
		self.Free_param_mapping =  Free_param_mapping
		self.cosmodict=cosmodict
		self.Astro_dict=Astro_dict
		self.h=self.cosmodict['H0']/100.0  
	def __call__(self, ctx, single_run=False):
		p = ctx.getParams()
		try:
			Free_params = {}
			for k,v in self.Free_param_mapping.items():
				Free_params[k] = p[v]

			#build_interpolator pass it to Refshift_Evolution
			esc_PopII_redshift=np.array([3, 8, 13, 18]).reshape(-1, 1)
			#esc_Pop_II_val=np.array([Free_params['epsilon_a0'], Free_params['epsilon_a1'], Free_params['epsilon_a2'], Free_params['epsilon_a3'], Free_params['epsilon_a4'], Free_params['epsilon_a5']]).reshape(-1, 1)
			
			esc_Pop_II_val=np.array([Free_params['epsilon_a0'], Free_params['epsilon_a1'], Free_params['epsilon_a2'], Free_params['epsilon_a3']]).reshape(-1, 1)
			
			GPR_interpolator=self.gaussian_regress_process(esc_Pop_II_val, esc_PopII_redshift)
			#print(GPR_interpolator.predict(Z[0].reshape(1,-1), return_std=True))
			mean_prediction, std_prediction = GPR_interpolator.predict(Z.reshape(-1,1), return_std=True)

			if single_run:
				plt.scatter(esc_PopII_redshift, esc_Pop_II_val, label="Observations")
				plt.plot(Z, mean_prediction, label="Mean prediction")
				plt.legend()
				plt.show()

			if np.all(mean_prediction>0.0):
				Redshift, QHII, tau, dnlldz, gamma_PI, QHII5point8=Redshift_Evolution(self.cosmodict['H0'], self.cosmodict['ombh2'], self.cosmodict['omch2'], self.cosmodict['sigma8'], self.cosmodict['ns'], 
 self.Astro_dict['esc_pop_III'], GPR_interpolator, Free_params['lambda_0']).quntity_for_MCMC()


				ctx.add('Redshift', Redshift)
				ctx.add('QHII_array', QHII)
				ctx.add('opt_depth', tau)
				ctx.add('Q_HII_at_Z5.8', QHII5point8)
				ctx.add('lymanLimit', dnlldz)
				ctx.add('Gamma_PI', gamma_PI)

				return 1.0
			else:
				return 0.0

		except:
			raise  Exception("error either in assigning values in ctx or while computing some quantity from the model")


	def gaussian_regress_process(self, esc_Pop_II_array, esc_PopII_redshift, n_restarts_optimizer=9 ):
		kernel = 1 * RBF()
		gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
		gaussian_process.fit(esc_PopII_redshift, esc_Pop_II_array)
		return gaussian_process

'''
	def setup(self):
		#sanity check for known parameters
		#Redshift, QHII, tau, dnlldz, gamma_PI, QHII5point3=Redshift_Evolution(self.cosmodict['H0'], self.cosmodict['ombh2'], self.cosmodict['omch2'], self.cosmodict['ns'], self.cosmodict['sigma8'], GPR_params['epsilon_a0'], GPR_params['epsioln_a1'], GPR_params['epsioln_a2'], GPR_params['epsioln_a3'], GPR_params['epsioln_a4'], GPR_params['epsioln_a5']).quntity_for_MCMC()
'''
