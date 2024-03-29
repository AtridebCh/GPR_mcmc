import sys
import os
import scipy as sp
#sys.path.append('/Data/atrideb/parameter_estimation_reionization21cm_pop2') #please change the '/home/atrideb' to the path of the folder where you keep the parent directory "parameter_estimation_reionization" parent directory


import numpy as np
import logging

from CoredataGenerator import CoreModule
from ChainContext import ChainContext
import matplotlib.pyplot as plt
from utils import getLogger, simple_plotting
from scipy import interpolate


class Likelihood(object):

	def __init__(self, CoreModule, Planck_dict, Astro_dict, min_param=None, max_param=None):
		"""
		Constructor for the likelihood chain

		:param min: array 
			lower bound for the parameters
		:param max: array
			upper bound for the parameters
        """
		self.min = min_param
		self.max = max_param
		self.cosmodict=Planck_dict
		self.Astro_dict=Astro_dict
		#self.EvoluteEscape=EvoluteEscape

	def isValid(self, p):
		"""
		checks if the given parameters are valid 
		"""
		if(self.min is not None):
			for i in range(len(p)):
				if (p[i]<self.min[i]):
					getLogger().debug("Params out of bounds i="+str(i)+" params "+str(p))
					return False
        
		if(self.max is not None):
			for i in range(len(p)):
				if (p[i]>self.max[i]):
					getLogger().debug("Params out of bounds i="+str(i)+" params "+str(p))
					return False
        
		return True



	def ComputeLikelihood(self, ctx, single_run=False):
		r"""
		call the native code to compute log likelihood
		:param ctx: an instance of a ChainContext
        
		:returns: the sum of the log likelihoods
		"""




		redshift=ctx.get('Redshift')
		Q_HII=ctx.get('QHII_array')
		tau=ctx.get('opt_depth')
		Q_HII_at_Z5point8=ctx.get('Q_HII_at_Z5.8')
		LL=ctx.get('lymanLimit')
		gamma_PI=ctx.get('Gamma_PI')

		gamma_PI_interpolate=interpolate.interp1d(redshift, gamma_PI, fill_value='extrapolate')
		gamma_model=gamma_PI_interpolate(self.redshiftGamma)
		
		LL_interpolate=interpolate.interp1d(redshift, LL, fill_value='extrapolate')
		LL_model=LL_interpolate(self.lymanRedshift)


		if single_run:
			plt.plot(redshift[20:], LL[20:])
			plt.xlabel('redshift(z)')
			plt.ylabel(r'$dnLL/dz$')
			plt.savefig('Q_HII.pdf')
			plt.show()

		if tau>0.01 and Q_HII_at_Z5point8>0.94:  #dark pixel fraction gives an upper limit of x_HI=0.06 hence QHII>(1-0.06)
			loglikeLymanSystem=-0.5 * np.sum((self.lymanLimitData - LL_model) ** 2 /self.lymanError**2)
			loglikeGamma=-0.5 * np.sum((self.Gamma_log - np.log10(gamma_model)) ** 2 /self.error_log**2)
			logliketau=-0.5 * (tau - 0.054)**2 / (0.007**2)
			loglike_tot=loglikeLymanSystem+loglikeGamma+logliketau
			return loglike_tot, tau, Q_HII_at_Z5point8
		else:
			return -np.inf, tau, Q_HII_at_Z5point8


	def __call__(self, p, single_run=False):
		getLogger().debug("pid: %s, processing: %s"%(os.getpid(), p))
		ctx = self.createChainContext(p)

		if np.all(p>0.0) and np.all(p[:-1]<1.0) :	
			model=self.Core(ctx, single_run=single_run)
			if model:
				likelihood, tau, Q_HII_at_Z5point8=self.ComputeLikelihood(ctx, single_run=single_run) #blobs feature is used and you have to use array for this
				blobs=np.array([tau, Q_HII_at_Z5point8])
				return likelihood, blobs
			else:
				return -np.inf, [1.0, np.nan]
		else:
			return -np.inf, [1.0, np.nan]  #if len(blobs)>1 then use [1.0, 1.0]
		
			

	def createChainContext(self, p):
		"""
		Returns a new instance of a chain context 
		"""
		try:
			p = Params(*zip(self.params.keys, p))
		except Exception:
            # no params or params has no keys
			pass
		return ChainContext(self, p)


	def setup(self, gammadatafile, Lymanlimitdatafile, single_run=False):
		self.redshiftGamma, Gamma, Gamma_max, Gamma_min=np.loadtxt(gammadatafile, usecols=(0,1,2,3), unpack=True)        
		self.Gamma_log=np.log10(Gamma) 
		error_up=Gamma_max-Gamma
		self.error_log=error_up/Gamma
		self.lymanRedshift,self.lymanLimitData,self.lymanError=np.loadtxt(Lymanlimitdatafile, usecols=(0,1,2), unpack=True)
		self.Core=CoreModule(self.cosmodict, self.Astro_dict)


        
