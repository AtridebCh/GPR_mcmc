import math
import numpy as np
import scipy as sp
from scipy import integrate
from scipy.integrate import quad
import cosmolopy

class NORMALIZE:
	def __init__(self,MCMC_H0, MCMC_ombh2, MCMC_omch2, MCMC_ns, sigma_8):
		self.h=MCMC_H0/100.0
		self.omb=MCMC_ombh2/self.h**2
		self.omega_m = (MCMC_ombh2+MCMC_omch2)/self.h**2
		self.omega_k = 0.0
		self.omega_l=1-self.omega_k-self.omega_m
		#self.As=MCMC_As
		self.n_s = MCMC_ns
		self.dn_dlnk= 0.0
		self.sigma_8=sigma_8
		
		self.Y_He = 0.2453

		self.number_density_hydrogen=2.0*10**-1*(MCMC_ombh2/0.022)*(1-self.Y_He) 

		self.cdict = {'omega_M_0' :self.omega_m, 'omega_lambda_0':self.omega_l, 'omega_b_0' : self.omb, 'omega_k_0' :self.omega_k, 'omega_n_0' : 0.0,'N_nu' :0.0, 'h' : self.h, 'n' : self.n_s,'sigma_8' :self.sigma_8 , 'baryonic_effects' :False}

		self.mean_dens = self.cdict["omega_M_0"] * cosmolopy.density.cosmo_densities(**self.cdict)[0] / self.cdict["h"]**2

		self.normfac=self.normalize()


	def sigmasq_integrand(self,lnk, M):
		return self.pk(lnk)*np.exp(3 * lnk)*self.top_hat_window(M, lnk)
	
	def normalize(self):
		M_eight=self.radius_to_mass(8.0)
		return self.sigma_8/math.sqrt(quad(self.sigmasq_integrand,-20,20,args=(M_eight,))[0]/(2.0*np.pi**2))

	def sigma_direct(self, M, normbool=False):
		return self.normfac*math.sqrt(quad(self.sigmasq_integrand,-20,20,args=(M,))[0]/(2.0*np.pi**2))

	def pk(self, lnk):
		q = np.exp(lnk) / (self.cdict["omega_M_0"] * self.cdict["h"]*self.cdict["h"])
		aa, bb, cc, powr = 6.4 , 3.0, 1.7, 1.13
		lnpk = self.cdict["n"] * lnk - 2 * np.log((1 + (aa*q + (bb*q) ** 1.5 + (cc*q) ** 2) ** powr) ** (1.0/powr))
		return np.exp(lnpk)


	def top_hat_window(self,M, lnk):
		kR = np.exp(lnk) * self.mass_to_radius(M)
		return (3 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)) ** 2


	def mass_to_radius(self, M):
		return (3.*M / (4.*np.pi * self.mean_dens)) ** (1. / 3.)

	def radius_to_mass(self,R):
		return 4 * np.pi * R ** 3 * self.mean_dens / 3

instantiate=NORMALIZE(67.70, 0.0223, 0.12, 0.96, 0.81)
#lnkmin,lnkmax,difflnk=-20.0,20.0,0.1
#lnk,lnpk=instantiate.initialize_power_spectrum(lnkmin, lnkmax, difflnk)
M_eight=instantiate.radius_to_mass(9.0)
print(instantiate.sigma_direct(M_eight)) # as we understand


