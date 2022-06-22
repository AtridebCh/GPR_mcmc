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
		
		#self.a=self.norm_sourav(self.sigma_8)
		#print(self.h, self.omb, self.omega_m, self.omega_k, self.omega_l, self.sigma_8)

		self.cdict = {'omega_M_0' :self.omega_m, 'omega_lambda_0':self.omega_l, 'omega_b_0' : self.omb, 'omega_k_0' :self.omega_k, 'omega_n_0' : 0.0,'N_nu' :0.0, 'h' : self.h, 'n' : self.n_s,'sigma_8' :self.sigma_8 , 'baryonic_effects' :False}

		self.mean_dens = self.cdict["omega_M_0"] * cosmolopy.density.cosmo_densities(**self.cdict)[0] / self.cdict["h"]**2


	def initialize_power_spectrum(self,lnkmin, lnkmax, difflnk, tf='BBKS', mDM=0.):
		cdict=self.cdict
		lnk = np.arange(lnkmin, lnkmax, difflnk) 
		k = np.exp(lnk)  ## k in h/Mpc
		if tf=='EH':
			lnpk_unnorm = cdict["n"] * lnk + 2 * np.log(cosmolopy.perturbation.transfer_function_EH(k * cdict["h"], **cdict))[0] ## unnormalized power spectrum
		elif tf=='BBKS':
			q = np.exp(lnk) / (self.cdict["omega_M_0"] * cdict["h"]*cdict["h"])
			aa, bb, cc, powr = 6.4 , 3.0, 1.7, 1.13
			lnpk_unnorm = cdict["n"] * lnk - 2 * np.log((1 + (aa*q + (bb*q) ** 1.5 + (cc*q) ** 2) ** powr) ** (1.0/powr))
		if mDM > 0.:
        	### from Barkana, Haiman & Ostriker (2001) eqs (3) and (4)
        	### mDM is in keV
        	# gX = 1.5
        	# eps = 0.361
        	# eta = 5.0
        	# nuDM = 1.2
        	# mX = mDM
        	# h =  cdict["h"]
        	# OmegaX_hsq = cdict["omega_M_0"] * h ** 2
        	# Rc = 0.201 * (OmegaX_hsq / 0.15) ** 0.15 * (gX / 1.5) ** -0.29 * (mX / 1.0) ** -1.15 ### in Mpc
        	# Rc = Rc * h ### in Mpc / h because the k is in h/Mpc

        	### from Schneider et al (2011) eqs (4) and (5)
        	### mDM is in keV
			eta = 5.0
			nuDM = 1.12
			mX = mDM
			h =  cdict["h"]
			OmegaX = cdict["omega_M_0"]
			Rc = 0.049 * (OmegaX / 0.25) ** 0.11 * (h / 0.7) ** 1.22 * (mX / 1.0) ** -1.15 ### in Mpc / h

			lnpk_unnorm = lnpk_unnorm - 2 * (eta / nuDM) * np.log(1 + (k * Rc) ** (2 * nuDM))

		mean_dens = cdict["omega_M_0"] * cosmolopy.density.cosmo_densities(**cdict)[0] / cdict["h"]**2
		lnpk, normfac = self.normalize(lnpk_unnorm, lnk)
		print('normfac',normfac**2)

		return lnk, lnpk
    

    
	def normalize(self, lnpk_unnorm, lnk):

    	# Calculate the value of sigma_8 without prior normalization.
		unnorm_sigma_8 = self.sigma(4.*np.pi * 8 ** 3 * self.mean_dens / 3., lnpk_unnorm, lnk)

    	# Calculate the normalization factor
		normfac = self.sigma_8 / unnorm_sigma_8

    	# Normalize the previously calculated power spectrum.
		lnpk = 2 * np.log(normfac) + lnpk_unnorm

		return lnpk, normfac

	def sigma(self, M, lnpk, lnk, scheme='simps'):


		dlnk = lnk[1] - lnk[0]
		kcube_pk = np.exp(lnpk + 3 * lnk)
		integrand = kcube_pk * self.top_hat_window(M, lnk)
		if scheme == "trapz":
			print('here')
			sigmasq = (0.5 / np.pi ** 2) * sp.integrate.trapz(integrand, dx=dlnk)
		elif scheme == "simps":
			print('not here')
			sigmasq = (0.5 / np.pi ** 2) * sp.integrate.simps(integrand, dx=dlnk)
		elif scheme == 'romb':
			sigmasq = (0.5 / np.pi ** 2) * sp.integrate.romb(integrand, dx=dlnk)
		return np.sqrt(sigmasq)




	def top_hat_window(self,M, lnk):

    	## M is a scalar, lnk is an array
		kR = np.exp(lnk) * self.mass_to_radius(M)
    	# # The following 2 lines cut the integral at small scales to prevent numerical error.
		Wsq = np.ones(len(kR))
		kR = kR[kR > 1.4e-6]
		if len(kR) > 0:
			Wsq[-len(kR):] = (3 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)) ** 2
		return Wsq

	def mass_to_radius(self, M):
		return (3.*M / (4.*np.pi * self.mean_dens)) ** (1. / 3.)

	def radius_to_mass(self,R):
		return 4 * np.pi * R ** 3 * self.mean_dens / 3

instantiate=NORMALIZE(67.70, 0.0223, 0.12, 0.96,0.81)

lnkmin,lnkmax,difflnk=-20.0,20.0,0.1
lnk,lnpk=instantiate.initialize_power_spectrum(lnkmin, lnkmax, difflnk)
M_eight=instantiate.radius_to_mass(9.0)
print(instantiate.sigma(M_eight,lnpk,lnk)) # as we understand


