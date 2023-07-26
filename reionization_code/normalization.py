import math
import numpy as np
from scipy import integrate
from scipy.integrate import quad
import cosmolopy
from constants import rho_c

class NORMALIZE:

	def __init__(self,MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, **kwargs):

		self.h=MCMC_H0/100.0
		self.omb=MCMC_ombh2/self.h**2
		self.omega_m = (MCMC_ombh2+MCMC_omch2)/self.h**2
		self.omega_k = 0.0
		self.omega_r=0.0
		self.omega_l=1-self.omega_k-self.omega_m
		#self.As=MCMC_As
		self.n_s = MCMC_ns
		self.dn_dlnk= 0.0
		self.sigma_8=sigma_8
		
		self.Y_He = 0.2453
		self.rho_0=rho_c*self.h**2*self.omega_m

		self.number_density_hydrogen=2.0*10**-1*(MCMC_ombh2/0.022)*(1-self.Y_He) 
		self.cdict= {'omega_M_0' :self.omega_m, 'omega_lambda_0':self.omega_l, 'omega_b_0' : self.omb, 'omega_k_0' :self.omega_k, 'omega_n_0' : 0.0,'N_nu' :0.0, 'h' : self.h, 'n' : self.n_s,'sigma_8' :self.sigma_8 , 'baryonic_effects' :False}
		
		#super(NORMALIZE, self).__init__(**kwargs)
		
		self.a=self.norm_sourav(self.sigma_8)

	################################################## normalize ######################################

	def transfun(self, q):
		aa=6.40
		bb=3.0
		cc=1.70
		power=1.13
		return 1.0/((1.0+(aa*q +(bb*q)**1.50 +(cc*q)**2)**power)**(1.0/power))

	
	def pspec(self, k):
		gamma=self.omega_m*self.h
		gam_h=gamma*self.h
		K0=0.05
		n0=self.n_s+0.5*self.dn_dlnk*math.log(k/K0)
		return (k**n0)*self.transfun(k/gam_h)*self.transfun(k/gam_h)


	def pspec_EH(self,k):
		cdict=self.cdict
		K0=0.05
		n0=self.n_s+0.5*self.dn_dlnk*math.log(k/K0)
		return  (k**n0)*(cosmolopy.perturbation.transfer_function_EH(k * cdict["h"], **cdict)[0])**2
		
	def window(self, x):
		return (3.0*((math.sin(x)/x**3)-(math.cos(x)/x**2)))**2


	def pswin(self, k):
		r=8.0/self.h
		return self.pspec(k)*k**2*self.window(k*r)

	
	def norm_sourav(self, sigma):
		return sigma**2*2*math.pi**2/quad(self.pswin,0.0,np.inf, epsabs=1.49e-09, epsrel=1.49e-03)[0]


	def sigma_sourav(self, x):  #x=r is in Mpc and NOT in h*mpc^{-1}
		return math.sqrt(quad(self.sigmasq_integrand,0.0,np.inf,args=(x,),epsabs=1.49e-09, epsrel=1.49e-05)[0]/(2.0*np.pi**2))
	
	def sigmasq_integrand(self,k,r):
		return self.pspec(k)*k**2*self.window(k*r)


#instantiate=NORMALIZE(67.70, 0.0223, 0.12, 0.81, 0.96)
#print(instantiate.sigma_8)



