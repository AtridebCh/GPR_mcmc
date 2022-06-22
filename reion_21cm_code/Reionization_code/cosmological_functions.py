import math
import numpy as np

from scipy import integrate
from scipy.integrate import quad

from normalization import NORMALIZE
from constants import mprot, kboltz

class Cosmology(NORMALIZE):

	def __init__(self, MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, **kwargs):
		super(Cosmology, self).__init__(MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, **kwargs)



	def Hubble_dist(self, z):
		return (math.sqrt(self.omega_k*(1.0+z)**2+self.omega_m*(1.0+z)**3+self.omega_r*(1.0+z)**4+self.omega_l))**-1


	def func(self, z):
		return self.Hubble_dist(z)/(1.0+z)


	def omega_z(self, z):	
		return self.Hubble_dist(z)**2*(1.0+z)**3*self.omega_m
	
	def d(self, z):  #this is equivalent to growth of cosmolopy

		if ((self.omega_l==0.0) and (self.omega_m==1.0)):
			dinv=1.0+z
		elif ((self.omega_l==0.0) and (self.omega_m!=1.0)):
			dinv=1.0+2.5*self.omega_m*z/(1.0+1.5*self.omega_m)
		else:
			dinv=(1.0+((1.0+z)**3-1.0)/(1.0+0.45450*(self.omega_l/self.omega_m)))**(1.0/3.0)
		return 1.0/dinv


	def sigmasq_b_sourav(self, x, z, T, mu):	
		return self.a*self.d(z)**2*quad(self.integrand_sigmasq_b, 0.0, np.inf, args=(x,T,z,mu),epsabs=1.49e-03, epsrel=1.49e-03)[0]/(2.0*math.pi**2)	

		
	def xbsq(self,T,z,mu):
		return (2.0*kboltz*T/(3.0*mu*mprot*self.omega_m*self.h**2*1e14))/(1.0+z)

		
	def pspec_b(self, k,T,z,mu):
		return self.pspec(k)/(1.0+self.xbsq(T,z,mu)*k**2)**2

	
	def integrand_sigmasq_b(self, k,x,T,z,mu):
		if (x<1e-07):
			return (1.0-(k*k*x*x/5.0))*k*k*self.pspec_b(k,T,z,mu)
		else:
			return (3.0*(math.sin(k*x)-k*x*math.cos(k*x)))**2*self.pspec_b(k,T,z,mu)/(k*k*x**3)**2
	
	def probdist(self, nu):
		return math.sqrt(2.0/math.pi)*np.exp(-nu**2/2.0)
