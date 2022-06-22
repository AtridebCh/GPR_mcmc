import math
import numpy as np
import scipy as sp
from scipy import integrate
from scipy.integrate import quad
import scipy.integrate as intg
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
import cosmolopy
import astropy.units as u
import matplotlib.pyplot as plt

#calculate all quantity at z=0.0
'''
M in h^{-1} M_solar
R in h^{-1} Mpc
k in hMPc^{-1}
rho in h^2 M_solar/Mpc^3
'''


class NORMALIZE:
	def __init__(self,MCMC_H0, MCMC_ombh2, MCMC_omch2, MCMC_ns, M_min, sigma_8):
		self.h=MCMC_H0/100.0
		self.omb=MCMC_ombh2/self.h**2
		self.omega_m = (MCMC_ombh2+MCMC_omch2)/self.h**2
		self.omgea_c=self.omega_m-self.omb
		self.omega_k = 0.0
		self.omega_l=1-self.omega_k-self.omega_m
		#self.As=MCMC_As
		self.n_s = MCMC_ns
		self.dn_dlnk= 0.0
		self.sigma_8=sigma_8
		self.Mmin=M_min #log10(m_min) 

		self.Mmax= 15
		self.dlog10m = 0.01
		self.M_array=self.h*10 ** np.arange(self.Mmin, self.Mmax, self.dlog10m) #in h^{-1}M_{solar}

		self.lnk_min, self.lnk_max, self.dlnk=-20.0,20.0,0.1

		self.Y_He = 0.2453

		self.number_density_hydrogen=2.0*10**-1*(MCMC_ombh2/0.022)*(1-self.Y_He) 

		self.cdict = {'omega_M_0' :self.omega_m, 'omega_lambda_0':self.omega_l, 'omega_b_0' : self.omb, 'omega_k_0' :self.omega_k, 'omega_n_0' : 0.0,'N_nu' :0.0, 'h' : self.h, 'n' : self.n_s,'sigma_8' :self.sigma_8 , 'baryonic_effects' :False}

		self.mean_dens = self.cdict["omega_M_0"] * cosmolopy.density.cosmo_densities(**self.cdict)[0] / self.cdict["h"]**2 # at z=0.0 in M_solar/Mpc^3
		self.lnk=np.arange(self.lnk_min, self.lnk_max, self.dlnk)
		self.k=np.exp(self.lnk)

		self.normfac=self.normalize()


	def dndm(self, z=0.0):
		return self.fsigma(z) * self.mean_dens * np.abs(self._dlnsdlnm()) / self.M_array** 2  #h^4/(M_solar*Mpc^3)


	def sigma_direct(self, R):
		M=self.radius_to_mass(R)
		integ=self.sigmasq_integrand(M)
		unn_sigma=np.sqrt(intg.simps(integ, dx=self.dlnk, axis=-1)/(2.0*np.pi**2))
		return self.normfac*unn_sigma

	def sigmasq_integrand(self, M):
		return self.unnormalised_power()*np.exp(3 * self.lnk)*self.top_hat_window(M)
	
	def normalize(self):
		M_eight=self.radius_to_mass(8.0)
		integ=self.sigmasq_integrand(M_eight)
		unn_sigma_8=np.sqrt(intg.simps(integ, dx=self.dlnk, axis=-1)/(2.0*np.pi**2))
		return self.sigma_8/unn_sigma_8


	'''def pk(self):
		q = self.k/ (self.cdict["omega_M_0"] * self.cdict["h"]*self.cdict["h"])
		aa, bb, cc, powr = 6.4 , 3.0, 1.7, 1.13
		lnpk = self.cdict["n"] * self.lnk - 2 * np.log((1 + (aa*q + (bb*q) ** 1.5 + (cc*q) ** 2) ** powr) ** (1.0/powr))
	return np.exp(lnpk)'''

	def top_hat_window(self,M):
		#kR = np.exp(lnk) * self.mass_to_radius(M)
		kR=np.outer(self.mass_to_radius(M), self.k)
		return (3 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)) ** 2


	def mass_to_radius(self, M):
		return (3.*M / (4.*np.pi * self.mean_dens)) ** (1. / 3.)

	def radius_to_mass(self,R):
		return 4 * np.pi * R ** 3 * self.mean_dens / 3


	def dw_dlnkr(self, kr):
		return np.where(
			kr > 1e-3,
			(9 * kr * np.cos(kr) + 3 * (kr ** 2 - 3) * np.sin(kr)) / kr ** 3,
			0,
			)

	def dlnss_dlnr(self, r):
		dlnk = np.log(self.k[1] / self.k[0])
		s = self.sigma_direct(r)
		rk = np.outer(r, self.k)

		rest = self.power0() * self.k ** 3
		w = self.top_hat_window(self.M_array)
		dw = self.dw_dlnkr(rk)
		integ = w * dw * rest	
		return intg.simps(integ, dx=dlnk, axis=-1) / (np.pi ** 2 * s ** 2)


	def fsigma(self, z):
		A = 0.3222
		a = 0.764
		p = 0.3
		nu=self.nu(self.radii(), z)
		return (
            A
            * np.sqrt(2.0 * a / np.pi)
            * nu
            * np.exp(-(a * nu**2) / 2.0)
            * (1 + (1.0 / (a *nu**2)) ** p)
        )



	def _dlnsdlnm(self):
		r"""
		The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln m}\right|`, ``len=len(m)``

		Notes
		-----

		.. math:: frac{d\ln\sigma}{d\ln m} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk

		"""
		return 0.5 * self.dlnss_dlnm(self.radii())




	def dlnss_dlnm(self, r):
		r"""
        The logarithmic slope of mass variance with mass.

        This is an important quantity, and is used directly to calculate
        :math:`\frac{dn}{dm}`.

        Parameters
        ----------
        r : array_like
            Radii.
		"""
		return self.dlnss_dlnr(r) * self.dlnr_dlnm()


	def dlnr_dlnm(self):
		r"""
        The derivative of log radius with log mass.

        For the usual :math:`m\propto r^3` mass assignment, this is just 1/3.

        Parameters
        ----------
        r : array_like
            Radii.
		"""
		return 1.0 / 3.0


	def radii(self):
		"""The radii corresponding to the masses `m`.

        Note that these are not the halo radii -- they are the radii containing mass
        m given a purely background density.
		"""
		return self.mass_to_radius(self.M_array)



	def unnormalised_lnT(self):
		r"""
		The un-normalised transfer function.
		"""
		a = 2.34
		b = 3.89
		c = 16.1
		d = 5.47
		e = 6.71

		Gamma = self.omega_m * self.h
		q = (
			np.exp(self.lnk)
            / Gamma
            * np.exp(
                self.omb
                + np.sqrt(2 * self.h) * self.omb/ self.omega_m
					)
			)
		return np.log(
            		(
                np.log(1.0 + a * q)
                / (a * q)
                * (1 + b * q + (c * q) ** 2 + (d * q) ** 3 + (e * q) ** 4) ** (-0.25)
            		)
        			)


	def unnormalised_power(self):
		r"""
        Un-normalised CDM power at :math:`z=0` [units :math:`Mpc^3/h^3`]
		"""
		return self.k ** self.n_s * np.exp(self.unnormalised_lnT()) ** 2


	def transfer_function(self):
		r"""Normalised CDM log transfer function."""
		return self.normfac * np.exp(self.unnormalised_lnT())


	def power0(self):
		r"""
        Normalised power spectrum at z=0 [units :math:`Mpc^3/h^3`]
		"""
		return self.normfac ** 2 * self.unnormalised_power()




	def ngtm_func(self, dndm, mass_density=False):
		size = len(dndm)
		m = self.M_array
		#ngtm = int_gtm(m[dndm > 0], dndm[dndm > 0], mass_density)
		#return self.hmf_integral_gtm(m[dndm > 0], dndm[dndm > 0], mass_density)[0][:size]
		#return  self.hmf_integral_gtm(m[dndm > 0], dndm[dndm > 0], mass_density)[1]

	def d(self, z):  #this is equivalent to growth of cosmolopy

		if ((self.omega_l==0.0) and (self.omega_m==1.0)):
			dinv=1.0+z
		elif ((self.omega_l==0.0) and (self.omega_m!=1.0)):
			dinv=1.0+2.5*self.omega_m*z/(1.0+1.5*self.omega_m)
		else:
			dinv=(1.0+((1.0+z)**3-1.0)/(1.0+0.45450*(self.omega_l/self.omega_m)))**(1.0/3.0)
		return 1.0/dinv



	def sigma_at_z(self, z):
		return self.sigma_direct(self.radii())*self.d(z)

	def nu(self, r, z, delta_c=1.686):
		r"""
        Peak height, :math:`\frac{\delta_c^2}{\sigma^2(r)}`.

        Parameters
        ----------
        r : array_like
            Radii

        delta_c : float, optional
            Critical overdensity for collapse.
		"""
		return (delta_c / self.sigma_at_z(z))



	def hmf_integral_gtm(self, M, dndm, mass_density=False):
		r"""
		Cumulatively integrate dn/dm.

		Parameters
		----------
		M : array_like
			Array of masses.
		dndm : array_like
        	Array of dn/dm (corresponding to M)
    	mass_density : bool, `False`
        	Whether to calculate mass density (or number density).

    	Returns
    	-------
    	ngtm : array_like
    	    Cumulative integral of dndm.

    	Examples
    	--------
    	Using a simple power-law mass function:

    	>>> import numpy as np
    	>>> m = np.logspace(10,18,500)
    	>>> dndm = m**-2
    	>>> ngtm = hmf_integral_gtm(m,dndm)
		>>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    		True

    	The function always integrates to m=1e18, and extrapolates with a spline
    	if data not provided:

    	>>> m = np.logspace(10,12,500)
    	>>> dndm = m**-2
    	>>> ngtm = hmf_integral_gtm(m,dndm)
    	>>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    		True

		"""
    	# Eliminate NaN's
		m = self.M_array[np.logical_not(np.isnan(dndm))]
		dndm = dndm[np.logical_not(np.isnan(dndm))]
		dndlnm = m * dndm

    	# Calculate the mass function (and its integral) from the highest M up to 10**18
		if m[-1] < m[0] * 10 ** 18 / m[3]:
			m_upper = np.arange(
            np.log(m[-1]), np.log(10 ** 18), np.log(m[1]) - np.log(m[0])
			)
			mf_func = _spline(np.log(m), np.log(dndlnm), k=1)
			mf = mf_func(m_upper)

			if not mass_density:
				int_upper = intg.simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even="first")
			else:
				int_upper = intg.simps(
				np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1], even="first"
				)
		else:
			int_upper = 0

    # Calculate the cumulative integral (backwards) of [m*]dndlnm when asked for mass density (instead of number density)
		if not mass_density:
			ngtm = np.concatenate(
				(
				intg.cumtrapz(dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[::-1],
				np.zeros(1),
				)
					)
			integ_simp=intg.simps(dndlnm, dx=np.log(m[1]) - np.log(m[0]), axis=-1)
		else:
				ngtm = np.concatenate(
				(
				intg.cumtrapz(m[::-1] * dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[
                    ::-1
				],
				np.zeros(1),
				)
				)
				integ_simp=intg.simps(m*dndlnm, dx=np.log(m[1]) - np.log(m[0]), axis=-1)
		return ngtm + int_upper, integ_simp
###################### Alternate Mass function ###################################

	def lam_eff_fs(self,mx):
		r"""
        Effective free-streaming scale.

        From Schneider+2013, Eq. 6
		"""
		g_x=1.5
		return (
            0.049
            * mx ** -1.11
            * (self.omgea_c / 0.25) ** 0.11
            * (self.h / 0.7) ** 1.22
            * (1.5 / g_x) ** 0.29
        )


	def m_fs(self, mx):
		r"""
        Free-streaming mass scale.

        From Schneider+2012, Eq. 7
		"""
		return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_eff_fs(mx) / 2) ** 3


	def lam_hm(self, mx):
		r"""
        Half-mode scale.

        From Schneider+2012, Eq. 8.
		"""
		mu=1.12
		return (
            2
            * np.pi
            * self.lam_eff_fs(mx)
            * (2 ** (mu / 5) - 1) ** (-0.5 / mu)
        )


	def m_hm(self, mx):
		r"""
        Half-mode mass scale.

        From Schneider+2013, Eq. 8
		"""
		return (4.0 / 3.0) * np.pi * self.mean_dens* (self.lam_hm(mx) / 2) ** 3


	def dndm_WDM_Lovell(self, mx, z=0.0):
		beta = 0.99
		gamma = 2.7
		return self.dndm(z) * (1 + gamma * self.m_hm(mx) / self.M_array) ** (-beta)

	def dndm_WDM_schnider(self,mx, z=0.0):
		alpha=0.6
		return self.dndm(z) * (1 + self.m_hm(mx) / self.M_array) ** (-alpha)

	def dndm_WDM_v_schnider(self, mx, z=0.0):
		beta=1.16
		return self.dndm(z) * (1 + self.m_hm(mx) / self.M_array) ** (-beta)


instantiate=NORMALIZE(67.70, 0.0223, 0.12, 0.96, 10.0, 0.81)

#print(instantiate.sigma_direct(np.array([8.0, 9.0, 10.0]))) # Radius in h^{-1}Mpc


array=instantiate.ngtm_func(instantiate.dndm())
print(array)
#print(instantiate.ngtm_func(instantiate.dndm()))
plt.xscale('log')
plt.yscale('log')
plt.plot(instantiate.M_array ,instantiate.dndm_WDM_schnider(0.1))
plt.plot(instantiate.M_array , instantiate.dndm())
plt.plot(instantiate.M_array , instantiate.dndm(z=2.0))
#plt.plot(instantiate.M_array, instantiate.ngtm_func(instantiate.dndm()))
plt.show()
