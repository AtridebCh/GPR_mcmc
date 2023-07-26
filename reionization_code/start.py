import numpy as np
import scipy as sc

from scipy import integrate
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
import collections
import time
import math
import matplotlib.pyplot as plt
from array_descript import *
from utils import setspline_sigma
from constants import rho_c, yrbysec, e_QSO, Delta_H_overlap, Y_He, mp_by_mHe

from QSO_lumfun import lumfun_integral
from cosmological_functions import Cosmology
from Inhomo_reion import Inhomogenous_Reion
from SED_reader import get_stellar_SED, get_quasar_SED
from stellar_physics import *
from thermal import compute_species, Thermal
from cooling_heating_coeff import *

zstart = 2.0
zend = 25.0
dz= -0.2 #Note that we are going from redshift 25 to 2
redshift_bins=int(abs(((zend-zstart)/dz)))+1
n=redshift_bins
Z=np.linspace(zend,zstart,n)

'''
If you want to change range of Z. Please be careful, you also have to change it in running_reion.py (inside the folder Reionization_code)
'''

SEDFile_popII='/home/atridebchatterjee/reion_GPR/reion_21cm_code/SpecData/dndm_PopII_salpeter_Z0.004' #change this path to the appropriate path in your machine
SEDFile_popIII='/home/atridebchatterjee/reion_GPR/reion_21cm_code/SpecData/dndm_PopIII_salpeter_star' #change this path to the appropriate path in your machine


class Initialize(Inhomogenous_Reion, PopII_PopIII):

	def __init__(self, MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, MCMC_esc_PopII, MCMC_esc_PopIII, MCMC_lambda_0, **kwargs):
		
		if kwargs['initial_call']:
			self.cosmo_param_array=[MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8,  MCMC_ns]
			self.astrophysics_array=[MCMC_esc_PopII, MCMC_esc_PopIII, MCMC_lambda_0]

			self.MCMC_esc_PopII=MCMC_esc_PopII
			self.MCMC_esc_PopIII=MCMC_esc_PopIII
			self.MCMC_lambda_0=MCMC_lambda_0

			super(Initialize, self).__init__(*self.cosmo_param_array, call_for_first_time=True)

			self.call_for_first_time=False

			self.dtimedz=-(9.777e09*yrbysec)*super().func(Z[0])/self.h
			self.temp=0.0099*2.726*(1.0+Z[0])**2

			self.sigma_init=np.sqrt(super().sigmasq_b_sourav(0.0,Z[0],self.temp,1.0))

			self.norm=self.a
			
			self.delta_c=1.69/np.sqrt(self.norm) 
			self.rho_b=rho_c*self.omb*self.h**2
			self.rho_0=rho_c*self.h**2*self.omega_m
			self.logM, self.logsig=setspline_sigma(self.rho_0, self.sigma_sourav)
	
			self.tdyn=(super().Hubble_dist(Z[0])/self.h)*9.7776e09
			
			super().__init__(*self.cosmo_param_array, zcoll=Z[0], delta_c=self.delta_c , logM=self.logM, logsig=self.logsig, call_for_first_time=self.call_for_first_time, sigma=self.sigma_init, dtimedz=self.dtimedz, tdyn=self.tdyn, vcmax=1e09)
		else:
			self.call_for_first_time=False
			self.delta_c=kwargs['delta_c']
			self.sigma_z=kwargs['sigma']
			self.dtimedz_z=kwargs['dtimedz']	
			self.redshift=kwargs['current_redshift']
			super().__init__(*self.cosmo_param_array, zcoll=self.redshift,  logM=kwargs['logM'], logsig=kwargs['logsig'], delta_c=kwargs['delta_c'] , call_for_first_time=self.call_for_first_time, current_redshift=kwargs['current_redshift'], sigma=kwargs['sigma'], dtimedz=kwargs['dtimedz'], tdyn=kwargs['tdyn'], vcmax=1e09)


		lumfun_integral_qso[0]=e_QSO*lumfun_integral(Z[0])

	def collapse_frac(self):

		vcmin_pop2=vcmin_pop2_neut()
		vcmax=vc_max()

		dfcolldt['PopII']['ion']=0.0
		dfcolldt['PopII']['neut']=super().mass_integral_pop2(vcmin_pop2)
	
		dfcolldt['PopIII']['ion']=0.0
		dfcolldt['PopIII']['neut']=super().mass_integral_pop3(vcmin_pop3_neut())

		return dfcolldt




	def photon_count(self):

		dnphotdm['PopII']['HII'], dnphotdm['PopII']['HeII'], dnphotdm['PopII']['HeIII'], sigma_PI['PopII']['HII'], sigma_PI['PopII']['HeII'], sigma_PI['PopII']['HeIII'], sigma_PH['PopII']['HII'], sigma_PH['PopII']['HeII'], sigma_PH['PopII']['HeIII'] = get_stellar_SED(SEDFile_popII)

		dnphotdm['PopIII']['HII'], dnphotdm['PopIII']['HeII'], dnphotdm['PopIII']['HeIII'], sigma_PI['PopIII']['HII'], sigma_PI['PopIII']['HeII'], sigma_PI['PopIII']['HeIII'], sigma_PH['PopIII']['HII'], sigma_PH['PopIII']['HeII'], sigma_PH['PopIII']['HeIII'] = get_stellar_SED(SEDFile_popIII)

		sigma_PH['QSO']['HII'], sigma_PH['QSO']['HeII'], sigma_PH['QSO']['HeIII'], sigma_PI['QSO']['HII'], sigma_PI['QSO']['HeII'], sigma_PI['QSO']['HeIII'] = get_quasar_SED()


		return dnphotdm, sigma_PI, sigma_PH

	def IGM_region_initial(self):

		neutral_region['frac']['HI'][0]=1.0-Y_He-1.2e-05*np.sqrt(self.omega_m)/(self.h*self.omb)
		neutral_region['frac']['HeI'][0]=mp_by_mHe*Y_He
		neutral_region['frac']['HeIII'][0]=0.0
		neutral_region['T'][0]=0.0099*2.726*(1.0+Z[0])**2


	
		HII_region['frac']['HI'][0]=neutral_region['frac']['HI'][0]
		HII_region['frac']['HeI'][0]=neutral_region['frac']['HeI'][0]
		HII_region['frac']['HeIII'][0]=neutral_region['frac']['HeIII'][0]
		HII_region['T'][0]=neutral_region['T'][0]


		HeIII_region['frac']['HI'][0]=neutral_region['frac']['HI'][0]
		HeIII_region['frac']['HeI'][0]=neutral_region['frac']['HeI'][0]
		HeIII_region['frac']['HeIII'][0]=neutral_region['frac']['HeIII'][0]
		HeIII_region['T'][0]=neutral_region['T'][0]

		Global_region['frac']['HI'][0]=neutral_region['frac']['HI'][0]
		Global_region['frac']['HeI'][0]=neutral_region['frac']['HeI'][0]
		Global_region['frac']['HeIII'][0]=neutral_region['frac']['HeIII'][0]
		Global_region['T'][0]=neutral_region['T'][0]



		neutral_0['frac']['HI'][0]=neutral_region['frac']['HI'][0]
		neutral_0['frac']['HeI'][0]=neutral_region['frac']['HeI'][0]
		neutral_0['frac']['HeIII'][0]=neutral_region['frac']['HeIII'][0]
		neutral_0['T'][0]=neutral_region['T'][0]
	
		HII_0['frac']['HI'][0]=HII_region['frac']['HI'][0]
		HII_0['frac']['HeI'][0]=HII_region['frac']['HeI'][0]
		HII_0['frac']['HeIII'][0]=HII_region['frac']['HeIII'][0]
		HII_0['T'][0]=HII_region['T'][0]

		HeIII_0['frac']['HI'][0]=HeIII_region['frac']['HI'][0]
		HeIII_0['frac']['HeI'][0]=HeIII_region['frac']['HeI'][0]
		HeIII_0['frac']['HeIII'][0]=HeIII_region['frac']['HeIII'][0]
		HeIII_0['T'][0]=HeIII_region['T'][0]


		Global_0['frac']['HI'][0]=Global_region['frac']['HI'][0]
		Global_0['frac']['HeI'][0]=Global_region['frac']['HeI'][0]
		Global_0['frac']['HeIII'][0]=Global_region['frac']['HeIII'][0]
		Global_0['T'][0]=Global_region['T'][0]


		QH=Ionized_species['QH']
		QHe=Ionized_species['QHe']

		Sigma[0]=self.sigma_init

		Ionized_species['QH']['Q'][0]=1e-08        #last index is the index number of the redshift array i.e., redshift=25
		Ionized_species['QH']['Delta'][0]=Delta_H_overlap
		Ionized_species['QH']['F_V'][0]=super(Initialize, self).F_V(self.sigma_init, QH['Delta'][0])
		Ionized_species['QH']['F_M'][0]=super(Initialize, self).F_M(self.sigma_init, QH['Delta'][0])
		Ionized_species['QH']['R'][0]=super(Initialize, self).R(self.sigma_init, QH['Delta'][0])
		Ionized_species['QH']['mfp'][0]=1e-16
	
		

		Ionized_species['QHe']['Q'][0]=1e-10
		Ionized_species['QHe']['Delta'][0]=Delta_H_overlap
		Ionized_species['QHe']['F_V'][0]=super(Initialize, self).F_V(self.sigma_init, QHe['Delta'][0])
		Ionized_species['QHe']['F_M'][0]=super(Initialize, self).F_M(self.sigma_init, QHe['Delta'][0])
		Ionized_species['QHe']['R'][0]=super(Initialize, self).R(self.sigma_init, QHe['Delta'][0])
		Ionized_species['QHe']['mfp'][0]=0.0

		

		HII_region['X_HII'][0], HII_region['X_HeII'][0], HII_region['X_e'][0], HII_region['X'][0] = compute_species(HII_region['frac']['HI'][0], HII_region['frac']['HeI'][0], HII_region['frac']['HeIII'][0])

		HeIII_region['X_HII'][0], HeIII_region['X_HeII'][0], HeIII_region['X_e'][0], HeIII_region['X'][0] = compute_species(HeIII_region['frac']['HI'][0], HeIII_region['frac']['HeI'][0], HeIII_region['frac']['HeIII'][0])

		Global_region['X_HII'][0], Global_region['X_HeII'][0], Global_region['X_e'][0], Global_region['X'][0] = compute_species(Global_region['frac']['HI'][0], Global_region['frac']['HeI'][0], Global_region['frac']['HeIII'][0])


		HII_0['X_HII'][0], HII_0['X_HeII'][0], HII_0['X_e'][0], HII_0['X'][0] = compute_species(HII_0['frac']['HI'][0], HII_0['frac']['HeI'][0], HII_0['frac']['HeIII'][0])

		HeIII_0['X_HII'][0], HeIII_0['X_HeII'][0], HeIII_0['X_e'][0], HeIII_0['X'][0] = compute_species(HeIII_0['frac']['HI'][0], HeIII_0['frac']['HeI'][0], HeIII_0['frac']['HeIII'][0])

		Global_0['X_HII'][0], Global_0['X_HeII'][0], Global_0['X_e'][0], Global_0['X'][0] = compute_species(Global_0['frac']['HI'][0], Global_0['frac']['HeI'][0], Global_0['frac']['HeIII'][0])

		neutral_0['X_HII'][0], neutral_0['X_HeII'][0], neutral_0['X_e'][0], neutral_0['X'][0] = compute_species(neutral_0['frac']['HI'][0], neutral_0['frac']['HeI'][0], neutral_0['frac']['HeIII'][0])


		return neutral_region, Global_region, HII_region, HeIII_region, neutral_0, Global_0, HII_0, HeIII_0, Ionized_species, Gamma_PI, tau_elsc, lumfun_integral_qso



'''
	def set_escfrac(self):
		escfrac['PopII']['HII'], escfrac['PopII']['HeII'], escfrac['PopII']['HeIII']= self.MCMC_esc_PopII, self.MCMC_esc_PopII, self.MCMC_esc_PopII
		escfrac['PopIII']['HII'], escfrac['PopIII']['HeII'], escfrac['PopIII']['HeIII']= self.MCMC_esc_PopIII, self.MCMC_esc_PopIII, self.MCMC_esc_PopIII
		return escfrac
'''


