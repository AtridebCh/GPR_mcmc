import numpy as np
import scipy as sc
import math
import pandas as pd

from constants import *

from array_descript import *
from utils import setspline_sigma
from thermal import compute_species, Thermal
from cooling_heating_coeff import *
import matplotlib.pyplot as plt
from QSO_lumfun import *



zstart = 2.0
zend = 25.0
dz_step=0.2
dz=-0.2
n=int(abs(((zend-zstart)/dz_step)))+1

Z=np.linspace(zend,zstart,n)


from start import Initialize

class Redshift_Evolution(Initialize):

	def __init__(self, MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, MCMC_esc_PopIII, GPR_interpolator_PopII, MCMC_lambda_0): # GPR_interpolator_PopII
		
		
		super(Redshift_Evolution, self).__init__(MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, GPR_interpolator_PopII, MCMC_esc_PopIII, MCMC_lambda_0, initial_call=True, zcoll=Z[0]) # GPR_interpolator_PopII


		dfcolldt=super().collapse_frac()
		dnphotdm, sigma_PI, sigma_PH=super().photon_count()

		neutral_region, Global_region, HII_region, HeIII_region, neutral_0, Global_0, HII_0, HeIII_0, Ionized_species, Gamma_PI, tau_elsc, lumfun_integral_qso=super().IGM_region_initial()
		#escfrac=super().set_escfrac()
		delta_c=1.69/np.sqrt(self.norm) 

		self.Q_HII=np.zeros(n)
		self.Q_HII[0]=Ionized_species['QH']['Q'][0]
		self.gamma_PI=np.zeros(n)
		self.gamma_PI[0]=(Gamma_PI['PopII']['HII'][0]+Gamma_PI['PopIII']['HII'][0]+Gamma_PI['QSO']['HII'][0])/10**-12
		self.dNLLdz_array=np.zeros(n)
		self.dNLLdz_array[0]=dNLLdz[0]
		self.redshift_array=np.zeros(n)
		self.redshift_array[0]=zend
		self.PopII_sfr=np.zeros(n)
		self.PopII_sfr[0]=0.0
		self.PopIII_sfr=np.zeros(n)
		self.PopIII_sfr[0]=0.0
		self.GPR_initerpolator=GPR_interpolator_PopII
		self.x_HI_array=np.zeros(n)
		#self.use_21_cm=use_21_cm



		for k in range (1, n):
			totGamma_PI=np.zeros(3)
			totGamma_PH=np.zeros(3)

			escfrac=self.set_escfrac_evol_popII(Z[k])

			current_redshift=Z[k]
			tdyn_z=(super().Hubble_dist(Z[k])/self.h)*9.7776e09
			dtimedz_z=-(9.777e09*yrbysec)*super().func(Z[k])/self.h
			sigma_z=np.sqrt(super().sigmasq_b_sourav(0.0, Z[k],Global_region['T'][k-1],1.0/Global_region['X'][k-1]))

			Sigma[k]=sigma_z

			lumfun_integral_qso[k]=e_QSO*lumfun_integral(Z[k])
			tempz=Global_region['T'][k-1]

			

			super(Redshift_Evolution, self).__init__(*self.cosmo_param_array, *self.astrophysics_array,  logM=self.logM, logsig=self.logsig, delta_c=delta_c, initial_call=False, zcoll=Z[k], current_redshift=current_redshift, sigma=sigma_z, dtimedz=dtimedz_z, tdyn=tdyn_z, vcmax=1e09)

			dnphotdz_neut_z, dnphotdz_ion_z, dnphotdz['H'][k], dnphotdz['He'][k], sfr_pop2_ion, sfr_pop2_neut, sfr_pop3_ion, sfr_pop3_neut= super().get_sfr(Z[k],Ionized_species['QH']['Q'][k-1], escfrac, dnphotdm, lumfun_integral_qso[k], HII_region['T'][k-1], HII_region['X'][k-1])


			self.PopII_sfr[k]=-(sfr_pop2_ion+ sfr_pop2_neut)
			self.PopIII_sfr[k]=-(sfr_pop3_ion+ sfr_pop3_neut)

			Gamma_PI_z,Gamma_PH_z, Ionized_species['QH']['mfp'][k], Ionized_species['QHe']['mfp'][k] = super().get_ionflux(k, MCMC_lambda_0, Z[k], HII_0['T'][k-1], HII_0['X'][k-1], HeIII_0['T'][k-1], HeIII_0['X'][k-1], Ionized_species['QH'], Ionized_species['QHe'], sigma_PI, sigma_PH, dnphotdz_neut_z, dnphotdz_ion_z)			
			

			Gamma_PI['PopII']['HII'][k], Gamma_PI['PopII']['HeII'][k], Gamma_PI['PopII']['HeIII'][k]=Gamma_PI_z[0][0], Gamma_PI_z[0][1], Gamma_PI_z[0][2]

			Gamma_PI['PopIII']['HII'][k], Gamma_PI['PopIII']['HeII'][k], Gamma_PI['PopIII']['HeIII'][k]=Gamma_PI_z[1][0], Gamma_PI_z[1][1], Gamma_PI_z[1][2]

			Gamma_PI['QSO']['HII'][k], Gamma_PI['QSO']['HeII'][k], Gamma_PI['QSO']['HeIII'][k]=Gamma_PI_z[2][0], Gamma_PI_z[2][1], Gamma_PI_z[2][2]


			


			neutral_region['frac']['HI'][k]=neutral_region['frac']['HI'][k-1]
			neutral_region['frac']['HeI'][k]=neutral_region['frac']['HeI'][k-1]
			neutral_region['frac']['HeIII'][k]=neutral_region['frac']['HeIII'][k-1]
			neutral_region['T'][k]=neutral_region['T'][k-1]
			

			HII_region['frac']['HI'][k]=HII_region['frac']['HI'][k-1]
			HII_region['frac']['HeI'][k]=HII_region['frac']['HeI'][k-1]
			HII_region['frac']['HeIII'][k]=HII_region['frac']['HeIII'][k-1]
			HII_region['T'][k]=HII_region['T'][k-1]


			HeIII_region['frac']['HI'][k]=HeIII_region['frac']['HI'][k-1]
			HeIII_region['frac']['HeI'][k]=HeIII_region['frac']['HeI'][k-1]
			HeIII_region['frac']['HeIII'][k]=HeIII_region['frac']['HeIII'][k-1]
			HeIII_region['T'][k]=HeIII_region['T'][k-1]
			

			therm=Thermal(dtimedz_z, MCMC_ombh2)

			neutral_region['recrate']['HI'][k], neutral_region['recrate']['HeI'][k], neutral_region['recrate']['HeIII'][k], neutral_region['coolrate']['HI'][k], neutral_region['coolrate']['HeI'][k], neutral_region['coolrate']['HeIII'][k]=therm.get_recrates(Z[k], neutral_region['T'][k],1.0,1.0)


			

			HII_region['recrate']['HI'][k], HII_region['recrate']['HeI'][k], HII_region['recrate']['HeIII'][k], HII_region['coolrate']['HI'][k], HII_region['coolrate']['HeI'][k], HII_region['coolrate']['HeIII'][k]=therm.get_recrates(Z[k], HII_region['T'][k], Ionized_species['QH']['R'][k-1], 1.0) 

			
		
			HeIII_region['recrate']['HI'][k], HeIII_region['recrate']['HeI'][k], HeIII_region['recrate']['HeIII'][k], HeIII_region['coolrate']['HI'][k], HeIII_region['coolrate']['HeI'][k], HeIII_region['coolrate']['HeIII'][k]=therm.get_recrates(Z[k], HeIII_region['T'][k], Ionized_species['QHe']['R'][k-1], 1.0) 

			
			
			neutral_region['ionrate']['HI'][k], neutral_region['ionrate']['HeI'][k], neutral_region['ionrate']['HeIII'][k] = 0.0,0.0,0.0
			

			totGamma_PI[0]=Gamma_PI_z[0][0]+Gamma_PI_z[1][0]+Gamma_PI_z[2][0]
			totGamma_PI[1]=Gamma_PI_z[0][1]+Gamma_PI_z[1][1]+Gamma_PI_z[2][1]
			totGamma_PI[2]=0.0
		 		
			totGamma_PH[0]=Gamma_PH_z[0][0]+Gamma_PH_z[1][0]+Gamma_PH_z[2][0]
			totGamma_PH[1]=Gamma_PH_z[0][1]+Gamma_PH_z[1][1]+Gamma_PH_z[2][1]
			totGamma_PH[2]=0.0



		
			HII_region['ionrate']['HI'][k], HII_region['ionrate']['HeI'][k], HII_region['ionrate']['HeIII'][k], HII_region['heatrate']['HI'][k], HII_region['heatrate']['HeI'][k],  HII_region['heatrate']['HeIII'][k]=therm.get_ionrates(Z[k], totGamma_PI, totGamma_PH, Ionized_species['QH']['Q'][k-1])

			
			totGamma_PI[2]=(Gamma_PI_z[0][2]+Gamma_PI_z[1][2]+Gamma_PI_z[2][2])*( Ionized_species['QHe']['mfp'][k-1]/ Ionized_species['QH']['mfp'][k-1])*(Ionized_species['QH']['Q'][k-1]/Ionized_species['QHe']['Q'][k-1])
			totGamma_PH[2]=(Gamma_PH_z[0][2]+Gamma_PH_z[1][2]+Gamma_PH_z[2][2])*( Ionized_species['QHe']['mfp'][k-1]/ Ionized_species['QH']['mfp'][k-1])*(Ionized_species['QH']['Q'][k-1]/Ionized_species['QHe']['Q'][k-1])


			HeIII_region['ionrate']['HI'][k], HeIII_region['ionrate']['HeI'][k], HeIII_region['ionrate']['HeIII'][k], HeIII_region['heatrate']['HI'][k], HeIII_region['heatrate']['HeI'][k],  HeIII_region['heatrate']['HeIII'][k]=therm.get_ionrates(Z[k], totGamma_PI, totGamma_PH, Ionized_species['QH']['Q'][k-1])

			
			neutral_region['heatrate']['HI'][k], neutral_region['heatrate']['HeI'][k], neutral_region['heatrate']['HeIII'][k] =0.0, 0.0, 0.0


			neutral_region=therm.update_ionstate(Z[k], k, dz, neutral_region)
			HII_region=therm.update_ionstate(Z[k], k, dz, HII_region)

			HeIII_region=therm.update_ionstate(Z[k], k, dz, HeIII_region)


			neutral_0['frac']['HI'][k]=neutral_0['frac']['HI'][k-1]
			neutral_0['frac']['HeI'][k]=neutral_0['frac']['HeI'][k-1]
			neutral_0['frac']['HeIII'][k]=neutral_0['frac']['HeIII'][k-1]
			neutral_0['T'][k]=neutral_0['T'][k-1]


			HII_0['frac']['HI'][k]=HII_0['frac']['HI'][k-1]
			HII_0['frac']['HeI'][k]=HII_0['frac']['HeI'][k-1]
			HII_0['frac']['HeIII'][k]=HII_0['frac']['HeIII'][k-1]
			HII_0['T'][k]=HII_0['T'][k-1]

			HeIII_0['frac']['HI'][k]=HeIII_0['frac']['HI'][k-1]
			HeIII_0['frac']['HeI'][k]=HeIII_0['frac']['HeI'][k-1]
			HeIII_0['frac']['HeIII'][k]=HeIII_0['frac']['HeIII'][k-1]
			HeIII_0['T'][k]=HeIII_0['T'][k-1]


			neutral_0['recrate']['HI'][k], neutral_0['recrate']['HeI'][k], neutral_0['recrate']['HeIII'][k], neutral_0['coolrate']['HI'][k], neutral_0['coolrate']['HeI'][k], neutral_0['coolrate']['HeIII'][k]=therm.get_recrates(Z[k], neutral_0['T'][k],1.0,1.0)

			HII_0['recrate']['HI'][k], HII_0['recrate']['HeI'][k], HII_0['recrate']['HeIII'][k], HII_0['coolrate']['HI'][k], HII_0['coolrate']['HeI'][k], HII_0['coolrate']['HeIII'][k]=therm.get_recrates(Z[k], HII_0['T'][k],1.0,1.0)


			HeIII_0['recrate']['HI'][k], HeIII_0['recrate']['HeI'][k], HeIII_0['recrate']['HeIII'][k], HeIII_0['coolrate']['HI'][k], HeIII_0['coolrate']['HeI'][k], HeIII_0['coolrate']['HeIII'][k]=therm.get_recrates(Z[k], HeIII_0['T'][k],1.0,1.0)



			neutral_0['ionrate']['HI'][k]=neutral_region['ionrate']['HI'][k]
			neutral_0['ionrate']['HeI'][k]=neutral_region['ionrate']['HeI'][k]
			neutral_0['ionrate']['HeIII'][k]=neutral_region['ionrate']['HeIII'][k]

			HII_0['ionrate']['HI'][k]=HII_region['ionrate']['HI'][k]
			HII_0['ionrate']['HeI'][k]=HII_region['ionrate']['HeI'][k]
			HII_0['ionrate']['HeIII'][k]=HII_region['ionrate']['HeIII'][k]


			HeIII_0['ionrate']['HI'][k]=HeIII_region['ionrate']['HI'][k]
			HeIII_0['ionrate']['HeI'][k]=HeIII_region['ionrate']['HeI'][k]
			HeIII_0['ionrate']['HeIII'][k]=HeIII_region['ionrate']['HeIII'][k]



			neutral_0['heatrate']['HI'][k]=neutral_region['heatrate']['HI'][k]
			neutral_0['heatrate']['HeI'][k]=neutral_region['heatrate']['HeI'][k]
			neutral_0['heatrate']['HeIII'][k]=neutral_region['heatrate']['HeIII'][k]



			HII_0['heatrate']['HI'][k]=HII_region['heatrate']['HI'][k]
			HII_0['heatrate']['HeI'][k]=HII_region['heatrate']['HeI'][k]
			HII_0['heatrate']['HeIII'][k]=HII_region['heatrate']['HeIII'][k]


			HeIII_0['heatrate']['HI'][k]=HeIII_region['heatrate']['HI'][k]
			HeIII_0['heatrate']['HeI'][k]=HeIII_region['heatrate']['HeI'][k]
			HeIII_0['heatrate']['HeIII'][k]=HeIII_region['heatrate']['HeIII'][k]



			neutral_0=therm.update_ionstate(Z[k], k, dz, neutral_0)
			HII_0=therm.update_ionstate(Z[k], k, dz, HII_0)

			HeIII_0=therm.update_ionstate(Z[k], k, dz, HeIII_0)


			n_e_H=(HII_region['X_e'][k]+(Ionized_species['QHe']['Q'][k-1]/Ionized_species['QH']['Q'][k-1])*HeIII_region['X_e'][k])*self.rho_b/mprot*(1.8791e-29/2.7755e11)
			n_H=(HII_region['X_HII'][k]+HII_region['frac']['HI'][k])*self.rho_b/mprot*(1.8791e-29/2.7755e11)

			
			QH_z=np.array([Ionized_species['QH']['Q'][k], Ionized_species['QH']['Delta'][k], Ionized_species['QH']['F_V'][k], Ionized_species['QH']['F_M'][k], Ionized_species['QH']['R'][k]])
			
			QH_zprev=np.array([Ionized_species['QH']['Q'][k-1], Ionized_species['QH']['Delta'][k-1], Ionized_species['QH']['F_V'][k-1], Ionized_species['QH']['F_M'][k-1], Ionized_species['QH']['R'][k-1]])



			Ionized_species['QH']['Q'][k], Ionized_species['QH']['Delta'][k], Ionized_species['QH']['F_V'][k], Ionized_species['QH']['F_M'][k], Ionized_species['QH']['R'][k]=super().update_Q(Z[k], dz, QH_z, QH_zprev, Sigma[k-1], dnphotdz['H'][k], n_e_H, n_H, R_HII_e_A(HII_region['T'][k])+R_HII_e_B(HII_region['T'][k]))


			n_e_He=HeIII_region['X_e'][k]*self.rho_b/mprot*(1.8791e-29/2.7755e11)
			n_He=(HeIII_region['frac']['HeIII'][k]+HeIII_region['X_HeII'][k]+HeIII_region['frac']['HeI'][k])*self.rho_b/mprot*(1.8791e-29/2.7755e11)
			
			QHe_z=np.array([Ionized_species['QHe']['Q'][k], Ionized_species['QHe']['Delta'][k], Ionized_species['QHe']['F_V'][k], Ionized_species['QHe']['F_M'][k], Ionized_species['QHe']['R'][k]])
			QHe_zprev=np.array([Ionized_species['QHe']['Q'][k-1], Ionized_species['QHe']['Delta'][k-1], Ionized_species['QHe']['F_V'][k-1], Ionized_species['QHe']['F_M'][k-1], Ionized_species['QHe']['R'][k-1]])

			Ionized_species['QHe']['Q'][k], Ionized_species['QHe']['Delta'][k], Ionized_species['QHe']['F_V'][k], Ionized_species['QHe']['F_M'][k], Ionized_species['QHe']['R'][k]=super().update_Q(Z[k], dz, QHe_z, QHe_zprev, Sigma[k-1], dnphotdz['He'][k], n_e_He, n_He, R_HeIII_e_A(HeIII_region['T'][k])+R_HeIII_e_B(HeIII_region['T'][k]))



			

			QH=Ionized_species['QH']
			QHe=Ionized_species['QHe']








			Global_region['T'][k]=(1.0-QH['Q'][k]*QH['F_M'][k])*neutral_region['T'][k]+(QH['Q'][k]*QH['F_M'][k]-QHe['Q'][k]*QHe['F_M'][k])*HII_region['T'][k]+QHe['Q'][k]*QHe['F_M'][k]*HeIII_region['T'][k]
				
			
			Global_region['frac']['HI'][k]=(1.0-QH['Q'][k]*QH['F_M'][k])*neutral_region['frac']['HI'][k]+(QH['Q'][k]*QH['F_M'][k]-QHe['Q'][k]*QHe['F_M'][k])*HII_region['frac']['HI'][k]+QHe['Q'][k]*QHe['F_M'][k]*HeIII_region['frac']['HI'][k]

			Global_region['frac']['HeI'][k]=(1.0-QH['Q'][k]*QH['F_M'][k])*neutral_region['frac']['HeI'][k]+(QH['Q'][k]*QH['F_M'][k]-QHe['Q'][k]*QHe['F_M'][k])*HII_region['frac']['HeI'][k]+QHe['Q'][k]*QHe['F_M'][k]*HeIII_region['frac']['HeI'][k]

			Global_region['frac']['HeIII'][k]=(1.0-QH['Q'][k]*QH['F_M'][k])*neutral_region['frac']['HeIII'][k]+(QH['Q'][k]*QH['F_M'][k]-QHe['Q'][k]*QHe['F_M'][k])*HII_region['frac']['HeIII'][k]+QHe['Q'][k]*QHe['F_M'][k]*HeIII_region['frac']['HeIII'][k]


			
			Global_region['X_HII'][k],Global_region['X_HeII'][k],Global_region['X_e'][k],Global_region['X'][k]=compute_species(Global_region['frac']['HI'][k], Global_region['frac']['HeI'][k], Global_region['frac']['HeIII'][k])

			Global_0['T'][k]=(1.0-QH['Q'][k])*neutral_0['T'][k]+(QH['Q'][k]-QHe['Q'][k])*HII_0['T'][k]+QHe['Q'][k]*HeIII_0['T'][k]




			Global_0['frac']['HI'][k]=(1.0-QH['Q'][k])*neutral_0['frac']['HI'][k]+(QH['Q'][k]-QHe['Q'][k])*HII_0['frac']['HI'][k]+QHe['Q'][k]*HeIII_0['frac']['HI'][k]

			Global_0['frac']['HeI'][k]=(1.0-QH['Q'][k])*neutral_0['frac']['HeI'][k]+(QH['Q'][k]-QHe['Q'][k])*HII_0['frac']['HeI'][k]+QHe['Q'][k]*HeIII_0['frac']['HeI'][k]

			Global_0['frac']['HeIII'][k]=(1.0-QH['Q'][k])*neutral_0['frac']['HeIII'][k]+(QH['Q'][k]-QHe['Q'][k])*HII_0['frac']['HeIII'][k]+QHe['Q'][k]*HeIII_0['frac']['HeIII'][k]



			Global_0['X_HII'][k], Global_0['X_HeII'][k], Global_0['X_e'][k], Global_0['X'][k]=compute_species(Global_0['frac']['HI'][k], Global_0['frac']['HeI'][k], Global_0['frac']['HeIII'][k])


		
			dNLLdz[k]=1.0/((1.0+Z[k])*(math.sqrt(math.pi)*QH['mfp'][k]/(Mpcbycm*super().Hubble_dist(Z[k])*3e3/self.h)))

			n_e=Global_0['X_e'][k]*self.rho_b/mprot*(1.8791*1e-29/2.7755/1e11)

			tau_elsc[k]=tau_elsc[k-1]+dz*dtimedz_z*n_e*3*1e10*6.652*10**(-25)*(1.0+Z[k])**3


			self.dNLLdz_array[k]=dNLLdz[k]

			self.Q_HII[k]=Ionized_species['QH']['Q'][k]

			self.gamma_PI[k]=(Gamma_PI['PopII']['HII'][k]+Gamma_PI['PopIII']['HII'][k]+Gamma_PI['QSO']['HII'][k])/10**-12

			self.redshift_array[k]=Z[k]
			
			self.x_HI_array[k]=Global_0['frac']['HI'][k]/(Global_0['frac']['HI'][k]+Global_0['X_HII'][k])


		tau_elsc_today=tau_elsc[n-1]
		tau_elsc=tau_elsc_today-tau_elsc
		self.tau=tau_elsc_today

	def quntity_for_MCMC(self):	
		#if self.use_21_cm:
			#return self.redshift_array, self.Q_HII, self.tau, self.dNLLdz_array, self.gamma_PI, self.PopII_sfr, self.PopIII_sfr, self.Q_HII[96]
		#else:
		
		return self.redshift_array, self.Q_HII, self.tau, self.dNLLdz_array, self.gamma_PI, self.Q_HII[np.logical_and(Z<6.0, Z>5.6)], self.x_HI_array 

	def set_escfrac_evol_popII(self, z):
		#redshift=z.reshape(1,-1)
		mean_prediction, std_prediction = self.GPR_initerpolator.predict(z.reshape(-1,1), return_std=True)
		escfrac['PopII']['HII'], escfrac['PopII']['HeII'], escfrac['PopII']['HeIII']= mean_prediction, mean_prediction, mean_prediction
		escfrac['PopIII']['HII'], escfrac['PopIII']['HeII'], escfrac['PopIII']['HeIII']= self.MCMC_esc_PopIII, self.MCMC_esc_PopIII, self.MCMC_esc_PopIII
		return escfrac




#dict={'redshift': self.redshift_array, 'QHI': self.x_HI , 'dNLLdz': self.dNLLdz_array , 'gamma_PI_H': self.gamma_PI}
#df=pd.DataFrame(dict)
#print(df.head())
#df.to_csv('reion_product.csv', index=False)
#data=pd.read_csv('reion_product.csv')
#print(data.head())


