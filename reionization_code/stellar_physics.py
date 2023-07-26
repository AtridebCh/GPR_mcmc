import numpy as np
import scipy as sc

from scipy import integrate
from scipy.integrate import quad
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

from constants import rho_c, vc_min, e_sf_II, e_sf_III, chemfeedback, kboltz, mprot, yrbysec, nu_HII, nu_HeII, nu_HeIII, hPlanck, Mpcbycm
from cosmological_functions import Cosmology



def vcmin_pop3_neut():
	return vc_min
		
def vcmin_pop2_neut():
	return vc_min

def vcmin_pop3_ion(T, X):
	ans=np.sqrt(2.0*kboltz*T*X/mprot)*1e-05
	if (ans < vcmin_pop3_neut()):
		return vcmin_pop3_neut()
	else:
		return ans
		   

def vcmin_pop2_ion(T, X):
	ans=np.sqrt(2.0*kboltz*T*X/mprot)*1e-05
	if (ans < vcmin_pop2_neut()):
		return vcmin_pop2_neut()
	else:
		return ans


def vc_max():
	return 1e09
		


class PopII_PopIII(Cosmology):

	def __init__(self, MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8,  MCMC_ns, **kwargs):

		if kwargs['call_for_first_time']:
			pass
		else:
			self.zcoll=kwargs['zcoll']
			self.delta_c=kwargs['delta_c']
			self.logM=kwargs['logM']
			self.logsig=kwargs['logsig']
			self.vcmax=kwargs['vcmax']
			self.zmass=self.zcoll
			self.dtimedz=kwargs['dtimedz']
			self.tdyn=kwargs['tdyn']

		super(PopII_PopIII, self).__init__(MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, **kwargs)


	def get_sfr(self, z, QH_Q_prev_z, escfrac, dnphotdm, lumfun_integral_qso, temp_HII, electron_denisty_HII):	

		dtimedz=self.dtimedz
		tdyn=self.tdyn

		MinimumVcPop2Ion, MinimumVcPop2Neut= vcmin_pop2_ion(temp_HII,electron_denisty_HII), vcmin_pop2_neut() 
		MinimumVcPop3Ion, MinimumVcPop3Neut= vcmin_pop3_ion(temp_HII,electron_denisty_HII), vcmin_pop3_neut()
	
		dnphotdz_ion=np.zeros((3,3))
		dnphotdz_neut=np.zeros((3,3))

	
		prefactor=super().omega_z(z)**0.6

	
		dfcolldt_pop2_ion=self.mass_integral_pop2(MinimumVcPop2Ion)*(prefactor/tdyn)

	
		dfcolldt_pop2_neut=self.mass_integral_pop2(MinimumVcPop2Neut)*(prefactor/tdyn) 
	
	
	
		dfcolldt_pop3_ion=self.mass_integral_pop3(MinimumVcPop3Ion)*(prefactor/tdyn)
		dfcolldt_pop3_neut=self.mass_integral_pop3(MinimumVcPop3Neut)*(prefactor/tdyn)
	
		
	
	
		mass_integral_pop2_neut=e_sf_II*dfcolldt_pop2_neut
		mass_integral_pop2_ion=e_sf_II*dfcolldt_pop2_ion
		mass_integral_pop3_neut=e_sf_III*dfcolldt_pop3_neut
		mass_integral_pop3_ion=e_sf_III*dfcolldt_pop3_ion
	 
		sfrfactor=self.rho_b



		dnphotdz_ion[0]=[QH_Q_prev_z*dnphotdm['PopII']['HII']*mass_integral_pop2_ion*sfrfactor*escfrac['PopII']['HII']*dtimedz/yrbysec,QH_Q_prev_z*dnphotdm['PopII']['HeII']*mass_integral_pop2_ion*sfrfactor*escfrac['PopII']['HeII']*dtimedz/yrbysec,QH_Q_prev_z*dnphotdm['PopII']['HeIII']*mass_integral_pop2_ion*sfrfactor*escfrac['PopII']['HeIII']*dtimedz/yrbysec]
	
		dnphotdz_ion[1]=[QH_Q_prev_z*dnphotdm['PopIII']['HII']*mass_integral_pop3_ion*sfrfactor*escfrac['PopIII']['HII']*dtimedz/yrbysec, QH_Q_prev_z*dnphotdm['PopIII']['HeII']*mass_integral_pop3_ion*sfrfactor*escfrac['PopIII']['HeII']*dtimedz/yrbysec, QH_Q_prev_z*dnphotdm['PopIII']['HeIII']*mass_integral_pop3_ion*sfrfactor*escfrac['PopIII']['HeIII']*dtimedz/yrbysec]

		dnphotdz_ion[2]=[(1.0-QH_Q_prev_z)*yrbysec*10.0**18.05*(1.0-(nu_HII/nu_HeII)**1.570)*lumfun_integral_qso/(1.570*hPlanck)*dtimedz/yrbysec,(1.0-QH_Q_prev_z)*yrbysec*10.0**18.05*((nu_HII/(nu_HeII))**1.57-(nu_HII/(nu_HeIII))**1.57)*lumfun_integral_qso/(1.57*hPlanck)*dtimedz/yrbysec,(1.0-QH_Q_prev_z)*yrbysec*10.0**18.05*(nu_HII/(nu_HeIII))**1.57*lumfun_integral_qso/(1.57*hPlanck)*dtimedz/yrbysec]

		



		dnphotdz_neut[0]=[(1.0-QH_Q_prev_z)*dnphotdm['PopII']['HII']*mass_integral_pop2_neut*sfrfactor*escfrac['PopII']['HII']*dtimedz/yrbysec,(1.0-QH_Q_prev_z)*dnphotdm['PopII']['HeII']*mass_integral_pop2_neut*sfrfactor*escfrac['PopII']['HeII']*dtimedz/yrbysec,(1.0-QH_Q_prev_z)*dnphotdm['PopII']['HeIII']*mass_integral_pop2_neut*sfrfactor*escfrac['PopII']['HeIII']*dtimedz/yrbysec]

		dnphotdz_neut[1]=[(1.0-QH_Q_prev_z)*dnphotdm['PopIII']['HII']*mass_integral_pop3_neut*sfrfactor*escfrac['PopIII']['HII']*dtimedz/yrbysec,(1.0-QH_Q_prev_z)*dnphotdm['PopIII']['HeII']*mass_integral_pop3_neut*sfrfactor*escfrac['PopIII']['HeII']*dtimedz/yrbysec,(1.0-QH_Q_prev_z)*dnphotdm['PopIII']['HeIII']*mass_integral_pop3_neut*sfrfactor*escfrac['PopIII']['HeIII']*dtimedz/yrbysec]

		dnphotdz_neut[2]=[QH_Q_prev_z*yrbysec*10.0**18.05*(1.0-(nu_HII/nu_HeII)**1.570)*lumfun_integral_qso/(1.570*hPlanck)*dtimedz/yrbysec, QH_Q_prev_z*yrbysec*10.0**18.05*((nu_HII/(nu_HeII))**1.57-(nu_HII/(nu_HeIII))**1.57)*lumfun_integral_qso/(1.57*hPlanck)*dtimedz/yrbysec,QH_Q_prev_z*yrbysec*10.0**18.05*(nu_HII/(nu_HeIII))**1.570*lumfun_integral_qso/(1.57*hPlanck)*dtimedz/yrbysec]


		sfr_pop2_ion=-QH_Q_prev_z*mass_integral_pop2_ion*sfrfactor
	
		sfr_pop2_neut=-(1.0-QH_Q_prev_z)*mass_integral_pop2_neut*sfrfactor # These values would be needed when we use 21-cm signal along with the reionization
	
		sfr_pop3_ion=-QH_Q_prev_z*mass_integral_pop3_ion*sfrfactor

		sfr_pop3_neut=-(1.0-QH_Q_prev_z)*mass_integral_pop3_neut*sfrfactor


	


		dnphotdz_H=(dnphotdz_neut[0][0]+dnphotdz_neut[1][0]+dnphotdz_neut[2][0]+dnphotdz_neut[0][1]+dnphotdz_neut[1][1]+dnphotdz_neut[2][1])+\
		(dnphotdz_ion[0][0]+dnphotdz_ion[1][0]+dnphotdz_ion[2][0]+dnphotdz_ion[0][1]+dnphotdz_ion[1][1]+dnphotdz_ion[2][1])
	
		dnphotdz_He=(dnphotdz_neut[0][2]+dnphotdz_neut[1][2]+dnphotdz_neut[2][2])+(dnphotdz_ion[0][2]+dnphotdz_ion[1][2]+dnphotdz_ion[2][2])


		return  dnphotdz_neut, dnphotdz_ion, dnphotdz_H, dnphotdz_He, sfr_pop2_ion, sfr_pop2_neut, sfr_pop3_ion, sfr_pop3_neut



	def get_ionflux(self, k, lambda_0, z, HII_0_prev_temp, HII_0_prev_elect, HeIII_0_prev_temp, HeIII_0_prev_elect, QH, QHe, sigma_PI, sigma_PH, dnphotdz_neut, dnphotdz_ion):

		dtimedz=self.dtimedz


		Gamma_PI=np.zeros((3,3))
		Gamma_PH=np.zeros((3,3))

		lambda_0_H=lambda_0*np.sqrt(super().xbsq(HII_0_prev_temp, z, 1.0/HII_0_prev_elect))/(1.0+z)
		lambda_0_He=lambda_0*np.sqrt(super().xbsq(HeIII_0_prev_temp,z,1.0/HeIII_0_prev_elect))/(1.0+z)


		if (1.0-QH['F_V'][k-1]>= 0.0):
			QH['mfp'][k]=QH['Q'][k-1]**(1.0/3.0)*lambda_0_H*Mpcbycm/(1.0-QH['F_V'][k-1])**(2.0/3.0)
		else:
			QH['mfp'][k]=QH['mfp'][k-1]
			
		if (1.0-QHe['F_V'][k-1] >= 0.0):
			QHe['mfp'][k]=QHe['Q'][k-1]**(1.0/3.0)*lambda_0_He*Mpcbycm/(1.0-QHe['F_V'][k-1])**(2.0/3.0)
		else:
			QHe['mfp'][k]=QHe['mfp'][k-1]
			
	
	
		Gamma_PI[0]=[QH['mfp'][k]*sigma_PI['PopII']['HII']*(1.0+z)**3*(dnphotdz_neut[0][0]+dnphotdz_ion[0][0])/(dtimedz*Mpcbycm**3), QH['mfp'][k]*sigma_PI['PopII']['HeII']*(1.0+z)**3*(dnphotdz_neut[0][1]+dnphotdz_ion[0][1])/(dtimedz*Mpcbycm**3),QHe['mfp'][k]*sigma_PI['PopII']['HeIII']*(1.0+z)**3*(dnphotdz_neut[0][2]+dnphotdz_ion[0][2])/(dtimedz*Mpcbycm**3)]
	
	
		Gamma_PI[1]=[QH['mfp'][k]*sigma_PI['PopIII']['HII']*(1.0+z)**3*(dnphotdz_neut[1][0]+dnphotdz_ion[1][0])/(dtimedz*Mpcbycm**3), QH['mfp'][k]*sigma_PI['PopIII']['HeII']*(1.0+z)**3*(dnphotdz_neut[1][1]+dnphotdz_ion[1][1])/(dtimedz*Mpcbycm**3),QHe['mfp'][k]*sigma_PI['PopIII']['HeIII']*(1.0+z)**3*(dnphotdz_neut[1][2]+dnphotdz_ion[1][2])/(dtimedz*Mpcbycm**3)]
	
	
		Gamma_PI[2]=[QH['mfp'][k]*sigma_PI['QSO']['HII']*(1.0+z)**3*(dnphotdz_neut[2][0]+dnphotdz_ion[2][0])/(dtimedz*Mpcbycm**3), QH['mfp'][k]*sigma_PI['QSO']['HeII']*(1.0+z)**3*(dnphotdz_neut[2][1]+dnphotdz_ion[2][1])/(dtimedz*Mpcbycm**3),QHe['mfp'][k]*sigma_PI['QSO']['HeIII']*(1.0+z)**3*(dnphotdz_neut[2][2]+dnphotdz_ion[2][2])/(dtimedz*Mpcbycm**3)]

		Gamma_PH[0]=[hPlanck*nu_HII*QH['mfp'][k]*sigma_PH['PopII']['HII']*(1.0+z)**3*(dnphotdz_neut[0][0]+dnphotdz_ion[0][0])/(dtimedz*Mpcbycm**3), hPlanck*nu_HeII*QH['mfp'][k]*sigma_PH['PopII']['HeII']*(1.0+z)**3*(dnphotdz_neut[0][1]+dnphotdz_ion[0][1])/(dtimedz*Mpcbycm**3), hPlanck*nu_HeIII*QHe['mfp'][k]*sigma_PH['PopII']['HeIII']*(1.0+z)**3*(dnphotdz_neut[0][2]+dnphotdz_ion[0][2])/(dtimedz*Mpcbycm**3)]
	
	
	
		Gamma_PH[1]=[hPlanck*nu_HII*QH['mfp'][k]*sigma_PH['PopIII']['HII']*(1.0+z)**3*(dnphotdz_neut[1][0]+dnphotdz_ion[1][0])/(dtimedz*Mpcbycm**3),hPlanck*nu_HeII*QH['mfp'][k]*sigma_PH['PopIII']['HeII']*(1.0+z)**3*(dnphotdz_neut[1][1]+dnphotdz_ion[1][1])/(dtimedz*Mpcbycm**3), hPlanck*nu_HeIII*QHe['mfp'][k]*sigma_PH['PopIII']['HeIII']*(1.0+z)**3*(dnphotdz_neut[1][2]+dnphotdz_ion[1][2])/(dtimedz*Mpcbycm**3)]
	
	
		Gamma_PH[2]=[hPlanck*nu_HII*QH['mfp'][k]*sigma_PH['QSO']['HII']*(1.0+z)**3*(dnphotdz_neut[2][0]+dnphotdz_ion[2][0])/(dtimedz*Mpcbycm**3),hPlanck*nu_HeII*QH['mfp'][k]*sigma_PH['QSO']['HeII']*(1.0+z)**3*(dnphotdz_neut[2][1]+dnphotdz_ion[2][1])/(dtimedz*Mpcbycm**3), hPlanck*nu_HeIII*QHe['mfp'][k]*sigma_PH['QSO']['HeIII']*(1.0+z)**3*(dnphotdz_neut[2][2]+dnphotdz_ion[2][2])/(dtimedz*Mpcbycm**3)]


		
		return Gamma_PI,Gamma_PH, QH['mfp'][k], QHe['mfp'][k]



	def mass_integral_pop2(self, vcmin):
		zmass=self.zcoll
		if (abs(vcmin-self.vcmax) < 1e-08):
			return 0.0
		elif (vcmin>self.vcmax):
			return 0.0
		   
		mmin=self.mass_func(vcmin,self.zcoll)/self.h
		numin=self.nu_parameter(mmin)

		nu_array=np.linspace(numin,10.0,2000)
		dnu=nu_array[1]-nu_array[0]
		if (self.vcmax > 1e05):
			ans=sc.integrate.simps(self.mass_integrand_pop2(nu_array,zmass),dx=dnu)
			return ans  
		else:
			mmax=self.mass_func(self.vcmax,self.zcoll)/self.h
			numax=self.nu_parameter(mmax)
			ans,err=quad(self.mass_integrand_pop2,numin,numax,args=(zmass),epsabs=1.49e-03, epsrel=1.49e-03)
			return ans



	def mass_integral_pop3(self, vcmin):
		zmass=self.zcoll
		if (abs(vcmin-self.vcmax) < 1e-08):
			return 0.0
		elif (vcmin>self.vcmax):
			return 0.0
		   
		mmin=self.mass_func(vcmin, zmass)/self.h
		numin=self.nu_parameter(mmin)
		nu_array=np.linspace(numin,10.0,2000)
		dnu=nu_array[1]-nu_array[0]
		if (self.vcmax > 1e05):
			ans=sc.integrate.simps(self.mass_integrand_pop3(nu_array,zmass),dx=dnu)

			if ans<0.0:
				ans=0.0  
		else:
			mmax=self.mass_func(self)/h
			numax=nu_parameter(self,mmax)
			ans,err=quad(self.mass_integrand_pop3,numin,numax,args=(self,zmass),epsabs=1.49e-03, epsrel=1.49e-03)
			if ans<0.0:
				ans=0.0
		return ans



	def fbbym(self,m, e_sf_III, chemfeedback ):
		if ((feedback == 0) or (feedback == 1)):
			return 1.0



	def fpop3(self, m):
		if (e_sf_III < 1e-06):
			return 0.0
		if (chemfeedback== 0):
			return 1.0
		mass_m=10.0**m
		mass_min=self.mass_func(vcmin_pop3_neut(), self.zcoll)/self.h

		factor=1e06
		mass_form=mass_m-factor
		m_form=np.log10(mass_form)
		m_min=np.log10(mass_min)
		spl=splrep(self.logM, self.logsig)
		sigma_m=splev(m,spl)
		sigma_m=10.0**sigma_m
		   
		sigma_min=splev(m_min,spl)
		sigma_min=10.0**sigma_min

		sigma_form=splev(m_form,spl)
		sigma_form=10.0**sigma_form

		arg=np.sqrt(abs((sigma_form-sigma_m)/(sigma_min-sigma_form)))
		arg=np.where(arg<1e-05,arg,2.0*np.arctan(arg)/np.pi)
		#if self.use_21_cm:
			#return arg*(0.5+0.5*np.tanh((zcoll-z_trans)/dt_to_dz_tanh(z_trans)))
		#else:
		return arg



	def mass_integrand_pop2(self, nu, zmass):
		sig=self.delta_c/(nu*self.d(zmass))
		spl=splrep(np.sort(self.logsig),self.logM[np.argsort(self.logsig)])
		logM_new=splev(np.log10(sig),spl)
		mass_integrand_pop2_ans=self.probdist(nu)*(nu*nu-1.0)
		return mass_integrand_pop2_ans*(1.0-self.fpop3(logM_new))


		


	def mass_integrand_pop3(self, nu,zmass):
		sig=self.delta_c/(nu*self.d(zmass))
		spl=splrep(np.sort(self.logsig),self.logM[np.argsort(self.logsig)])
		logM_new=splev(np.log10(sig),spl)
		mass_integrand_pop3_ans=self.probdist(nu)*(nu*nu-1.0)
		return mass_integrand_pop3_ans*self.fpop3(logM_new)
	



	def nu_parameter(self,mass):
		rho_0=rho_c*self.omega_m*self.h**2
		x=(3.0*mass/(4.0*np.pi*rho_0))**(1.0/3.0)
		return self.delta_c/(self.sigma_sourav(x)*self.d(self.zcoll))


	def delvir(self):
		x=self.omega_z(self.zcoll)-1.0
		return 18.0*np.pi**2+82.0*x-39.0*x*x



	def coeff_vc_mass(self):
		hsq=1.0/(self.Hubble_dist(self.zcoll))**2
		return (3.504/(1e08**(1.0/3.0)))*(0.50*hsq*self.delvir())**(1.0/6.0)*(1.0 - (2.0*self.omega_l)/(3.0*hsq*self.delvir()))


  
	def mass_func(self,vc, zcoll):
		return (vc/self.coeff_vc_mass())**3
  
	def vc(self,mass,zcoll):
		return self.coeff_vc_mass()*mass**(1.0/3.0)





