from constants import *
from cooling_heating_coeff import *
import numpy as np
import scipy as sc
from scipy.optimize import fsolve


def compute_species(HI_frac, HeI_frac, HeIII_frac): 
	X_HII=1.0-Y_He-HI_frac
	X_HeII=mp_by_mHe*Y_He-HeI_frac-HeIII_frac
	X_e=X_HII+X_HeII+2.0*HeIII_frac
	X=HI_frac+ X_HII+ HeI_frac+X_HeII+HeIII_frac+X_e
	return X_HII, X_HeII, X_e, X





class Thermal:

	def __init__(self, dtimedz, ombh2):
		self.ombh2=ombh2
		self.rho_b=1.8791e-29*ombh2
		self.dtimedz=dtimedz

	def get_recrates(self, z, region_temp, clumping, Delta_g):
	
		region_recrate=[clumping*(R_HII_e_A(region_temp)+R_HII_e_B(region_temp))*(self.rho_b*Delta_g/mprot)*(1.0+z)**3*self.dtimedz,clumping*(R_HeII_e_A(region_temp)+R_HeII_e_B(region_temp))*(self.rho_b*Delta_g/mprot)*(1.0+z)**3*self.dtimedz,-clumping*(R_HeIII_e_A(region_temp)+R_HeIII_e_B(region_temp))*(self.rho_b*Delta_g/mprot)*(1.0+z)**3*self.dtimedz]
	
		
		region_coolrate=[-clumping*(RC_HII_A(region_temp)+RC_HII_B(region_temp))*((self.rho_b*Delta_g/mprot)*(1.0+z)**3)**2*self.dtimedz/Delta_g,-clumping*(RC_HeII_A(region_temp)+RC_HeII_B(region_temp))*((self.rho_b*Delta_g/mprot)*(1.0+z)**3)**2*self.dtimedz/Delta_g,-clumping*(RC_HeIII_A(region_temp)+RC_HeIII_B(region_temp))*((self.rho_b*Delta_g/mprot)*(1.0+z)**3)**2*self.dtimedz/Delta_g]
	
		return region_recrate[0], region_recrate[1],region_recrate[2], region_coolrate[0], region_coolrate[1], region_coolrate[2]





	def get_ionrates(self, z, Gamma_PI, Gamma_PH, Q):
		
		region_ionrate=[-(Gamma_PI[0]/Q)*self.dtimedz,-(Gamma_PI[1]/Q)*self.dtimedz,(Gamma_PI[2]/Q)*self.dtimedz]

		region_heatrate=[(Gamma_PH[0]/Q)*(self.rho_b/mprot)*(1.0+z)**3*self.dtimedz,(Gamma_PH[1]/Q)*(self.rho_b/mprot)*(1.0+z)**3*self.dtimedz, (Gamma_PH[2]/Q)*(self.rho_b/mprot)*(1.0+z)**3*self.dtimedz]
	
		return region_ionrate[0], region_ionrate[1], region_ionrate[2], region_heatrate[0], region_heatrate[1], region_heatrate[2]


	def update_ionstate(self, z, k, dz, region):
		xp=np.zeros(3)

		dumionstate=region
	
		xp[0]=region['frac']['HI'][k]
		xp[1]=region['frac']['HeI'][k]
		xp[2]=region['frac']['HeIII'][k]
		
		xpsol=sc.optimize.fsolve(self.funcv,xp,args=(-dz, k, dumionstate, xp), xtol=1.49012e-02) #dz_step=-dz=0.2
		
		region['frac']['HI'][k]=xpsol[0]
		region['frac']['HeI'][k]=xpsol[1]
		region['frac']['HeIII'][k]=xpsol[2]
	

		region['X_HII'][k], region['X_HeII'][k], region['X_e'][k], region['X'][k]= compute_species(region['frac']['HI'][k], region['frac']['HeI'][k], region['frac']['HeIII'][k])


		dQdz=region['heatrate']['HI'][k]*region['frac']['HI'][k]+region['heatrate']['HeI'][k]*region['frac']['HeI'][k]+region['heatrate']['HeIII'][k]*region['X_HeII'][k]+region['coolrate']['HI'][k]*region['X_HII'][k]*region['X_e'][k]+region['coolrate']['HeI'][k]*region['X_HeII'][k]*region['X_e'][k]+region['coolrate']['HeIII'][k]*region['frac']['HeIII'][k]*region['X_e'][k]


		if (z > 0.0): 
			dQdz=dQdz+self.dQ_compton_dt(region['T'][k], z)*region['X_e'][k]*self.dtimedz
		
		x1=2.0/(1.0+z)+(1.0/region['X'][k])*(xpsol[0]-xp[0]+xpsol[1]-xp[1]-xpsol[2]+xp[2])/dz 
		x2=2.0*mprot/(3.0*kboltz*self.rho_b*(1.0+z)**3*region['X'][k])*dQdz

		region['T'][k]=(region['T'][k]+dz*x2)/(1.0-dz*x1)

		return region
	

	def funcv(self, x, dz_step, k, dumionstate, x_old):
		dxdz=np.zeros(3)

	
		dumionstate['X_HII'][k],dumionstate['X_HeII'][k],dumionstate['X_e'][k],dumionstate['X'][k]= compute_species(x[0], x[1], x[2])

	
		dxdz=[dumionstate['ionrate']['HI'][k]*x[0] + dumionstate['recrate']['HI'][k]*dumionstate['X_HII'][k]*dumionstate['X_e'][k], dumionstate['ionrate']['HeI'][k]*x[1] + dumionstate['recrate']['HeI'][k]*dumionstate['X_HeII'][k]*dumionstate['X_e'][k], dumionstate['ionrate']['HeIII'][k]*dumionstate['X_HeII'][k] + dumionstate['recrate']['HeIII'][k]*x[2]*dumionstate['X_e'][k]]

	
		return (x[0]-x_old[0]-(-dz_step)*dxdz[0],x[1]-x_old[1]-(-dz_step)*dxdz[1],x[2]-x_old[2]-(-dz_step)*dxdz[2])  

	def dQ_compton_dt(self, T,z):
		return 6.35e-41*self.ombh2*(1.0+z)**7*(2.726*(1.0+z)-T)
