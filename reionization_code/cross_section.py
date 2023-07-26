#stand alone cross section fuunctins for different species hence need not to make any class, a separate python file will easily do the job

import math
import numpy as np
from scipy import integrate
from scipy.integrate import quad
from constants import *

def sigma_HI(nu):
	P=2.963
	sigma_0=5.475e-14
	nu_0=4.298e-1/hPlanck_eV
	#P=2.963
	x=nu/nu_0
	return sigma_0*((x-1.0)**2)*(x**(0.5*P-5.5))/(1+math.sqrt(x/32.88))**P

	    	
def sigma_HeI(nu):
	sigma_0=9.492e-16
	P=3.188
	yw=2.039
	y0=0.4434
	y1=2.136
	nu_0=1.361e1/hPlanck_eV
	x=nu/nu_0-y0
	yy=math.sqrt(x**2+y1**2)
	return sigma_0*(yw**2+(x-1.0)**2)*(yy**(0.5*P-5.5))/(1+np.sqrt(yy/1.469))**P
	
		
def sigma_HeII(nu):
	sigma_0=1.369e-14
	P=2.963
	nu_0=1.72/hPlanck_eV
	x=nu/nu_0
	return sigma_0*((x-1.0)**2)*(x**(0.5*P-5.5))/(1+np.sqrt(x/32.88))**P

def sigma_integrand_HI(nu,index):
	return nu**index*sigma_HI(nu)	

def sigma_integral_HI(index,numin,numax):
	ans,err=quad(sigma_integrand_HI,numin,numax,args=(index,),epsabs=1.49e-03, epsrel=1.49e-03)
	return ans
	
def sigma_integrand_HeI(nu,index):
	return nu**index*sigma_HeI(nu)	

def sigma_integral_HeI(index,numin,numax):
	ans,err=quad(sigma_integrand_HeI,numin,numax,args=(index,),epsabs=1.49e-03, epsrel=1.49e-03)
	return ans

def sigma_integrand_HeII(nu,index):
	return nu**index*sigma_HeII(nu)	


def sigma_integral_HeII(index,numin,numax):
	ans,err=quad(sigma_integrand_HeII,numin,numax,args=(index,),epsabs=1.49e-03, epsrel=1.49e-03)
	return ans
