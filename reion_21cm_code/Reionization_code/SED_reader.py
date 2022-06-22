import numpy as np
from constants import *
from cross_section import *

def get_stellar_SED(SEDFile):

	dnphotdm=np.zeros(3)
	sigma_PH=np.zeros(3)
	sigma_PI=np.zeros(3)


	data_nu_dndm=np.loadtxt(SEDFile)
	Nu=data_nu_dndm[:,0]
	dNdM=data_nu_dndm[:,1]
	nu_old=Nu[0]
	dndm_old=dNdM[0]

	for nu,dndm in zip(Nu[1:],dNdM[1:]):
		if nu>=nu_HII and nu<nu_HeII:
			dnphotdm[0]=dnphotdm[0]+(nu_old-nu)*0.50*(dndm+dndm_old)
		if nu>=nu_HeII and nu <nu_HeIII:
			dnphotdm[1]=dnphotdm[1]+(nu_old-nu)*0.50*(dndm+dndm_old)
		if nu>=nu_HeIII:
			dnphotdm[2]=dnphotdm[2]+(nu_old-nu)*0.50*(dndm+dndm_old)
		
					
		if nu>=nu_HII:
			if nu>=nu_HeIII:
				sigma_PI[0]=sigma_PI[0]+sigma_HI(nu)*0.50*(dndm+dndm_old)*(nu_old-nu)*(nu/nu_HeIII)**1.5 
				sigma_PH[0]=sigma_PH[0]+sigma_HI(nu)*hPlanck*(nu-nu_HII)*0.50*(dndm+dndm_old)*(nu_old-nu)*(nu/nu_HeIII)**1.5
			else:
				sigma_PI[0]=sigma_PI[0]+sigma_HI(nu)*0.50*(dndm+dndm_old)*(nu_old-nu)			
				sigma_PH[0]=sigma_PH[0]+sigma_HI(nu)*hPlanck*(nu-nu_HII)*0.50*(dndm+dndm_old)*(nu_old-nu)
		         	
		if nu>=nu_HeII:
			if nu>=nu_HeIII:
				sigma_PI[1]=sigma_PI[1]+sigma_HeI(nu)*0.50*(dndm+dndm_old)*(nu_old-nu)*(nu/nu_HeIII)**1.5 
				sigma_PH[1]=sigma_PH[1]+sigma_HeI(nu)*hPlanck*(nu-nu_HeII)*0.50*(dndm+dndm_old)*(nu_old-nu)*(nu/nu_HeIII)**1.5
			else:
				sigma_PI[1]=sigma_PI[1]+sigma_HeI(nu)*0.50*(dndm+dndm_old)*(nu_old-nu)			
				sigma_PH[1]=sigma_PH[1]+sigma_HeI(nu)*hPlanck*(nu-nu_HeII)*0.50*(dndm+dndm_old)*(nu_old-nu) 
		         	
		if nu>=nu_HeIII:
			sigma_PI[2]=sigma_PI[2]+sigma_HeII(nu)*0.50*(dndm+dndm_old)*(nu_old-nu)*(nu/nu_HeIII)**1.5
			sigma_PH[2]=sigma_PH[2]+sigma_HeII(nu)*hPlanck*(nu-nu_HeIII)*0.50*(dndm+dndm_old)*(nu_old-nu)*(nu/nu_HeIII)**1.5
		         	
	
		nu_old=nu
		dndm_old=dndm

	sigma_PH[0]=sigma_PH[0]/(dnphotdm[0]*hPlanck*nu_HII)
	sigma_PH[1]=sigma_PH[1]/(dnphotdm[1]*hPlanck*nu_HeII)
	sigma_PH[2]=sigma_PH[2]/(dnphotdm[2]*hPlanck*nu_HeIII)
		   	
	sigma_PI[0]=sigma_PI[0]/dnphotdm[0]
	sigma_PI[1]=sigma_PI[1]/dnphotdm[1]
	sigma_PI[2]=sigma_PI[2]/dnphotdm[2]   	
	return dnphotdm[0], dnphotdm[1], dnphotdm[2], sigma_PI[0], sigma_PI[1], sigma_PI[2], sigma_PH[0], sigma_PH[1], sigma_PH[2]


def get_quasar_SED():

	sigma_PH=np.zeros(3)
	sigma_PI=np.zeros(3)


	sigma_PH[0]=(1.57/nu_HII)*\
		(sigma_integral_HI(-1.57,nu_HII,nu_HeIII)*nu_HII**1.57- \
		sigma_integral_HI(-2.57,nu_HII,nu_HeIII)*nu_HII**2.57+ \
		sigma_integral_HI(-0.07,nu_HeIII,nu_HeIII*40.)*nu_HII**1.57*nu_HeIII**(-1.5)-\
		sigma_integral_HI(-1.07,nu_HeIII,nu_HeIII*40.)*nu_HII**2.57*nu_HeIII**(-1.5))/(1.-(nu_HII/(nu_HeII))**1.57)
		     
	sigma_PI[0]=1.57*\
	(sigma_integral_HI(-2.57,nu_HII,nu_HeIII)*nu_HII**1.570+\
	sigma_integral_HI(-1.07,nu_HeIII,nu_HeIII*40.0)*nu_HII**1.570*nu_HeIII**(-1.50))/(1.0-(nu_HII/(nu_HeII))**1.57)
		     
		     
	sigma_PH[1]=(1.570/nu_HeII)*\
		(sigma_integral_HeI(-1.57,nu_HeII,nu_HeIII)*nu_HII**1.570-\
		sigma_integral_HeI(-2.57,nu_HeII,nu_HeIII)*nu_HII**1.570*nu_HeII+\
		sigma_integral_HeI(-0.07,nu_HeIII,nu_HeIII*40.0)*nu_HII**1.570*nu_HeIII**(-1.5)-\
		sigma_integral_HeI(-1.07,nu_HeIII,nu_HeIII*40.0)*nu_HII**1.570*nu_HeII*nu_HeIII**(-1.50))/((nu_HII/(nu_HeII))**1.570-(nu_HII/(nu_HeIII))**1.570)
	sigma_PI[1]=1.570*\
		(sigma_integral_HeI(-2.570,nu_HeII,nu_HeIII)*nu_HII**1.57+
		sigma_integral_HeI(-1.070,nu_HeIII,nu_HeIII*40.0)*nu_HII**1.070*nu_HeIII**(-1.50))/((nu_HII/(nu_HeII))**1.570-(nu_HII/(nu_HeIII))**1.570)

	sigma_PH[2]=(1.57/nu_HeIII)*\
		(sigma_integral_HeII(-0.070,nu_HeIII,nu_HeIII*40.0)*nu_HII**1.57*nu_HeIII**(-1.50)-\
		sigma_integral_HeII(-1.070,nu_HeIII,nu_HeIII*40.0)*nu_HII**1.57*nu_HeIII**(-0.50))/(nu_HII/(nu_HeIII))**1.570
	sigma_PI[2]=1.570*sigma_integral_HeII(-1.070,nu_HeIII,nu_HeIII*40.0)*nu_HII**1.570*nu_HeIII**(-1.50)/(nu_HII/(nu_HeIII))**1.57
		     
	#print(sigma_PH[0],sigma_PH[1],sigma_PH[2],sigma_PI[0],sigma_PI[1],sigma_PI[2] )	     
	return sigma_PH[0],sigma_PH[1],sigma_PH[2],sigma_PI[0],sigma_PI[1],sigma_PI[2] 
