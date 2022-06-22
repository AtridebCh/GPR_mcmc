import numpy as np
import math

from constants import *

class Inhomogenous_Reion:

	def __init__(self,MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, **kwargs):
		if kwargs['call_for_first_time']:
			pass
		else:
			self.sigma=kwargs['sigma']
			self.Delta_V, self.mumean=self.mumean_delta_v(self.sigma)
			self.P_V_norm=self.get_LN_Norm()
			self.dtimedz=kwargs['dtimedz']
		super(Inhomogenous_Reion,self).__init__(MCMC_H0, MCMC_ombh2, MCMC_omch2, sigma_8, MCMC_ns, **kwargs)



	def update_Q(self, z, dz, Q, Qold, sigmaold, dnphotdz, n_e, n_H, R_e_B):
		dtimedz=self.dtimedz
		sigma=self.sigma
		if ((Qold[0] < 1.0) or (Qold[1] < Delta_H_overlap)):
			Q[1]=Delta_H_overlap
			dF_M_dz=(-math.log(Q[1])+0.50*sigma**2)/sigma*self.P_V(sigma,Q[1])*Q[1]**2*(sigma-sigmaold)/dz

			Q[3]=self.F_M(sigma,Q[1])
			Q[2]=self.F_V(sigma,Q[1])
			Q[4]=self.R(sigma,min(Q[1],1e04))
			Q[0]=(Qold[0]+dz*dnphotdz/(n_H*Mpcbycm**3*Q[3]))/(1.0+dz*(R_e_B*dtimedz*Q[4]*n_e*(1.0+z)**3+dF_M_dz)/Q[3])

		else:
			Q[0]=Qold[0]

		   #*****************************************
			flo=0.0
			fhi=1.0
			dxold=fhi-flo
			dx=dxold

			Q[3]=Qold[3]

			Q[1]=self.Delta_i_F_M(sigma,Q[3],Qold[1])
			Q[4]=self.R(sigma,min(Q[1],1e04))
			func1=Q[3]-Qold[3]+dz*(Q[0]*R_e_B*dtimedz*Q[4]*n_e*(1.0+z)**3-dnphotdz/(n_H*Mpcbycm**3))
			dfunc1=1.0+dz*(R_e_B*dtimedz*n_e*(1.0+z)**3)*Q[1]
		

		
			for j in range(1,50):
				if (((Q[3]-fhi)*dfunc1-func1)*((Q[3]-flo)*dfunc1-func1) >= 0.0 or abs(2.0*func1) > abs(dxold*dfunc1)):
					dxold=dx
					dx=0.5*(fhi-flo)
					Q[3]=flo+dx
				else:
					dxold=dx
					dx=func1/dfunc1
					Q[3]=Q[3]-dx
				if (abs(dx) < 1e-08):
					break
				Q[1]=self.Delta_i_F_M(sigma,Q[3],Qold[1])

				Q[4]=self.R(sigma,min(Q[1],1e04))

				func1=Q[3]-Qold[3]+dz*(Q[0]*R_e_B*dtimedz*Q[4]*n_e*(1.0+z)**3-dnphotdz/(n_H*Mpcbycm**3))
				dfunc1=1.0+dz*(R_e_B*dtimedz*n_e*(1.0+z)**3)*Q[1]
				if (func1< 0.0):
					flo=Q[3]
				else:
					fhi=Q[3]
		     
		   #*******************************************
	 
			if (Q[1] < Delta_H_overlap):
				Q[1]=Delta_H_overlap-1e-03
			Q[3]=self.F_M(sigma,Q[1])
			Q[2]=self.F_V(sigma,Q[1])
			Q[4]=self.R(sigma,min(Q[1],1e04))
		if (Q[0] > 1.0):
			Q[0]=1.0
		return Q

	def Delta_i_F_M(self, sigma,F_M_in,Delta_initial):
		Delta_i_F_M_ini=Delta_initial
		MAXIT=10
		if (Delta_initial > 1e6):
			return Delta_i_F_M_ini
	
		func1=self.F_M(sigma,Delta_i_F_M_ini)-F_M_in

		dfunc1=self.P_V(sigma,Delta_i_F_M_ini)*Delta_i_F_M_ini
		if (func1 < 0.0):
			flo=Delta_i_F_M_ini
			fhi=1e06
		else:
			flo=0.0
			fhi=Delta_i_F_M_ini
	  

		dxold=fhi-flo
		dx=dxold

	
		for j in range(0,MAXIT): 
			if (((Delta_i_F_M_ini-fhi)*dfunc1-func1)*((Delta_i_F_M_ini-flo)*dfunc1-func1) >= 0.0 or abs(2.0*func1) > abs(dxold*dfunc1)):
				dxold=dx
				dx=0.50*(fhi-flo)
				Delta_i_F_M_ini=flo+dx
			else:
				dxold=dx
				dx=func1/dfunc1
				Delta_i_F_M_ini=Delta_i_F_M_ini-dx
		   

			if (abs(dx) < 1e-08):
				return Delta_i_F_M_ini

			func1=self.F_M(sigma, Delta_i_F_M_ini)-F_M_in
			dfunc1=self.P_V(sigma, Delta_i_F_M_ini)*Delta_i_F_M_ini
			if (func1 < 0.0):
				flo=Delta_i_F_M_ini
			else:
				fhi=Delta_i_F_M_ini
		   
		return Delta_i_F_M_ini



	def P_V(self, sigma, Delta):
		Delta_V=self.Delta_V
		mumean=self.mumean
		if (Delta <= Delta_V):
			P_V_ini=math.exp(-0.50*((math.log(Delta)-mumean)/sigma)**2)/(math.sqrt(2*math.pi)*Delta*sigma)
		else:
			P_V_ini=(Delta/Delta_V)**betaindex*math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2*math.pi)*Delta_V*sigma)
		return P_V_ini*self.P_V_norm		



	def F_V(self, sigma, Delta_i):
		Delta_V=self.Delta_V
		mumean=self.mumean
		if (Delta_i <= Delta_V):
			F_V_unnormed=0.50*(1.0+math.erf((math.log(Delta_i)-mumean)/(sigma*math.sqrt(2.0))))
		else:
			F_V_unnormed=0.50*(1.0+math.erf((math.log(Delta_V)-mumean)/(sigma*math.sqrt(2.0))))+ \
		        math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2*math.pi)*Delta_V**(betaindex+1.0)*sigma)* \
		        (Delta_i**(betaindex+1.0)-Delta_V**(betaindex+1.0))/(betaindex+1.0)
		
		return self.P_V_norm*F_V_unnormed
	
   
	def R(self, sigma, Delta_i):
		Delta_V=self.Delta_V
		mumean=self.mumean
		
		if (Delta_i <= Delta_V):
			R_unnormed=0.50*math.exp(2.0*(mumean+sigma**2))* \
		        (1.0+math.erf((math.log(Delta_i)-mumean-2.0*sigma**2)/(sigma*math.sqrt(2.0))))
		else:
			R_unnormed=0.50*math.exp(2.0*(mumean+sigma**2))* \
		        (1.0+math.erf((math.log(Delta_V)-mumean-2.0*sigma**2)/(sigma*math.sqrt(2.0))))+ \
		        math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2.0*math.pi)*Delta_V**(betaindex+1.0)*sigma)* \
		        (Delta_i**(betaindex+3.0)-Delta_V**(betaindex+3.0))/(betaindex+3.0)
		
		return self.P_V_norm*R_unnormed
		


	def F_M(self, sigma, Delta_i):
		Delta_V=self.Delta_V
		mumean=self.mumean
		if (Delta_i <= Delta_V):
			F_M_unnormed=0.50*math.exp(mumean+0.50*sigma*sigma)* \
		        (1.0+math.erf((math.log(Delta_i)-mumean-sigma**2)/(sigma*math.sqrt(2.0))))
		else:
			F_M_unnormed=0.50*math.exp(mumean+0.50*sigma*sigma)* \
		        (1.0+math.erf((math.log(Delta_V)-mumean-sigma**2)/(sigma*math.sqrt(2.0))))+ \
		        math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2*math.pi)*Delta_V**(betaindex+1.0)*sigma)* \
		        (Delta_i**(betaindex+2.0)-Delta_V**(betaindex+2.0))/(betaindex+2.0)
		
		return self.P_V_norm*F_M_unnormed


	def mumean_delta_v(self,sigma):
		Delta_V=(0.50*(1.0-math.erf(sigma*(betaindex+1.0)/math.sqrt(2.0)))-\
		     math.exp(-0.50*sigma**2*(betaindex+1.0)**2)/(math.sqrt(2.0*math.pi)*\
		     (betaindex+1.0)*sigma))/ \
		     (0.50*math.exp(sigma**2*(betaindex+1.50))* \
		     (1.0-math.erf(sigma*(betaindex+2.0)/math.sqrt(2.0)))- \
		     math.exp(-0.50*sigma**2*(betaindex+1.0)**2)/(math.sqrt(2.0*math.pi)* \
		     (betaindex+2.0)*sigma))
		mumean=math.log(Delta_V)+sigma**2*(betaindex+1.0)
		return Delta_V, mumean


	def get_LN_Norm(self):
		Delta_V=self.Delta_V
		mumean=self.mumean
		sigma=self.sigma
		P_V_norm=0.50*(1.0+math.erf((math.log(Delta_V)-mumean)/(sigma*math.sqrt(2.0))))- \
		     math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2.0*math.pi)* \
		     (betaindex+1.0)*sigma)
		     
		return 1.0/P_V_norm



