import numpy as np
import scipy as sc
#import numba as nb
#from numba import jit,float64,cfunc
from scipy import integrate
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
import collections
import time
import math
####### Input Parameter ##################################
#output_root = Planck

#cdef double omeag_r,omega_m,omega_k,omega_l,domega_ini,h,omega_b_h2,Y_He,sigma_8,n_s,dn_dlnk,rho_c,Delta_H_overlap,e_sf_II,e_sf_III,e_QSO,omega_b,gamma,betaindex,esc_popII,esc_popIII,lambda_0,feedback,chemfeedback
#cdef double vc_min,dz_step,zstart,zend,kboltz,mprot,c,hPlanck,hPlanck_eV,Mpcbycm,yrbysec,mp_by_mHe,T_HI,T_HeI,T_HeII,E_HII,E_HeII,E_HeIII,nu_HII,nu_HeII,nu_HeIII,nu_alpha,nu_beta,nu_gamma,nu_delta

start_time = time.time()
def running_code(MCMC_H0,MCMC_ombh2,MCMC_omch2,MCMC_sigma8,MCMC_ns,MCMC_esc_PopII,MCMC_esc_PopIII,MCMC_lambda_0):
	h=MCMC_H0/100.0
	omega_r=0.0
	omega_m = (MCMC_ombh2+MCMC_omch2)/h**2
	omega_k = 0.0
	omega_l=1-omega_k-omega_m
	domega_ini=1.0
	omega_b_h2 = MCMC_ombh2
	Y_He = 0.2453
	sigma_8 = MCMC_sigma8
	n_s = MCMC_ns
	dn_dlnk= 0.0
	rho_c=2.7755e11
	Delta_H_overlap = 59.21
	e_sf_II = 1.0
	e_sf_III = 1.0
	e_QSO = 0.36	
	omega_b=omega_b_h2/h**2
	gamma=omega_m*h
	betaindex=-2.2
	which_pop = 'both' #other options are 'pop3' and 'both'
	esc_PopII = MCMC_esc_PopII
	esc_PopIII = MCMC_esc_PopIII
	#PopIII_IMF = 'salpeter'
	iflambda_0 = True
	lambda_0 = MCMC_lambda_0
	feedback = 1
	chemfeedback = 1
	vc_min = 13.3


	dz_step = 0.2
	zstart = 2.0
	zend = 25.0
	kboltz=1.38e-16
	mprot=1.67e-24
	c=3e5
	hPlanck=6.6260755e-27
	hPlanck_eV=4.1356692e-15
	Mpcbycm=3.0857e24
	yrbysec=365.0*24.0*3600
	mp_by_mHe=0.25180
	T_HI=157807.0
	T_HeI=285335.0
	T_HeII=631515.0
	E_HII=13.59840
	E_HeII=24.58740
	E_HeIII=54.4160
	nu_HII=E_HII/hPlanck_eV
	nu_HeII=E_HeII/hPlanck_eV
	nu_HeIII=E_HeIII/hPlanck_eV
	nu_alpha=c/1215.670*1e13
	nu_beta=c/1025.720*1e13
	nu_gamma=c/972.540*1e13
	nu_delta=c/949.740*1e13

	M_sun_g = 1.98892e33
	Mpc_cm = 3.08568025e24
	G_const_Mpc_Msun_s = M_sun_g * (6.673e-8) / Mpc_cm**3.

	dz=-0.2
	omega_bhsq=omega_b_h2
	n=int(abs(((zend-zstart)/dz_step)))+1

	Z=np.linspace(zend,zstart,n)

	##############################################################

	####################################### cosmobasic ###########################
	 
	#@jit("float64(float64)")
	def Hubble_dist(z):
		return (math.sqrt(omega_k*(1.0+z)**2+omega_m*(1.0+z)**3+omega_r*(1.0+z)**4+omega_l))**-1

	#@jit((float64(float64)), nopython=True, fastmath=True)
	def func(z):
		return Hubble_dist(z)/(1.0+z)

	#@jit((float64(float64)), nopython=True, fastmath=True)
	def omega_z(z):	
		return Hubble_dist(z)**2*(1.0+z)**3*omega_m


	###################################################################################################

	################################################## normalize ######################################
	#@jit((float64(float64)), nopython=True,nogil=True, fastmath=True)
	def transfun(q):
		aa=6.40
		bb=3.0
		cc=1.70
		power=1.13
		return 1.0/((1.0+(aa*q +(bb*q)**1.50 +(cc*q)**2)**power)**(1.0/power))
	
	#@jit((float64(float64)), nopython=True,nogil=True, fastmath=True)
	def pspec(k):
		gamma=omega_m*h
		gam_h=gamma*h
		K0=0.05
		n0=n_s+0.5*dn_dlnk*math.log(k/K0)
		return (k**n0)*transfun(k/gam_h)*transfun(k/gam_h)

	#@jit((float64(float64)), nopython=True,nogil=True, fastmath=True)
	def window(x):
		return (3.0*((math.sin(x)/x**3)-(math.cos(x)/x**2)))**2

	#@jit((float64(float64)), nopython=True,nogil=True)	
	def pswin(k):
		r=8.0/h
		return pspec(k)*k**2*window(k*r)

	'''
	def klims(r):
		logk = np.arange(-20., 20., 0.1)
		integrand=sigmasq_integrand_log(logk,r)

		maxintegrand = np.max(integrand)
		factor = 1.e-4
		highmask = integrand > maxintegrand * factor
		while highmask.ndim > logk.ndim:
		    highmask = numpy.logical_or.reduce(highmask)

		mink = numpy.min(logk[highmask])
		maxk = numpy.max(logk[highmask])

		return mink, maxk
		
	'''  	
	#@jit((float64(float64,float64)), nopython=True)	
	def norm_sourav(sigma,h):
		#ans,err=quad(pswin,0.0,np.inf)
		return sigma**2*2*math.pi**2/quad(pswin,0.0,np.inf,epsabs=1.49e-09, epsrel=1.49e-03)[0]

	def sigma_sourav(x):
		#ans,err=quad(sigmasq_integrand,0.0,np.inf,args=(x,))
		return math.sqrt(quad(sigmasq_integrand,0.0,np.inf,args=(x,),epsabs=1.49e-09, epsrel=1.49e-05)[0]/(2.0*np.pi**2))
	
	#@jit((float64(float64,float64)), nopython=True,nogil=True, fastmath=True)	
	def sigmasq_integrand(k,r):
		#return (3.0*(math.sin(k*r)-k*r*math.cos(k*r)))**2*pspec(k)/(k*k*r**3)**2
		return pspec(k)*k**2*window(k*r)

	def sigmasq_b_sourav(x,z,T,mu):
		#ans,err=quad(integrand_sigmasq_b,0.0,np.inf,args=(x,T,z,mu))	
		a=norm_sourav(sigma_8,h)
		return a*d(z)**2*quad(integrand_sigmasq_b,0.0,np.inf,args=(x,T,z,mu),epsabs=1.49e-03, epsrel=1.49e-03)[0]/(2.0*math.pi**2)	

	
	#@jit((float64(float64,float64,float64)), nopython=True,nogil=True, fastmath=True)		
	def xbsq(T,z,mu):
		return (2.0*kboltz*T/(3.0*mu*mprot*omega_m*h**2*1e14))/(1.0+z)

	
	#@jit((float64(float64,float64,float64,float64)), nopython=True)	
	def pspec_b(k,T,z,mu):
		return pspec(k)/(1.0+xbsq(T,z,mu)*k*k)**2

	
	#@jit((float64(float64,float64,float64,float64,float64)), nopython=True,nogil=True, fastmath=True)
	def integrand_sigmasq_b(k,x,T,z,mu):
		if (x<1e-07):
			return (1.0-(k*k*x*x/5.0))*k*k*pspec_b(k,T,z,mu)
		else:
			return (3.0*(math.sin(k*x)-k*x*math.cos(k*x)))**2*pspec_b(k,T,z,mu)/(k*k*x**3)**2

	#@jit((float64(float64)), nopython=True,nogil=True, fastmath=True) 
	def d(z):

		if ((omega_l==0.0) and (omega_m==1.0)):
			dinv=1.0+z
		elif ((omega_l==0.0) and (omega_m!=1.0)):
			dinv=1.0+2.5*omega_m*z/(1.0+1.5*omega_m)
		else:
			dinv=(1.0+((1.0+z)**3-1.0)/(1.0+0.45450*(omega_l/omega_m)))**(1.0/3.0)
		return 1.0/dinv

		  
	#@jit((float64(float64)), nopython=True,nogil=True, fastmath=True)    
	def probdist(nu):
		return math.sqrt(2.0/math.pi)*np.exp(-nu**2/2.0)



	#############################################################################################################################################

		
	#################################################### all basic func ########################################################################

	def get_stellar_SED(SEDFile,dnphotdm,sigma_PI,sigma_PH):
		#cdef np.ndarray[np.float64_t, ndim=2] data_nu_dndm
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
		   	
		return dnphotdm[0],dnphotdm[1],dnphotdm[2],sigma_PH[0],sigma_PH[1],sigma_PH[2],sigma_PI[0],sigma_PI[1],sigma_PI[2]


	def get_quasar_SED(sigma_PI,sigma_PH):
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
		     
		     
		return sigma_PH[0],sigma_PH[1],sigma_PH[2],sigma_PI[0],sigma_PI[1],sigma_PI[2]

	#@jit((float64(float64)), nopython=True,nogil=True, fastmath=True)	
	def sigma_HI(nu):
		P=2.963
		sigma_0=5.475e-14
		nu_0=4.298e-1/hPlanck_eV
		#P=2.963
		x=nu/nu_0
		return sigma_0*((x-1.0)**2)*(x**(0.5*P-5.5))/(1+math.sqrt(x/32.88))**P

	#@jit((float64(float64)), nopython=True,nogil=True, fastmath=True)	    	
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
	
	
	#@jit((float64(float64)), nopython=True,nogil=True, fastmath=True)	
	def sigma_HeII(nu):
		sigma_0=1.369e-14
		P=2.963
		nu_0=1.72/hPlanck_eV
		x=nu/nu_0
		return sigma_0*((x-1.0)**2)*(x**(0.5*P-5.5))/(1+np.sqrt(x/32.88))**P


		
	#@jit((float64(float64,float64)), nopython=True,nogil=True, fastmath=True)	
	def sigma_integrand_HI(nu,index):
		return nu**index*sigma_HI(nu)	

	def sigma_integral_HI(index,numin,numax):
		ans,err=quad(sigma_integrand_HI,numin,numax,args=(index,),epsabs=1.49e-03, epsrel=1.49e-03)
		return ans
	#@jit((float64(float64,float64)), nopython=True,nogil=True, fastmath=True)
	def sigma_integrand_HeI(nu,index):
		return nu**index*sigma_HeI(nu)	

	def sigma_integral_HeI(index,numin,numax):
		#print(numax,numin)
		ans,err=quad(sigma_integrand_HeI,numin,numax,args=(index,),epsabs=1.49e-03, epsrel=1.49e-03)
		return ans
	#@jit((float64(float64,float64)), nopython=True,nogil=True, fastmath=True)
	def sigma_integrand_HeII(nu,index):
		return nu**index*sigma_HeII(nu)	


	def sigma_integral_HeII(index,numin,numax):
		#print(numax,numin)
		ans,err=quad(sigma_integrand_HeII,numin,numax,args=(index,),epsabs=1.49e-03, epsrel=1.49e-03)
		return ans





	#print(sigma_integral_HeI(-1.07,constants['nu_HeIII'],constants['nu_HeIII']*40.0))
	#@jit((float64(float64,float64,float64,float64,float64,float64)), nopython=True,nogil=True, fastmath=True)     	
	def lumfun_integrand(lb,z,logphistar,loglstar,gamma1,gamma2):
		x=lb
		psi=10.0**logphistar/(x**gamma1+x**gamma2)
		quasar_lumfun=10.0**loglstar*psi/(math.log(10.0)*lb)
		lumfun_integrand=quasar_lumfun*lb
		lstarby10=10.0**loglstar*1e-10
		lband=8.99833*(x*lstarby10)**-0.0115970+6.24800*(x*lstarby10)**-0.370587
		return lumfun_integrand/lband

	def lumfun_integral(z):
		P0,P1,P2,P3,P4,P5,P6,P7,P8,P9=-4.8250643, 13.035753, 0.63150872, -11.763560, -14.249833, 0.41698725, -0.62298947, 2.1744386,  1.4599393, -0.79280099
		if z>12.0:
			return 0.0
		betamin=1.30
		xi=math.log10((1.0+z)/3.0)
		logphistar=P0
		loglstar=P1+P2*xi+P3*xi*xi+P4*xi*xi*xi
		gamma1=P5*10.0**(P6*xi)
		gamma2=2.0*P7/(10.0**(P8*xi)+10.0**(P9*xi))
		gamma2=max(betamin,gamma2)
		#ans,err=quad(lumfun_integrand,0.0,10000,args=(z,logphistar,loglstar,gamma1,gamma2))
		return quad(lumfun_integrand,0.0,np.inf,args=(z,logphistar,loglstar,gamma1,gamma2),epsabs=1.49e-03, epsrel=1.49e-03)[0]

	#@nb.jit(nb.types.UniTuple(nb.float64,3)(float64),nopython=True,nogil=True, fastmath=True)
	#@jit(float64(float64,float64))
	def get_LN_Norm(sigma):
		Delta_V=(0.50*(1.0-math.erf(sigma*(betaindex+1.0)/math.sqrt(2.0)))-\
		     math.exp(-0.50*sigma**2*(betaindex+1.0)**2)/(math.sqrt(2.0*math.pi)*\
		     (betaindex+1.0)*sigma))/ \
		     (0.50*math.exp(sigma**2*(betaindex+1.50))* \
		     (1.0-math.erf(sigma*(betaindex+2.0)/math.sqrt(2.0)))- \
		     math.exp(-0.50*sigma**2*(betaindex+1.0)**2)/(math.sqrt(2.0*math.pi)* \
		     (betaindex+2.0)*sigma))
		mumean=math.log(Delta_V)+sigma**2*(betaindex+1.0)
		P_V_norm=0.50*(1.0+math.erf((math.log(Delta_V)-mumean)/(sigma*math.sqrt(2.0))))- \
		     math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2.0*math.pi)* \
		     (betaindex+1.0)*sigma)
		     
		P_V_norm=1.0/P_V_norm
			
		return Delta_V, mumean, P_V_norm
		


	   
	#@jit(float64(float64,float64,float64,float64,float64),nopython=True,nogil=True, fastmath=True)
	def F_V(sigma, Delta_i,Delta_V,mumean,P_V_norm):
		if (Delta_i <= Delta_V):
			F_V_unnormed=0.50*(1.0+math.erf((math.log(Delta_i)-mumean)/(sigma*math.sqrt(2.0))))
		else:
			F_V_unnormed=0.50*(1.0+math.erf((math.log(Delta_V)-mumean)/(sigma*math.sqrt(2.0))))+ \
		        math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2*math.pi)*Delta_V**(betaindex+1.0)*sigma)* \
		        (Delta_i**(betaindex+1.0)-Delta_V**(betaindex+1.0))/(betaindex+1.0)
		
		return P_V_norm*F_V_unnormed
	
	#@jit(float64(float64,float64,float64,float64,float64),nopython=True,nogil=True, fastmath=True)    
	def R(sigma,Delta_i,Delta_V,mumean,P_V_norm):
		if (Delta_i <= Delta_V):
			R_unnormed=0.50*math.exp(2.0*(mumean+sigma**2))* \
		        (1.0+math.erf((math.log(Delta_i)-mumean-2.0*sigma**2)/(sigma*math.sqrt(2.0))))
		else:
			R_unnormed=0.50*math.exp(2.0*(mumean+sigma**2))* \
		        (1.0+math.erf((math.log(Delta_V)-mumean-2.0*sigma**2)/(sigma*math.sqrt(2.0))))+ \
		        math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2.0*math.pi)*Delta_V**(betaindex+1.0)*sigma)* \
		        (Delta_i**(betaindex+3.0)-Delta_V**(betaindex+3.0))/(betaindex+3.0)
		
		return P_V_norm*R_unnormed
		

	#@jit(float64(float64,float64,float64,float64,float64),nopython=True,nogil=True, fastmath=True)
	def F_M(sigma, Delta_i,Delta_V,mumean,P_V_norm):
		if (Delta_i <= Delta_V):
			F_M_unnormed=0.50*math.exp(mumean+0.50*sigma*sigma)* \
		        (1.0+math.erf((math.log(Delta_i)-mumean-sigma**2)/(sigma*math.sqrt(2.0))))
		else:
			F_M_unnormed=0.50*math.exp(mumean+0.50*sigma*sigma)* \
		        (1.0+math.erf((math.log(Delta_V)-mumean-sigma**2)/(sigma*math.sqrt(2.0))))+ \
		        math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2*math.pi)*Delta_V**(betaindex+1.0)*sigma)* \
		        (Delta_i**(betaindex+2.0)-Delta_V**(betaindex+2.0))/(betaindex+2.0)
		
		return P_V_norm*F_M_unnormed
		
	#@jit(nb.types.UniTuple(float64,4)(nb.float64[:,:]),nopython=True,nogil=True, fastmath=True)
	def compute_species(region): #region is a 2d array
		region[6][0]=1.0-Y_He-region[0][0]
		region[7][0]=mp_by_mHe*Y_He-region[0][1]-region[0][2]
		region[8][0]=region[6][0]+region[7][0]+2.0*region[0][2]
		region[9][0]=region[0][0]+region[6][0]+region[0][1]+region[7][0]+region[0][2]+region[8][0]
		return region[6][0],region[7][0],region[8][0],region[9][0]
	 
	
	#@jit(float64(float64),nopython=True,nogil=True)	
	def vcmin_pop3_neut(z):
		return vc_min
		
	#@jit(float64(float64,float64,float64),nopython=True,nogil=True, fastmath=True)
	def vcmin_pop3_ion(z,T,X):
		ans=math.sqrt(2.0*kboltz*T*X/mprot)*1e-05
		if (ans < vcmin_pop3_neut(z)):
			return vcmin_pop3_neut(z)
		else:
			return ans
		
	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def vcmin_pop2_neut(z):
		return vc_min
		   
	#@jit(float64(float64,float64,float64),nopython=True,nogil=True, fastmath=True)
	def vcmin_pop2_ion(z,T,X):
		ans=math.sqrt(2.0*kboltz*T*X/mprot)*1e-05
		if (ans < vcmin_pop2_neut(z)):
			#print('vcmin',vcmin_pop2_neut(z))
			return vcmin_pop2_neut(z)
		else:
			return ans

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def vc_max(z):
		return 1e09

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def delvir(zcoll):
		x=omega_z(zcoll)-1.0
		return 18.0*math.pi**2+82.0*x-39.0*x*x


	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def coeff_vc_mass(zcoll):
		hsq=1.0/(Hubble_dist(zcoll))**2
		return (3.504/(1e08**(1.0/3.0)))*(0.50*hsq*delvir(zcoll))**(1.0/6.0)*(1.0 - (2.0*omega_l)/(3.0*hsq*delvir(zcoll)))


		
	#@jit(float64(float64,float64),nopython=True,nogil=True, fastmath=True)    
	def vc(mass,zcoll):
		return coeff_vc_mass(zcoll)*mass**(1.0/3.0)


	def fpop3(zcoll,m,logM,logsig):
		if (e_sf_III < 1e-06):
			return 0.0
		if (chemfeedback== 0):
			return 1.0
		#print(m)
		mass_m=10.0**m
		#print(mass_m[3])
		mass_min=mass(vcmin_pop3_neut(zcoll),zcoll)/h
		#print(len(mass_min))
		factor=1e06
		mass_form=mass_m-factor
		m_form=np.log10(mass_form)
		m_min=np.log10(mass_min)
		spl=splrep(logM,logsig)
		sigma_m=splev(m,spl)
		sigma_m=10.0**sigma_m
		   
		sigma_min=splev(m_min,spl)
		sigma_min=10.0**sigma_min

		sigma_form=splev(m_form,spl)
		sigma_form=10.0**sigma_form

		arg=np.sqrt(abs((sigma_form-sigma_m)/(sigma_min-sigma_form)))
		arg=np.where(arg<1e-05,arg,2.0*np.arctan(arg)/np.pi)
		return arg
	#@jit(float64(float64),nopython=True,nogil=True) 
	def fbbym(m):
		if ((feedback == 0) or (feedback == 1)):
			return 1.0
	


	def mass_integrand_pop2(nu,zmass,delta_c,logM,logsig):
		a=norm_sourav(sigma_8,h)
		#delta_c=1.69/a
		sig=delta_c/(nu*d(zmass))
		spl=splrep(np.sort(logsig),logM[np.argsort(logsig)])
		#f1=interp1d(logsig,logM,kind='cubic',fill_value='extrapolate')
	
		logM_new=splev(np.log10(sig),spl)
		#print(logM_new[3])
		mass_integrand_pop2_ans=probdist(nu)*(nu*nu-1.0)
		return mass_integrand_pop2_ans*(1.0-fpop3(zmass,logM_new,logM,logsig))


	def mass_integral_pop2(zcoll,vcmin,vcmax,delta_c,logM,logsig):
		zmass=zcoll
		if (abs(vcmin-vcmax) < 1e-08):
			return 0.0
		elif (vcmin>vcmax):
			return 0.0
		   
		mmin=mass(vcmin,zcoll)/h
		numin=nu_parameter(mmin,zcoll,delta_c)
		nu_array=np.linspace(numin,10.0,2000)
		dnu=nu_array[1]-nu_array[0]
		#print(numin)
		#print('mmin',mmin,numin)
		if (vcmax > 1e05):
			ans=sc.integrate.simps(mass_integrand_pop2(nu_array,zmass,delta_c,logM,logsig),dx=dnu)
			return ans    
		   
		else:
			mmax=mass(vcmax,zcoll)/h
			numax=nu_parameter(mmax,zcoll,delta_c)
			ans,err=quad(mass_integrand_pop2,numin,numax,args=(zmass,delta_c,logM,logsig),epsabs=1.49e-03, epsrel=1.49e-03)
			#print('here')
			return ans
		


	def mass_integrand_pop3(nu,zmass,delta_c,logM,logsig):
		#a=normalize.norm(cdict['sigma_8'])
		#delta_c=1.69/a
		sig=delta_c/(nu*d(zmass))
		#logM,logsig=interpolating_mass_from_sigma()
		spl=splrep(np.sort(logsig),logM[np.argsort(logsig)])
		logM_new=splev(np.log10(sig),spl)
		mass_integrand_pop3_ans=probdist(nu)*(nu*nu-1.0)
		return mass_integrand_pop3_ans*fpop3(zmass,logM_new,logM,logsig)




	def mass_integral_pop3(zcoll,vcmin,vcmax,delta_c,logM,logsig):
		zmass=zcoll
		if (abs(vcmin-vcmax) < 1e-08):
			return 0.0
		elif (vcmin>vcmax):
			return 0.0
		   
		mmin=mass(vcmin,zcoll)/h
		numin=nu_parameter(mmin,zcoll,delta_c)
		nu_array=np.linspace(numin,10.0,2000)
		dnu=nu_array[1]-nu_array[0]
		#print('numin',numin)
		if (vcmax > 1e05):
			ans=sc.integrate.simps(mass_integrand_pop3(nu_array,zmass,delta_c,logM,logsig),dx=dnu)
			#print('ans',ans)
			if ans<0.0:
				ans=0.0  
		else:
			mmax=mass(vcmax,zcoll)/h
			numax=nu_parameter(mmax,zcoll,delta_c)
			ans,err=quad(mass_integrand_pop3,numin,numax,args=(zmass,delta_c,logM,logsig),epsabs=1.49e-03, epsrel=1.49e-03)
			if ans<0.0:
				ans=0.0
		#print(ans)
		return ans
	
	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def vcmax_func(z):
		return 1e09
	
	def setspline_sigma():
		len_R=100
		rho_0=rho_c*h**2*omega_m
		logM=np.linspace(0.2,20.0,int(20.0/0.2))
		Rcube=3.0*(10**logM)/(4.0*math.pi*rho_0)
		R=Rcube**(1/3.0)
		logsig=np.zeros(len(R))
		#len_R=100
		#logsig=np.log10(normalize.sigma_sourav(R))
		for i in range (len_R):
			logsig[i]=math.log10(sigma_sourav(R[i]))
		return logM,logsig

	#@jit(float64(float64,float64),nopython=True,nogil=True, fastmath=True) 	
	def mass(vc,zcoll):
		#print('coeff',coeff_vc_mass(zcoll))
		return (vc/coeff_vc_mass(zcoll))**3


	def nu_parameter(mass,z,delta_c):
		rho_0=rho_c*omega_m*h**2
		x=(3.0*mass/(4.0*math.pi*rho_0))**(1.0/3.0)
		return delta_c/(sigma_sourav(x)*d(z))

	#@jit(float64(float64),nopython=True,nogil=True)    	
	def R_HII_e_A(T):
		#lambda_HI=2.0*T_HI/T
		#ans=1.269e-13*lambda_HI**1.503/(1.0+(lambda_HI/0.5220)**0.470)**1.923
		#ans=0.0
		return 0.0



	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def RC_HII_A(T):
		#lambda_HI=2.0*T_HI/T
		#ans=1.778e-29*T*lambda_HI**1.965/(1.0+(lambda_HI/0.5410)**0.502)**2.697
		#ans=0.0
		return 0.0

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def R_HeII_e_A(T):
		#ans=0.0
		return 0.0

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def RC_HeII_A(T):
		return 0.0

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def R_HeIII_e_A(T):
		return 0.0


	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)	
	def RC_HeIII_A(T):
		ans=0.0
		return ans
	
	
	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def R_HII_e_B(T):
		lambda_HI=2.0*T_HI/T
		#print(T)
		#print('lambda_HI',lambda_HI)
		return 2.753e-14*lambda_HI**1.5/(1.0+(lambda_HI/2.740)**0.407)**2.242

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def RC_HII_B(T):
		lambda_HI=2.0*T_HI/T
		return 3.435e-30*T*lambda_HI**1.97/(1.0+(lambda_HI/2.250)**0.376)**3.72




	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def R_HeII_e_B(T):
		lambda_HeI=2.0*T_HeI/T
		return 1.26e-14*lambda_HeI**0.75

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def RC_HeII_B(T):
		return kboltz*T*R_HeII_e_B(T)

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def R_HeIII_e_B(T):
		lambda_HeII=2.0*T_HeII/T
		return 2.0*2.753e-14*lambda_HeII**1.5/(1.0+(lambda_HeII/2.74)**0.407)**2.242

	#@jit(float64(float64),nopython=True,nogil=True, fastmath=True)
	def RC_HeIII_B(T):
		lambda_HeII=2.0*T_HeII/T
		return 8.0*3.435e-30*T*lambda_HeII**1.97/(1.0+(lambda_HeII/2.25)**0.376)**3.72




		
	#@nb.jit(nb.types.Tuple((nb.float64[:],nb.float64[:]))(float64,nb.float64[:,:],float64,float64),nopython=True,nogil=True, fastmath=True)	
	def get_recrates(z,region,clumping,Delta_g):
	
		rho_b=1.8791e-29*omega_b_h2
		dtimedz=-(9.777e9*yrbysec)*func(z)/h
	
		region[1]=[clumping*(R_HII_e_A(region[5][0])+R_HII_e_B(region[5][0]))*(rho_b*Delta_g/mprot)*(1.0+z)**3*dtimedz,clumping*(R_HeII_e_A(region[5][0])+R_HeII_e_B(region[5][0]))*(rho_b*Delta_g/mprot)*(1.0+z)**3*dtimedz,-clumping*(R_HeIII_e_A(region[5][0])+R_HeIII_e_B(region[5][0]))*(rho_b*Delta_g/mprot)*(1.0+z)**3*dtimedz]
	
		
		region[3]=[-clumping*(RC_HII_A(region[5][0])+RC_HII_B(region[5][0]))*((rho_b*Delta_g/mprot)*(1.0+z)**3)**2*dtimedz/Delta_g,-clumping*(RC_HeII_A(region[5][0])+RC_HeII_B(region[5][0]))*((rho_b*Delta_g/mprot)*(1.0+z)**3)**2*dtimedz/Delta_g,-clumping*(RC_HeIII_A(region[5][0])+RC_HeIII_B(region[5][0]))*((rho_b*Delta_g/mprot)*(1.0+z)**3)**2*dtimedz/Delta_g]
	
		return region[1],region[3]

	#@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(float64, nb.float64[:,:],float64[:],float64[:],float64),nopython=True,nogil=True, fastmath=True)
	def get_ionrates(z,region,Gamma_PI,Gamma_PH,Q):
	
		rho_b=1.8791e-29*omega_b_h2
		dtimedz=-(9.777e9*yrbysec)*func(z)/h
	
		region[2]=[-(Gamma_PI[0]/Q)*dtimedz,-(Gamma_PI[1]/Q)*dtimedz,(Gamma_PI[2]/Q)*dtimedz]
	
		region[4]=[(Gamma_PH[0]/Q)*(rho_b/mprot)*(1.0+z)**3*dtimedz,(Gamma_PH[1]/Q)*(rho_b/mprot)*(1.0+z)**3*dtimedz,(Gamma_PH[2]/Q)*(rho_b/mprot)*(1.0+z)**3*dtimedz]
	
		return region[2],region[4]


	def update_ionstate(z,dz,region):
		xp=np.zeros(3)
		#print('dz',dz)
		rho_b=1.8791e-29*omega_b_h2
		dtimedz=-(9.777e09*yrbysec)*func(z)/h
		#print('dtimedz',dtimedz)
		#dumdz=dz
		dumionstate=region
	
		xp[0]=region[0][0]
		xp[1]=region[0][1]
		xp[2]=region[0][2]
	
		xpsol=sc.optimize.fsolve(funcv,xp,args=(dumionstate,xp),xtol=1.49012e-02)
	
		region[0]=[xpsol[0],xpsol[1],xpsol[2]]
	

		region[6][0],region[7][0],region[8][0],region[9][0]= compute_species(region)
	
		dQdz=region[4][0]*region[0][0]+region[4][1]*region[0][1]+region[4][2]*region[7][0]+region[3][0]*region[6][0]*region[8][0]+region[3][1]*region[7][0]*region[8][0]+region[3][2]*region[0][2]*region[8][0]
	
			
		if (z > 0.0): 
			dQdz=dQdz+dQ_compton_dt(region[5][0],z)*region[8][0]*dtimedz
		x1=2.0/(1.0+z)+(1.0/region[9][0])*(xpsol[0]-xp[0]+xpsol[1]-xp[1]-xpsol[2]+xp[2])/dz #changed right now
		x2=2.0*mprot/(3.0*kboltz*rho_b*(1.0+z)**3*region[9][0])*dQdz
	
		region[5][0]=(region[5][0]+dz*x2)/(1.0-dz*x1)
	
		return region
	
	#@nb.jit(nb.types.Tuple((nb.float64,nb.float64,float64))(float64[:],nb.float64[:,:],float64[:]),nopython=True,nogil=True, fastmath=True)
	def funcv(x,dumionstate,x_old):
		dxdz=np.zeros(3)
	
		dumionstate[0]=[x[0],x[1],x[2]]
	
	
		dumionstate[6][0],dumionstate[7][0],dumionstate[8][0],dumionstate[9][0]= compute_species(dumionstate)
	
		dxdz=[dumionstate[2][0]*x[0] + dumionstate[1][0]*dumionstate[6][0]*dumionstate[8][0],dumionstate[2][1]*x[1] + dumionstate[1][1]*dumionstate[7][0]*dumionstate[8][0],dumionstate[2][2]*dumionstate[7][0] + dumionstate[1][2]*x[2]*dumionstate[8][0]]
	
		return (x[0]-x_old[0]-(-dz_step)*dxdz[0],x[1]-x_old[1]-(-dz_step)*dxdz[1],x[2]-x_old[2]-(-dz_step)*dxdz[2])  #the most critical thing Sourav take dz=-0.2

	#@jit(float64(float64,float64),nopython=True,nogil=True, fastmath=True)
	def dQ_compton_dt(T,z):
		return 6.35e-41*omega_b_h2*(1.0+z)**7*(2.726*(1.0+z)-T)
	
	


	#@nb.jit(float64(float64,float64,float64,float64,float64),nopython=True,nogil=True, fastmath=True)
	def P_V(sigma,Delta,Delta_V,mumean,P_V_norm):
		if (Delta <= Delta_V):
			P_V_ini=math.exp(-0.50*((math.log(Delta)-mumean)/sigma)**2)/(math.sqrt(2*math.pi)*Delta*sigma)
		else:
			P_V_ini=(Delta/Delta_V)**betaindex*math.exp(-0.50*((math.log(Delta_V)-mumean)/sigma)**2)/(math.sqrt(2*math.pi)*Delta_V*sigma)
		return P_V_ini*P_V_norm


	#@jit(float64(nb.float64,float64,float64,float64,float64,nb.float64,nb.int32),nopython=True,nogil=True, fastmath=True)

	def Delta_i_F_M(sigma,F_M_in,Delta_initial,Delta_V,mumean,P_V_norm,ierr):
		Delta_i_F_M_ini=Delta_initial
		MAXIT=10
		ierr=0
		if (Delta_initial > 1e6):
			ierr=1
			return Delta_i_F_M_ini
	
		func1=F_M(sigma,Delta_i_F_M_ini,Delta_V, mumean, P_V_norm)-F_M_in
		dfunc1=P_V(sigma,Delta_i_F_M_ini,Delta_V, mumean, P_V_norm)*Delta_i_F_M_ini
		if (func1 < 0.0):
			flo=Delta_i_F_M_ini
			fhi=1e06
		else:
			flo=0.0
			fhi=Delta_i_F_M_ini
	  

		dxold=fhi-flo
		dx=dxold

		#print('enter do')
	
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
			func1=F_M(sigma,Delta_i_F_M_ini,Delta_V, mumean, P_V_norm)-F_M_in
			dfunc1=P_V(sigma,Delta_i_F_M_ini,Delta_V, mumean, P_V_norm)*Delta_i_F_M_ini
			if (func1 < 0.0):
				flo=Delta_i_F_M_ini
			else:
				fhi=Delta_i_F_M_ini
		   
		ierr=1
		return Delta_i_F_M_ini



	#@nb.jit(float64[:](float64,float64,nb.float64[:],nb.float64[:],float64,float64,float64,float64,float64,float64,float64,float64,float64,nb.int64),nopython=True,nogil=True, fastmath=True)

	def update_Q(z,dz,Q,Qold,sigma,sigmaold,dnphotdz,n_e,n_H,R_e_B,Delta_V,mumean,P_V_norm,ierr):
		ierr=0
		dtimedz=-(9.777e09*yrbysec)*func(z)/h

		if ((Qold[0] < 1.0) or (Qold[1] < Delta_H_overlap)):
			Q[1]=Delta_H_overlap
			#print('enter loop')
			dF_M_dz=(-math.log(Q[1])+0.50*sigma**2)/sigma*P_V(sigma,Q[1],Delta_V, mumean, P_V_norm)*Q[1]**2*(sigma-sigmaold)/dz

			Q[3]=F_M(sigma,Q[1],Delta_V,mumean, P_V_norm)
			Q[2]=F_V(sigma,Q[1],Delta_V,mumean, P_V_norm)
			Q[4]=R(sigma,min(Q[1],1e04),Delta_V,mumean, P_V_norm)
			Q[0]=(Qold[0]+dz*dnphotdz/(n_H*Mpcbycm**3*Q[3]))/(1.0+dz*(R_e_B*dtimedz*Q[4]*n_e*(1.0+z)**3+dF_M_dz)/Q[3])
			#print(Qold[0],dz,dnphotdz,n_H,Q[3],R_e_B,dtimedz,Q[4],n_e,dF_M_dz)
			#Q=[(Qold[0]+dz*dnphotdz/(n_H*Mpcbycm**3*Q[3]))/(1.0+dz*(R_e_B*dtimedz*Q[4]*n_e*(1.0+z)**3+dF_M_dz)/Q[3]),Delta_H_overlap,F_V(sigma,Q[1],Delta_V,mumean, P_V_norm),F_M(sigma,Q[1],Delta_V,mumean, P_V_norm)]
		else:
			Q[0]=Qold[0]

		   #*****************************************
			flo=0.0
			fhi=1.0
			dxold=fhi-flo
			dx=dxold

			Q[3]=Qold[3]
			Q[1]=Delta_i_F_M(sigma,Q[3],Qold[1],Delta_V, mumean, P_V_norm,ierr)
			if (ierr != 0):
				return Q
			Q[4]=R(sigma,min(Q[1],1e04),Delta_V, mumean, P_V_norm)
			func1=Q[3]-Qold[3]+dz*(Q[0]*R_e_B*dtimedz*Q[4]*n_e*(1.0+z)**3-dnphotdz/(n_H*Mpcbycm**3))
			dfunc1=1.0+dz*(R_e_B*dtimedz*n_e*(1.0+z)**3)*Q[1]
		
			#cdef int j
		
			for j in range(1,50):
				if (((Q[3]-fhi)*dfunc1-func1)*((Q[3]-flo)*dfunc1-func1) >= 0.0 or abs(2.0*func1) > abs(dxold*dfunc1)):
					dxold=dx
					dx=0.5*(fhi-flo)
					Q[3]=flo+dx
				else:
					dxold=dx
					dx=func1/dfunc1
					#print('else')
					Q[3]=Q[3]-dx
					#print(dx)
				if (abs(dx) < 1e-08):
					break
				Q[1]=Delta_i_F_M(sigma,Q[3],Qold[1],Delta_V, mumean, P_V_norm,ierr)
				if (ierr != 0):
					return Q
				Q[4]=R(sigma,min(Q[1],1e04),Delta_V, mumean, P_V_norm)
				func1=Q[3]-Qold[3]+dz*(Q[0]*R_e_B*dtimedz*Q[4]*n_e*(1.0+z)**3-dnphotdz/(n_H*Mpcbycm**3))
				dfunc1=1.0+dz*(R_e_B*dtimedz*n_e*(1.0+z)**3)*Q[1]
				if (func1< 0.0):
					flo=Q[3]
				else:
					fhi=Q[3]
		     
		   #*******************************************
			Q[1]=Delta_i_F_M(sigma,Q[3],Qold[1],Delta_V, mumean, P_V_norm,ierr)
			if (ierr != 0):
				return 	 
			if (Q[1] < Delta_H_overlap):
				Q[1]=Delta_H_overlap-1e-03
			Q[3]=F_M(sigma,Q[1],Delta_V, mumean, P_V_norm)
			Q[2]=F_V(sigma,Q[1],Delta_V, mumean, P_V_norm)
			Q[4]=R(sigma,min(Q[1],1e04),Delta_V, mumean, P_V_norm)
		if (Q[0] > 1.0):
			Q[0]=1.0
		return Q


		




	#################################################################################################################################################


	################################################################# reionz_sourav ################################################################

	def get_SED():
		dnphotdm=np.zeros((3,3)) #1st dim for pop2 or pop3 or qso (type ionsource) 2nd dim for HII,HeII,HeIII
		sigma_PH=np.zeros((3,3))
		sigma_PI=np.zeros((3,3))
	
		dnphotdm[0][0],dnphotdm[0][1],dnphotdm[0][2],sigma_PH[0][0],sigma_PH[0][1],sigma_PH[0][2],sigma_PI[0][0],\
			sigma_PI[0][1],sigma_PI[0][2]=get_stellar_SED("/home/atrideb/parameter_estimation_reionization/SpecData/dndm_PopII_salpeter_Z0.004",dnphotdm[0],sigma_PI[0],sigma_PH[0]) #in units of h**2/Msun/mpc**3
		dnphotdm[1][0],dnphotdm[1][1],dnphotdm[1][2],sigma_PH[1][0],sigma_PH[1][1],sigma_PH[1][2],sigma_PI[1][0],\
			sigma_PI[1][1],sigma_PI[1][2]=get_stellar_SED("/home/atrideb/parameter_estimation_reionization/SpecData/dndm_PopIII_salpeter_star",dnphotdm[1],sigma_PI[1],sigma_PH[1])
		sigma_PH[2][0],sigma_PH[2][1],sigma_PH[2][2],sigma_PI[2][0],sigma_PI[2][1],sigma_PI[2][2]=get_quasar_SED(sigma_PI[2],sigma_PH[2])
		print('sigma_PI', sigma_PI)
        #please change /home/atrideb to the path of the parameter_estimation_reionization
		return dnphotdm,sigma_PH,sigma_PI
	
	
	

	def Initialize():
	
		a=norm_sourav(sigma_8,h)
		#print('A_s=',1/(a*6.0*np.pi**2))
		#print('sigma_8',(sigma_sourav(8.0/h)/0.8251083)**2/(4.0*np.pi**4))	
		delta_c=1.69/np.sqrt(a) 
		rho_b=rho_c*omega_b_h2 
	
		rho_0=rho_c*h**2*omega_m
		#print('rho_0',rho_0)
		logM,logsig=setspline_sigma()
	
		dnphotdm,sigma_PH,sigma_PI=get_SED()
		dtimedz=np.zeros(n)
		tdyn_array=np.zeros(n)
		lumfun_integral_qso=np.zeros(n)
	
		for i in range (n):
			dtimedz[i]=-(9.777e09*yrbysec)*func(Z[i])/h
			tdyn_array[i]=(Hubble_dist(Z[i])/h)*9.7776e09
			lumfun_integral_qso[i]=e_QSO*lumfun_integral(Z[i])
	
		dnphotdz_neut=np.zeros((n,3,3))
		dnphotdz_ion=np.zeros((n,3,3))
		Gamma_PI=np.zeros((n,3,3))
		Gamma_PH=np.zeros((n,3,3))
	
		esc_II=esc_PopII*np.ones(n)
		esc_III=esc_PopIII*np.ones(n)
	
	
		dfcolldt_pop3_ion=np.zeros(n)
		dfcolldt_pop2_ion=np.zeros(n)
		dfcolldt_pop3_neut=np.zeros(n)
		dfcolldt_pop2_neut=np.zeros(n)
		dnphotdz_H=np.zeros(n)
		dnphotdz_He=np.zeros(n)
	
		dfcolldt_pop3_ion[0]=0.0
		dfcolldt_pop2_ion[0]=0.0
		dfcolldt_pop3_neut[0]=mass_integral_pop3(Z[0],vcmin_pop3_neut(Z[0]),vc_max(Z[0]),delta_c,logM,logsig)
		dfcolldt_pop2_neut[0]=mass_integral_pop2(Z[0],vcmin_pop2_neut(Z[0]),vc_max(Z[0]),delta_c,logM,logsig)
	
		sfr_pop3_ion=np.zeros(n)
		sfr_pop3_neut=np.zeros(n)
		sfr_pop2_ion=np.zeros(n)
		sfr_pop2_neut=np.zeros(n)
		rhostar_pop3_ion=np.zeros(n)
		rhostar_pop3_neut=np.zeros(n)
		rhostar_pop2_ion=np.zeros(n)
		rhostar_pop2_neut=np.zeros(n)
		mass_integral_pop2_neut=np.zeros(n)
		mass_integral_pop2_ion=np.zeros(n)
		mass_integral_pop3_neut=np.zeros(n)
		mass_integral_pop3_ion=np.zeros(n)
		tau_elsc=np.zeros(n)
		Sigma=np.zeros(n)
		dNLLdz=np.zeros(n)
		nzero=np.zeros(n)
	
	
		neutral=np.zeros((n,10,3))
		HII=np.zeros((n,10,3))
		HeIII=np.zeros((n,10,3))
		Global=np.zeros((n,10,3))
		Global_0=np.zeros((n,10,3))
		neutral_0=np.zeros((n,10,3))
		HII_0=np.zeros((n,10,3))
		HeIII_0=np.zeros((n,10,3))
		
			

		neutral[0][0][0]=1.0-Y_He-1.2e-05*np.sqrt(omega_m)/(h*omega_b)
		neutral[0][0][1]=mp_by_mHe*Y_He
		neutral[0][0][2]=0.0
		neutral[0][5][0]=0.0099*2.726*(1.0+Z[0])**2
		#HII=np.zeros((n,10,3))
		HII[0]=neutral[0]
		#HeIII=np.zeros((n,10,3))
		HeIII[0]=neutral[0]
		#Global=np.zeros((n,10,3))
		Global[0]=neutral[0]
		#neutral_0=np.zeros((n,10,3))
		neutral_0[0]=neutral[0]
		#HII_0=np.zeros((n,10,3))
		HII_0[0]=HII[0]
		#HeIII_0=np.zeros((n,10,3))
		HeIII_0[0]=HeIII[0]
		#Global_0=np.zeros((n,10,3))
		Global_0[0]=Global[0]
		
		#dnphotdz_H=np.zeros(n)
	
		Sigma[0]=np.sqrt(sigmasq_b_sourav(0.0,Z[0],Global[0][5][0],1.0))
		#print('Sigma[0]',Sigma[0],Global[0][5][0])
	
		Delta_V, mumean, P_V_norm=get_LN_Norm(Sigma[0])
		#print('Delta_V,mumean,P_V_norm',Delta_V,mumean,P_V_norm)
	
		QH=np.zeros((n,6))
		QH[0][0]=1e-08
		QH[0][1]=Delta_H_overlap
		QH[0][2]=F_V(Sigma[0],QH[0][1],Delta_V, mumean, P_V_norm)
		QH[0][3]=F_M(Sigma[0],QH[0][1],Delta_V, mumean, P_V_norm)
		QH[0][4]=R(Sigma[0],QH[0][1],Delta_V, mumean, P_V_norm)
		QH[0][5]=1e-16
	
		QHe=np.zeros((n,6))

		QHe[0][0]=1e-10
		QHe[0][1]=Delta_H_overlap
		QHe[0][2]=F_V(Sigma[0],QHe[0][1],Delta_V, mumean, P_V_norm)
		QHe[0][3]=F_M(Sigma[0],QHe[0][1],Delta_V, mumean, P_V_norm)
		QHe[0][4]=R(Sigma[0],QHe[0][1],Delta_V, mumean, P_V_norm)
		QHe[0][5]=0.0
		HII[0][6][0],HII[0][7][0],HII[0][8][0],HII[0][9][0]= compute_species(HII[0])
		HeIII[0][6][0],HeIII[0][7][0],HeIII[0][8][0],HeIII[0][9][0]= compute_species(HeIII[0])
		Global[0][6][0],Global[0][7][0],Global[0][8][0],Global[0][9][0]= compute_species(Global[0])

			
		neutral_0[0][6][0],neutral_0[0][7][0],neutral_0[0][8][0],neutral_0[0][9][0]= compute_species(neutral_0[0])
		HII_0[0][6][0],HII_0[0][7][0],HII_0[0][8][0],HII_0[0][9][0] = compute_species(HII_0[0])
		HeIII_0[0][6][0],HeIII_0[0][7][0],HeIII_0[0][8][0],HeIII_0[0][9][0]= compute_species(HeIII_0[0])
		Global_0[0][6][0],Global_0[0][7][0],Global_0[0][8][0],Global_0[0][9][0]=compute_species(Global_0[0])

	
	
		return rho_b,dnphotdm,sigma_PH,sigma_PI,Z,n,dtimedz,tdyn_array,lumfun_integral_qso,dnphotdz_neut,dnphotdz_ion,dfcolldt_pop3_ion,dfcolldt_pop2_ion,dfcolldt_pop3_neut,dfcolldt_pop2_neut,\
		Gamma_PI,Gamma_PH,sfr_pop3_ion,sfr_pop3_neut,sfr_pop2_ion,sfr_pop2_neut,tau_elsc,neutral,HeIII,Global,HII,neutral_0,Global_0,HeIII_0,HII_0,\
		dnphotdz_H,dnphotdz_He,Sigma,dNLLdz,nzero,QH,QHe,esc_II,esc_III,mass_integral_pop2_neut,mass_integral_pop2_ion,\
		mass_integral_pop3_neut,mass_integral_pop3_ion,logM,logsig,rho_0
	
	#A=Initialize()
	#print(A)	

	def Adjust_Lambda(ierr):
		#A=Initialize()
		#print(len(A))
	
		#start_time=time.time()
	
		rho_b,dnphotdm,sigma_PH,sigma_PI,Z,n,dtimedz,tdyn_array,lumfun_integral_qso,dnphotdz_neut,dnphotdz_ion,dfcolldt_pop3_ion,dfcolldt_pop2_ion,dfcolldt_pop3_neut,dfcolldt_pop2_neut,\
		Gamma_PI,Gamma_PH,sfr_pop3_ion,sfr_pop3_neut,sfr_pop2_ion,sfr_pop2_neut,tau_elsc,neutral,HeIII,Global,HII,neutral_0,Global_0,HeIII_0,HII_0,\
		dnphotdz_H,dnphotdz_He,Sigma,dNLLdz,nzero,QH,QHe,esc_II,esc_III,mass_integral_pop2_neut,mass_integral_pop2_ion,\
		mass_integral_pop3_neut,mass_integral_pop3_ion,logM,logsig,mean_dens=Initialize()
	
	
	
		a=norm_sourav(sigma_8,h)
		esc_frac=np.zeros((3,3))
		delta_c_z=1.69/math.sqrt(a)
		totGamma_PI=np.zeros(3)
		totGamma_PH=np.zeros(3)
	
		#start_time = time.time()
		for k in range (1,n):
		
			#print(time.time())
			escfrac=set_escfrac(esc_II[k],esc_III[k]) #np.zeros(3,3)
			#print(time.time()-start_time,Z[k])
		
			#start_time = time.time()
			dnphotdz_H[k],dnphotdz_He[k],dnphotdz_ion[k],dnphotdz_neut[k]= get_sfr(k,rho_b,dfcolldt_pop2_neut,dfcolldt_pop2_ion,dfcolldt_pop3_neut,dfcolldt_pop3_ion,tdyn_array,mass_integral_pop2_neut,
									mass_integral_pop2_ion,mass_integral_pop3_neut,mass_integral_pop3_ion,dnphotdz_ion,sfr_pop2_neut,sfr_pop2_ion,dnphotdz_neut,sfr_pop3_ion,sfr_pop3_neut,
									QH,dnphotdm,escfrac,dnphotdz_H,dnphotdz_He,dtimedz,Z[k],HII,lumfun_integral_qso,delta_c_z,logM,logsig,mean_dens)
		
			#print(Z[k], dnphotdz_neut[k][2][0])
		
			Sigma[k]=np.sqrt(sigmasq_b_sourav(0.0,Z[k],Global[k-1][5][0],1.0/Global[k-1][9][0]))
		 	
		 	
			Delta_V, mumean, P_V_norm=get_LN_Norm(Sigma[k])
			#print(Z[k])
			Gamma_PI[k],Gamma_PH[k]= get_ionflux(k,HII_0,HeIII_0,Z[k],QH,QHe,Gamma_PI,sigma_PI,dnphotdz_neut,dnphotdz_ion,dtimedz,Gamma_PH,sigma_PH)
			#print('sigma_PH',sigma_PH)
			neutral[k][0]=neutral[k-1][0]
			neutral[k][5][0]=neutral[k-1][5][0]
			HII[k][0]=HII[k-1][0]
			#print(Z[k], HII[k-1][5][0] )
			HII[k][5][0]=HII[k-1][5][0]
			#print(Z[k], HII[k][5][0] )
			HeIII[k][0]=HeIII[k-1][0]
			HeIII[k][5][0]=HeIII[k-1][5][0]
		
			#print(time.time()-start_time)
		
			neutral[k][1],neutral[k][3]=get_recrates(Z[k],neutral[k],1.0,1.0)
			HII[k][1],HII[k][3]=get_recrates(Z[k],HII[k],QH[k-1][4],1.0)
			HeIII[k][1],HeIII[k][3]=get_recrates(Z[k],HeIII[k],QHe[k-1][4],1.0)
			#print('later HII[k]',HII[k])
			neutral[k][2]=[0.0,0.0,0.0]

			totGamma_PI[0]=Gamma_PI[k][0][0]+Gamma_PI[k][1][0]+Gamma_PI[k][2][0]
			totGamma_PI[1]=Gamma_PI[k][0][1]+Gamma_PI[k][1][1]+Gamma_PI[k][2][1]
			totGamma_PI[2]=0.0
		 		
			totGamma_PH[0]=Gamma_PH[k][0][0]+Gamma_PH[k][1][0]+Gamma_PH[k][2][0]
			totGamma_PH[1]=Gamma_PH[k][0][1]+Gamma_PH[k][1][1]+Gamma_PH[k][2][1]
			totGamma_PH[2]=0.0
		 
			#print('Gamma_PI',totGamma_PI[2])
		
			HII[k][2],HII[k][4]=get_ionrates(Z[k],HII[k],totGamma_PI,totGamma_PH,QH[k-1][0])
			totGamma_PI[2]=(Gamma_PI[k][0][2]+Gamma_PI[k][1][2]+Gamma_PI[k][2][2])*(QHe[k-1][5]/QH[k-1][5])*(QH[k-1][0]/QHe[k-1][0])
			totGamma_PH[2]=(Gamma_PH[k][0][2]+Gamma_PH[k][1][2]+Gamma_PH[k][2][2])*(QHe[k-1][5]/QH[k-1][5])*(QH[k-1][0]/QHe[k-1][0])
			HeIII[k][2],HeIII[k][4]=get_ionrates(Z[k],HeIII[k],totGamma_PI,totGamma_PH,QH[k-1][0])
			neutral[k][4]=[0.0,0.0,0.0]
		 	
			#print(time.time())
			#print(Z[k], HII[k][2])
			neutral[k]=update_ionstate(Z[k],dz,neutral[k])
			HII[k]=update_ionstate(Z[k],dz,HII[k])
			#print(Z[k], HeIII[k])
			HeIII[k]=update_ionstate(Z[k],dz,HeIII[k])
			#print(Z[k], HeIII[k][5][0] )
			neutral_0[k][0]=neutral_0[k-1][0]
			neutral_0[k][5][0]=neutral_0[k-1][5][0]
			HII_0[k][0]=HII_0[k-1][0]
			HII_0[k][5][0]=HII_0[k-1][5][0]
			HeIII_0[k][0]=HeIII_0[k-1][0]
			HeIII_0[k][5][0]=HeIII_0[k-1][5][0]
			neutral_0[k][1],neutral_0[k][3]=get_recrates(Z[k],neutral_0[k],1.0,1.0)
			HII_0[k][1],HII_0[k][3]=get_recrates(Z[k],HII_0[k],1.0,1.0)
			HeIII_0[k][1],HeIII_0[k][3]=get_recrates(Z[k],HeIII_0[k],1.0,1.0)
		 	
			neutral_0[k][2]=neutral[k][2]
			HII_0[k][2]=HII[k][2]
			HeIII_0[k][2]=HeIII[k][2]

			neutral_0[k][4]=neutral[k][4]
			HII_0[k][4]=HII[k][4]
			HeIII_0[k][4]=HeIII[k][4]
			neutral_0[k]=update_ionstate(Z[k],dz,neutral_0[k])
			HII_0[k]=update_ionstate(Z[k],dz,HII_0[k])
			HeIII_0[k]=update_ionstate(Z[k],dz,HeIII_0[k])
		 		



			n_e_H=(HII[k][8][0]+(QHe[k-1][0]/QH[k-1][0])*HeIII[k][8][0])*rho_b/mprot*(1.8791e-29/2.7755e11)
			n_H=(HII[k][6][0]+HII[k][0][0])*rho_b/mprot*(1.8791e-29/2.7755e11)
			#print(Z[k], QH[k][0])
			#print(Z[k], HII[k][5][0], R_HII_e_B(HII[k][5][0]))
			QH[k]=update_Q(Z[k],dz,QH[k],QH[k-1],Sigma[k],Sigma[k-1],dnphotdz_H[k],n_e_H,n_H,R_HII_e_A(HII[k][5][0])+R_HII_e_B(HII[k][5][0]),Delta_V, mumean, P_V_norm,ierr)
			#print(Z[k], QH[k][0])
			if (ierr != 0):
				return QH[:,0],QH[:,1],QH[:,4],QH[:,5],QHe[:,0],QHe[:,1],QHe[:,4],QHe[:,5],
		

			n_e_He=HeIII[k][8][0]*rho_b/mprot*(1.8791e-29/2.7755e11)
			n_He=(HeIII[k][0][2]+HeIII[k][7][0]+HeIII[k][0][1])*rho_b/mprot*(1.8791e-29/2.7755e11)
			QHe[k]=update_Q(Z[k],dz,QHe[k],QHe[k-1],Sigma[k],Sigma[k-1],dnphotdz_He[k],n_e_He,n_He,R_HeIII_e_A(HeIII[k][5][0])+R_HeIII_e_B(HeIII[k][5][0]),Delta_V, mumean, P_V_norm,ierr) 

		 		
			Global[k][5][0]=(1.0-QH[k][0]*QH[k][3])*neutral[k][5][0]+(QH[k][0]*QH[k][3]-QHe[k][0]*QHe[k][3])*HII[k][5][0]+QHe[k][0]*QHe[k][3]*HeIII[k][5][0]
			#print(Z[k],Global[k][5][0],QH[k][0],QH[k][3],neutral[k][5][0],QHe[k][0],QHe[k][3],HII[k][5][0],HeIII[k][5][0])
			Global[k][0]=[(1.0-QH[k][0]*QH[k][3])*neutral[k][0][0]+(QH[k][0]*QH[k][3]-QHe[k][0]*QHe[k][3])*HII[k][0][0]+QHe[k][0]*QHe[k][3]*HeIII[k][0][0],(1.0-QH[k][0]*QH[k][3])*neutral[k][0][1]+(QH[k][0]*QH[k][3]-QHe[k][0]*QHe[k][3])*HII[k][0][1]+QHe[k][0]*QHe[k][3]*HeIII[k][0][1],(1.0-QH[k][0]*QH[k][3])*neutral[k][0][2]+(QH[k][0]*QH[k][3]-QHe[k][0]*QHe[k][3])*HII[k][0][2]+QHe[k][0]*QHe[k][3]*HeIII[k][0][2]]
			
			Global[k][6][0],Global[k][7][0],Global[k][8][0],Global[k][9][0]=compute_species(Global[k])

			Global_0[k][5][0]=(1.0-QH[k][0])*neutral_0[k][5][0]+(QH[k][0]-QHe[k][0])*HII_0[k][5][0]+QHe[k][0]*HeIII_0[k][5][0]
			Global_0[k][0]=[(1.0-QH[k][0])*neutral_0[k][0][0]+(QH[k][0]-QHe[k][0])*HII_0[k][0][0]+QHe[k][0]*HeIII_0[k][0][0],(1.0-QH[k][0])*neutral_0[k][0][1]+(QH[k][0]-QHe[k][0])*HII_0[k][0][1]+QHe[k][0]*HeIII_0[k][0][1],(1.0-QH[k][0])*neutral_0[k][0][2]+(QH[k][0]-QHe[k][0])*HII_0[k][0][2]+QHe[k][0]*HeIII_0[k][0][2]]
		
			Global_0[k][6][0],Global_0[k][7][0],Global_0[k][8][0],Global_0[k][9][0]=compute_species(Global_0[k])


		
			dNLLdz[k]=1.0/((1.0+Z[k])*(math.sqrt(math.pi)*QH[k][5]/(Mpcbycm*Hubble_dist(Z[k])*3e3/h)))
			n_e=Global_0[k][8][0]*rho_b/mprot*(1.8791*1e-29/2.7755/1e11)
			tau_elsc[k]=tau_elsc[k-1]+dz*dtimedz[k]*n_e*3*1e10*6.652*10**(-25)*(1.0+Z[k])**3
			#print(Z[k], QH[k][0])
		tau_elsc_today=tau_elsc[n-1]
		tau_elsc=tau_elsc_today-tau_elsc
		#print('tau_elsc_today',tau_elsc_today)
		return QH[:,0],tau_elsc_today,dNLLdz,(Gamma_PI[:,0,0]+Gamma_PI[:,1,0]+Gamma_PI[:,2,0])/10**-12



	#@nb.jit(float64[:,:](nb.float64,nb.float64),nopython=True,nogil=True,fastmath=True)	
	def set_escfrac(esc_II,esc_III):
		esc_II_param=esc_II
		esc_III_param=esc_III
		escfrac=np.zeros((3,3))  #1st for pop2 or pop 3 or QSO 3rd dim H2,He2,He3
		if which_pop=='both':
			escfrac=[[esc_II_param,esc_II_param,esc_II_param],[esc_III_param,esc_III_param,esc_III_param]]
			#escfrac[0][0]=esc_II_param
			#escfrac[0][1]=esc_II_param
			#escfrac[0][2]=esc_II_param
		
			#escfrac[1][0]=esc_III_param
			#escfrac[1][1]=esc_III_param
			#escfrac[1][2]=esc_III_param
		return escfrac
		   	


	def get_sfr(
		k,rho_b,dfcolldt_pop2_neut,dfcolldt_pop2_ion,dfcolldt_pop3_neut,dfcolldt_pop3_ion,tdyn_array,mass_integral_pop2_neut,mass_integral_pop2_ion,mass_integral_pop3_neut,mass_integral_pop3_ion,
	dnphotdz_ion,sfr_pop2_neut,sfr_pop2_ion,dnphotdz_neut,sfr_pop3_ion,sfr_pop3_neut,QH,dnphotdm,escfrac,dnphotdz_H,dnphotdz_He,dtimedz,z,HII,lumfun_integral_qso,delta_c_z,logM,logsig,mean_dens):
	
	
	
		prefactor=omega_z(z)**0.6
	
	
	
		dfcolldt_pop2_ion[k]=mass_integral_pop2(z,vcmin_pop2_ion(z,HII[k-1][5][0],HII[k-1][9][0]),vcmax_func(z),delta_c_z,logM,logsig)*(prefactor/tdyn_array[k])
	
		#start_time = time.time()
		dfcolldt_pop2_neut[k]=mass_integral_pop2(z,vcmin_pop2_neut(z),vcmax_func(z),delta_c_z,logM,logsig)*(prefactor/tdyn_array[k])
	
		#print(time.time()-start_time,Z[k])
	
	
		dfcolldt_pop3_ion[k]=mass_integral_pop3(z,vcmin_pop3_ion(z,HII[k-1][5][0],HII[k-1][9][0]),vcmax_func(z),delta_c_z,logM,logsig)*(prefactor/tdyn_array[k])
		dfcolldt_pop3_neut[k]=mass_integral_pop3(z,vcmin_pop3_neut(z),vcmax_func(z),delta_c_z,logM,logsig)*(prefactor/tdyn_array[k])
	
	
	
	
		mass_integral_pop2_neut[k]=e_sf_II*dfcolldt_pop2_neut[k]
		mass_integral_pop2_ion[k]=e_sf_II*dfcolldt_pop2_ion[k]
		mass_integral_pop3_neut[k]=e_sf_III*dfcolldt_pop3_neut[k]
		mass_integral_pop3_ion[k]=e_sf_III*dfcolldt_pop3_ion[k]
	 
		sfrfactor=rho_b
		#print(sfrfactor)
		#print('dfdt',dfcolldt_pop2_ion[k],dfcolldt_pop2_neut[k],dfcolldt_pop3_neut[k],dfcolldt_pop3_ion[k])
	
		dnphotdz_ion[k][0]=[QH[k-1][0]*dnphotdm[0][0]*mass_integral_pop2_ion[k]*sfrfactor*escfrac[0][0]*dtimedz[k]/yrbysec,QH[k-1][0]*dnphotdm[0][1]*mass_integral_pop2_ion[k]*sfrfactor*escfrac[0][1]*dtimedz[k]/yrbysec,QH[k-1][0]*dnphotdm[0][2]*mass_integral_pop2_ion[k]*sfrfactor*escfrac[0][2]*dtimedz[k]/yrbysec]
	
		sfr_pop2_ion[k]=-QH[k-1][0]*mass_integral_pop2_ion[k]*sfrfactor
	
		dnphotdz_neut[k][0]=[(1.0-QH[k-1][0])*dnphotdm[0][0]*mass_integral_pop2_neut[k]*sfrfactor*escfrac[0][0]*dtimedz[k]/yrbysec,(1.0-QH[k-1][0])*dnphotdm[0][1]*mass_integral_pop2_neut[k]*sfrfactor*escfrac[0][1]*dtimedz[k]/yrbysec,(1.0-QH[k-1][0])*dnphotdm[0][2]*mass_integral_pop2_neut[k]*sfrfactor*escfrac[0][2]*dtimedz[k]/yrbysec]
	
		sfr_pop2_neut[k]=-(1.0-QH[k-1][0])*mass_integral_pop2_neut[k]*sfrfactor
	
		dnphotdz_ion[k][1]=[QH[k-1][0]*dnphotdm[1][0]*mass_integral_pop3_ion[k]*sfrfactor*escfrac[1][0]*dtimedz[k]/yrbysec,QH[k-1][0]*dnphotdm[1][1]*mass_integral_pop3_ion[k]*sfrfactor*escfrac[1][1]*dtimedz[k]/yrbysec,QH[k-1][0]*dnphotdm[1][2]*mass_integral_pop3_ion[k]*sfrfactor*escfrac[1][2]*dtimedz[k]/yrbysec]
	
		sfr_pop3_ion[k]=-QH[k-1][0]*mass_integral_pop3_ion[k]*sfrfactor

		#print('dnphot',dnphotdz_neut[k][1][1])
		
		dnphotdz_neut[k][1]=[(1.0-QH[k-1][0])*dnphotdm[1][0]*mass_integral_pop3_neut[k]*sfrfactor*escfrac[1][0]*dtimedz[k]/yrbysec,(1.0-QH[k-1][0])*dnphotdm[1][1]*mass_integral_pop3_neut[k]*sfrfactor*escfrac[1][1]*dtimedz[k]/yrbysec,(1.0-QH[k-1][0])*dnphotdm[1][2]*mass_integral_pop3_neut[k]*sfrfactor*escfrac[1][2]*dtimedz[k]/yrbysec]
		sfr_pop3_neut[k]=-(1.0-QH[k-1][0])*mass_integral_pop3_neut[k]*sfrfactor


		#rhostar_pop3_ion[k]=rhostar_pop3_ion[k-1]+dz*dtimedz[k]*sfr_pop3_ion[k]/yrbysec
		#rhostar_pop3_neut[k]=rhostar_pop3_neut[k-1]+dz*dtimedz[k]*sfr_pop3_neut[k]/yrbysec
		#rhostar_pop2_ion[k]=rhostar_pop2_ion[k-1]+dz*dtimedz[k]*sfr_pop2_ion[k]/yrbysec
		#rhostar_pop2_neut[k]=rhostar_pop2_neut[k-1]+dz*dtimedz[k]*sfr_pop2_neut[k]/yrbysec

		dnphotdz_neut[k][2]=[QH[k-1][0]*yrbysec*10.0**18.05*(1.0-(nu_HII/nu_HeII)**1.570)*lumfun_integral_qso[k]/(1.570*hPlanck)*dtimedz[k]/yrbysec,QH[k-1][0]*yrbysec*10.0**18.05*((nu_HII/(nu_HeII))**1.57-(nu_HII/(nu_HeIII))**1.57)*lumfun_integral_qso[k]/(1.57*hPlanck)*dtimedz[k]/yrbysec,QH[k-1][0]*yrbysec*10.0**18.05*(nu_HII/(nu_HeIII))**1.570*lumfun_integral_qso[k]/(1.57*hPlanck)*dtimedz[k]/yrbysec]
	
		    
		dnphotdz_ion[k][2]=[(1.0-QH[k-1][0])*yrbysec*10.0**18.05*(1.0-(nu_HII/nu_HeII)**1.570)*lumfun_integral_qso[k]/(1.570*hPlanck)*dtimedz[k]/yrbysec,(1.0-QH[k-1][0])*yrbysec*10.0**18.05*((nu_HII/(nu_HeII))**1.57-(nu_HII/(nu_HeIII))**1.57)*lumfun_integral_qso[k]/(1.57*hPlanck)*dtimedz[k]/yrbysec,(1.0-QH[k-1][0])*yrbysec*10.0**18.05*(nu_HII/(nu_HeIII))**1.57*lumfun_integral_qso[k]/(1.57*hPlanck)*dtimedz[k]/yrbysec]
	
	
			
		

		dnphotdz_H[k]=(dnphotdz_neut[k][0][0]+dnphotdz_neut[k][1][0]+dnphotdz_neut[k][2][0]+dnphotdz_neut[k][0][1]+dnphotdz_neut[k][1][1]+dnphotdz_neut[k][2][1])+\
		(dnphotdz_ion[k][0][0]+dnphotdz_ion[k][1][0]+dnphotdz_ion[k][2][0]+dnphotdz_ion[k][0][1]+dnphotdz_ion[k][1][1]+dnphotdz_ion[k][2][1])
	
		dnphotdz_He[k]=(dnphotdz_neut[k][0][2]+dnphotdz_neut[k][1][2]+dnphotdz_neut[k][2][2])+(dnphotdz_ion[k][0][2]+dnphotdz_ion[k][1][2]+dnphotdz_ion[k][2][2])

	
	
	 
		return  dnphotdz_H[k],dnphotdz_He[k],dnphotdz_ion[k],dnphotdz_neut[k]
	  	

	#@nb.jit(nb.types.Tuple((nb.float64[:,:],nb.float64[:,:]))(nb.int32,nb.float64[:,:,:],float64[:,:,:],float64,float64[:,:],float64[:,:],float64[:,:,:],float64[:,:],float64[:,:,:],float64[:,:,:],float64[:],float64[:,:,:],float64[:,:]),nopython=True,nogil=True, fastmath=True)
	def get_ionflux(k,HII_0,HeIII_0,z,QH,QHe,Gamma_PI,sigma_PI,dnphotdz_neut,dnphotdz_ion,dtimedz,Gamma_PH,sigma_PH):
		lambda_0_H=lambda_0*np.sqrt(xbsq(HII_0[k-1][5][0],z,1.0/HII_0[k-1][9][0]))/(1.0+z)
		lambda_0_He=lambda_0*np.sqrt(xbsq(HeIII_0[k-1][5][0],z,1.0/HeIII_0[k-1][9][0]))/(1.0+z)
		if (1.0-QH[k-1][2] >= 0.0):
			QH[k][5]=QH[k-1][0]**(1.0/3.0)*lambda_0_H*Mpcbycm/(1.0-QH[k-1][2])**(2.0/3.0)
		else:
			QH[k][5]=QH[k-1][5]
			
		if (1.0-QHe[k-1][2] >= 0.0):
			QHe[k][5]=QHe[k-1][0]**(1.0/3.0)*lambda_0_He*Mpcbycm/(1.0-QHe[k-1][2])**(2.0/3.0)
		else:
			QHe[k][5]=QHe[k-1][5]
			
			
	
		Gamma_PI[k][0]=[QH[k][5]*sigma_PI[0][0]*(1.0+z)**3*(dnphotdz_neut[k][0][0]+dnphotdz_ion[k][0][0])/(dtimedz[k]*Mpcbycm**3),QH[k][5]*sigma_PI[0][1]*(1.0+z)**3*(dnphotdz_neut[k][0][1]+dnphotdz_ion[k][0][1])/(dtimedz[k]*Mpcbycm**3),QHe[k][5]*sigma_PI[0][2]*(1.0+z)**3*(dnphotdz_neut[k][0][2]+dnphotdz_ion[k][0][2])/(dtimedz[k]*Mpcbycm**3)]
	
	
		Gamma_PI[k][1]=[QH[k][5]*sigma_PI[1][0]*(1.0+z)**3*(dnphotdz_neut[k][1][0]+dnphotdz_ion[k][1][0])/(dtimedz[k]*Mpcbycm**3),QH[k][5]*sigma_PI[1][1]*(1.0+z)**3*(dnphotdz_neut[k][1][1]+dnphotdz_ion[k][1][1])/(dtimedz[k]*Mpcbycm**3),QHe[k][5]*sigma_PI[1][2]*(1.0+z)**3*(dnphotdz_neut[k][1][2]+dnphotdz_ion[k][1][2])/(dtimedz[k]*Mpcbycm**3)]
	

	
		Gamma_PI[k][2]=[QH[k][5]*sigma_PI[2][0]*(1.0+z)**3*(dnphotdz_neut[k][2][0]+dnphotdz_ion[k][2][0])/(dtimedz[k]*Mpcbycm**3),QH[k][5]*sigma_PI[2][1]*(1.0+z)**3*(dnphotdz_neut[k][2][1]+dnphotdz_ion[k][2][1])/(dtimedz[k]*Mpcbycm**3),QHe[k][5]*sigma_PI[2][2]*(1.0+z)**3*(dnphotdz_neut[k][2][2]+dnphotdz_ion[k][2][2])/(dtimedz[k]*Mpcbycm**3)]
			
		#print(sigma_PI[2][0])	
		Gamma_PH[k][0]=[hPlanck*nu_HII*QH[k][5]*sigma_PH[0][0]*(1.0+z)**3*(dnphotdz_neut[k][0][0]+dnphotdz_ion[k][0][0])/(dtimedz[k]*Mpcbycm**3),hPlanck*nu_HeII*QH[k][5]*sigma_PH[0][1]*(1.0+z)**3*(dnphotdz_neut[k][0][1]+dnphotdz_ion[k][0][1])/(dtimedz[k]*Mpcbycm**3),hPlanck*nu_HeIII*QHe[k][5]*sigma_PH[0][2]*(1.0+z)**3*(dnphotdz_neut[k][0][2]+dnphotdz_ion[k][0][2])/(dtimedz[k]*Mpcbycm**3)]
	
	
	
		Gamma_PH[k][1]=[hPlanck*nu_HII*QH[k][5]*sigma_PH[1][0]*(1.0+z)**3*(dnphotdz_neut[k][1][0]+dnphotdz_ion[k][1][0])/(dtimedz[k]*Mpcbycm**3),hPlanck*nu_HeII*QH[k][5]*sigma_PH[1][1]*(1.0+z)**3*(dnphotdz_neut[k][1][1]+dnphotdz_ion[k][1][1])/(dtimedz[k]*Mpcbycm**3),hPlanck*nu_HeIII*QHe[k][5]*sigma_PH[1][2]*(1.0+z)**3*(dnphotdz_neut[k][1][2]+dnphotdz_ion[k][1][2])/(dtimedz[k]*Mpcbycm**3)]
	
	
		Gamma_PH[k][2]=[hPlanck*nu_HII*QH[k][5]*sigma_PH[2][0]*(1.0+z)**3*(dnphotdz_neut[k][2][0]+dnphotdz_ion[k][2][0])/(dtimedz[k]*Mpcbycm**3),hPlanck*nu_HeII*QH[k][5]*sigma_PH[2][1]*(1.0+z)**3*(dnphotdz_neut[k][2][1]+dnphotdz_ion[k][2][1])/(dtimedz[k]*Mpcbycm**3),hPlanck*nu_HeIII*QHe[k][5]*sigma_PH[2][2]*(1.0+z)**3*(dnphotdz_neut[k][2][2]+dnphotdz_ion[k][2][2])/(dtimedz[k]*Mpcbycm**3)]
		
		return Gamma_PI[k],Gamma_PH[k]
	

	#print('before calling Adjust lambda',time.time()-start_time)
	#start_time=time.time()
	A=Adjust_Lambda(0)
	#print("compare %s seconds ---" % (time.time() - start_time))

	#QH[:,0]0,QH[:,1]1,tau 2,QH[:,4]3,QH[:,5]4,QHe[:,0]5,QHe[:,1]6,QHe[:,4]7,QHe[:,5]8,dNLLdz9,Gamma_PI[:,0,0]10,10Gamma_PI[:,0,2],11Global[:,5,0],12HII[:,5,0],13HeIII[:,5,0],14tot_sfr,15sfr_pop3,16sfr_pop2,17x_HI						
	#np.savetxt('QHe.dat',np.c_[Z,A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8],A[9],A[10],A[11],A[12],A[13],A[14],A[15],A[16],A[17]])
	
	return Z,A[0],A[1],A[2],A[3],A[0][96]
