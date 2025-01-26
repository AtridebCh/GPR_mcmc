import numpy as np
import scipy as sp
from numpy import genfromtxt
from scipy import stats
from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.interpolate import interp1d

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



cLight   =2.99792458e+08 #S.I     
hPlanck  =6.6260755e-34      
kBoltz   =1.380658e-23     
mElect   =9.1093897e-31       
mHatom   =1.673725e-27       
G        =6.67259e-11   
aRad    = 7.565914e-16  
sigmaT   =6.65e-29
mpc_to_m=3.0*10**22
mpc_to_cm=3.0*10**24
boltzmann_constant=1.38*10**-23  #S.I
lyalpha_pop2=1.13e61/(8e15)  #M_solar^{-1}/Hz; number of Ly-alpha photon from PopII stars using starburst99
lyalpha_pop3=1.15e61/(8e15)  #M_solar^{-1}/Hz

solar_mass_to_kg=2.0*10**30
clight=3.0*10**10 #c.g.s
spectral_index=-1.1
Y_He=0.25

class generating_21cm_signal:




	def __init__(self,MCMC_H0, MCMC_ombh2, MCMC_omch2, f_X, f_alpha, redshift_EDGES, SFRD_file, QHII_file): #redshift_EDGES is the redshift where you want to calculate the signal
		self.h=MCMC_H0/100.0
		self.omb=MCMC_ombh2/self.h**2
		self.omega_m = (MCMC_ombh2+MCMC_omch2)/self.h**2
		self.f_R_2=0.0 #radio efficiency from PopII star. use zero for now
		self.f_X_2=f_X # X-ray efficiency, one can use 0.2, Furlanetto;06
		self.f_alpha_2=f_alpha #Ly-alpha efficiency one can take 0.1, Chatterjee, 24
		self.redshift_array, self.pop2 = np.loadtxt(SFRD_file, unpack = True) #SFRD file contains two columns z, SFRD(solar_mass/yr/mpc^3)
		_, QHII = np.loadtxt(QHII_file, unpack = True) #QHII file contains two columns z, QHII
		self.pop2_sfrd=sp.interpolate.interp1d(self.redshift_array, pop2) #solar_mass/yr/mpc^3
		_, self.QHII=sp.interpolate.interp1d(self.redshift_array, QHII)
		self.redshift_edges=redshift_EDGES
		self.number_density_hydrogen=2.0*10**-1*(MCMC_ombh2/0.022)*(1-Y_He) 
		self.QHII=self.QHII(self.redshift_edges)
		
	def Hubble(self,z):
		H_0=100.0*self.h
		omegaR=0.0
		om_k=0.0
		omega_l=1-om_k-self.omega_m
		return np.sqrt(H_0**2*(self.omega_m*(1+z)**3+omegaR*(1+z)**4+om_k*(1+z)**2+omega_l))*10**3/(3.0*10**22)	#unit /sec

	def integrand_for_radio(self,z_prime,z):
		return self.f_R_2*self.pop2_sfrd(z_prime)/((1+z_prime)**2.1*self.Hubble(z_prime)*(mpc_to_m)**3)

	def integrand_for_lyalpha(self,z_prime):
		return self.f_alpha_2*lyalpha_pop2*self.pop2_sfrd(z_prime)/((1+z_prime)*365*24*3600*self.Hubble(z_prime))
		#f_alpha_2*lyalpha_pop2*pop2_sfrd should have an unit of mpc^{-3}sec{-1}Hz{-1}, Now pop2_sfrd already have an unit of 
		#M_solar/mpc^3/yr and lyalpha_pop2 have an unit of M_solar{-1}Hz{-1}, so all good and divide the whole thing by (365*24*3600) to convert yr to sec, other wise J_alpha will be in year not in sec.
	
	def J_alpha(self,Z):
		z_init = Z[0]
		lyphoton=np.zeros(len(Z))
		for i,z in enumerate(Z):
			Z_integration_points=np.linspace(z_init,z,int(abs((z-z_init)/0.01+1)))
			lyalpha_array=self.integrand_for_lyalpha(Z_integration_points)
			ans=integrate.simps(lyalpha_array,Z_integration_points)
			lyphoton[i]=-ans*clight*(1+z)**3/(4*np.pi)/(mpc_to_cm)**3
		return lyphoton
		
	def x_alpha(self,z):
		return 1.81*self.J_alpha(z)*(1+z)**-1*10**11

	def func(self,z,T_k): #add PBH heating inside this function, in this code, I take only X-ray heating
		dT_kdz=2.0*T_k/(1.0+z)-2.0/3.0*self.f_X_2*3.4*(10**33)*self.pop2_sfrd(z)/(boltzmann_constant*(1+z)*self.number_density_hydrogen*(mpc_to_m)**3*self.Hubble(z)) #Furlanetto, 06; 
		return dT_kdz

	def background_temp(self,z,z_init):
		sol,err=sp.integrate.quad(self.integrand_for_radio,z_init,z,args=(z,),epsabs=1.49e-03,epsrel=1.49e-03)
		T_R=-10**22*(142.0/15.0)**-1.1*(1+z)**(3.0-spectral_index)*(cLight)**3/(4*np.pi)*(1.0/(2.0*boltzmann_constant*(1420*10**6)**2))*sol
		T_background=2.73*(1+z)+T_R
		return T_background	
		
	def signal_generator(self):
		z_init=self.redshift_edges[0]
		z_eor=self.redshift_edges[-1]
		
		T_Bright=np.zeros(len(self.redshift_edges))
		T_k0=[(1+z_init)**2/(1+50.0)**2*50.0]
		
		sol =solve_ivp(lambda z, T_k: self.func(z,T_k),(z_init,z_eor),T_k0,'BDF',t_eval=self.redshift_edges, first_step=(z_init-z_eor))
		T_k=sol.y[0]
		if (np.any(T_k > 1.e8)): return T_Bright - 1.e8
		
		x_alpha_1=self.x_alpha(self.redshift_edges)
		
		for count,z in enumerate(self.redshift_edges):
			T_bg = self.background_temp(z,z_init)
			T_s = (T_bg * T_k[count] * (1 + x_alpha_1[count]) ) / (T_k[count] + x_alpha_1[count] * T_bg)
		        #print(z, T_bg, T_k[count], T_s)
			if (T_s < 1.e-8):
		        	#T_b = 1000.0 * (T_s - T_bg) / (1 + z)
				T_b=1000.0*(T_s-T_bg)/(1+z)*(1-np.exp(-0.0092*(1+z)**1.5/T_s))*(1-self.QHII[count])
			else:
		        	#T_b = 1000.0 * (T_s - T_bg) / (1 + z) * ( 1 - np.exp(-0.0092 * (1 + z) ** 1.5 / T_s) )
				T_b=1000.0*(T_s-T_bg)/(1+z)*(1-np.exp(-0.0092*(1+z)**1.5/T_s))*(1-self.QHII[count])
			T_Bright[count]=T_b
			#T_S[count]=T_s-T_bg
		        #T_radio[count]=T_bg-2.73*(1+z)
		        #print(z, T_b, T_k[count], T_s, T_bg, x_alpha_1[count])
		return T_Bright

'''
redshift_edges,bright_temp = genfromtxt('/home/atrideb/Downloads/figure2_plotdata.csv', delimiter=',', skip_header=27, usecols=(1,4),unpack=True)


start=time.time()
Z,x_HII,tau_elsc_today,dNLLdz,Gamma_PI_12,pop2,pop3,x_HI5=running_code_with_21(71.22007, 0.02259486, 0.1259488, 0.819696266003826, 0.9480701, 0.004602334, 0.0001, 5.451408)
#pop2=np.zeros(len(pop2))
#70.0,0.0226,0.122,0.8,0.96,0.0046,0.0001,5.44
signal=generating_21cm_signal(71.22007, 0.02259486, 0.1259488,0.05,1.7e4,5.0, redshift_edges, Z, pop2, pop3, 0.0,0.0, 0.0)

#71.22007 0.02259486 0.1259488 0.819696266003826 0.9480701 0.004602334 0.0001 5.451408
T_bright=signal.signal_generator()
print('time required', time.time()-start)
plt.plot(redshift_edges,T_bright,label='fiducial')
#(70.0, 0.0226, 0.122,f_star_2*f_X_2, f_star_3*f_R_3, f_star_3*f_alpha_3, redshift_edges,Z, pop2, pop3, f_star_2*f_R_2, f_star_3*f_X_3, f_star_2*f_alpha_2)
plt.plot(redshift_edges,bright_temp*1000,label='edges')
plt.xlabel('z')
plt.ylabel('T_b (mK)')
plt.legend()
plt.show()
#plt.semilogy(Z,0.1*pop2,'ro')
#plt.semilogy(Z,pop3,'bo')
#plt.xlim(redshift_edges[0],redshift_edges[-1])
#plt.plot(redshift_edges, bright_temp)
#plt.show()

'''


















    
    
    
