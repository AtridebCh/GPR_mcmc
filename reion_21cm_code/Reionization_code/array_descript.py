import numpy as np

zstart = 2.0
zend = 25.0
dz_step=0.2
dz=-0.2 #Note that we are going from redshift 25 to 2
n=int(abs(((zend-zstart)/dz_step)))+1

Z=np.linspace(zend,zstart,n)


Ionized_species= {'QH': {'Q': np.zeros(n), 'Delta': np.zeros(n), 'F_V': np.zeros(n), 'F_M': np.zeros(n), 'R': np.zeros(n), 'mfp': np.zeros(n) },
          'QHe': {'Q': np.zeros(n), 'Delta': np.zeros(n), 'F_V': np.zeros(n), 'F_M': np.zeros(n), 'R': np.zeros(n), 'mfp': np.zeros(n) }}



neutral_region={'frac':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'recrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'ionrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'coolrate': {'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'heatrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'T': np.zeros(n), 'X_HII': np.zeros(n), 'X_HeII': np.zeros(n), 'X_e': np.zeros(n), 'X': np.zeros(n)} 

HII_region={'frac':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'recrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'ionrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'coolrate': {'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'heatrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'T': np.zeros(n), 'X_HII': np.zeros(n), 'X_HeII': np.zeros(n), 'X_e': np.zeros(n), 'X': np.zeros(n)} 

HeIII_region={'frac':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'recrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'ionrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'coolrate': {'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'heatrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'T': np.zeros(n), 'X_HII': np.zeros(n), 'X_HeII': np.zeros(n), 'X_e': np.zeros(n), 'X': np.zeros(n)} 

Global_region={'frac':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'recrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'ionrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'coolrate': {'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'heatrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'T': np.zeros(n), 'X_HII': np.zeros(n), 'X_HeII': np.zeros(n), 'X_e': np.zeros(n), 'X': np.zeros(n)} 


neutral_0={'frac':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'recrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'ionrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'coolrate': {'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'heatrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'T': np.zeros(n), 'X_HII': np.zeros(n), 'X_HeII': np.zeros(n), 'X_e': np.zeros(n), 'X': np.zeros(n)} 

HII_0={'frac':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'recrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'ionrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'coolrate': {'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'heatrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'T': np.zeros(n), 'X_HII': np.zeros(n), 'X_HeII': np.zeros(n), 'X_e': np.zeros(n), 'X': np.zeros(n)} 

HeIII_0={'frac':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'recrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'ionrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'coolrate': {'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'heatrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'T': np.zeros(n), 'X_HII': np.zeros(n), 'X_HeII': np.zeros(n), 'X_e': np.zeros(n), 'X': np.zeros(n)} 

Global_0={'frac':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'recrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'ionrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'coolrate': {'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)},'heatrate':{'HI':np.zeros(n), 'HeI':np.zeros(n), 'HeIII': np.zeros(n)}, 'T': np.zeros(n), 'X_HII': np.zeros(n), 'X_HeII': np.zeros(n), 'X_e': np.zeros(n), 'X': np.zeros(n)} 


dnphotdz_neut={'PopII':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}, 'PopIII':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}, 'QSO':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}}
dnphotdz_ion={'PopII':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}, 'PopIII':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}, 'QSO':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}}
Gamma_PI={'PopII':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}, 'PopIII':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}, 'QSO':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}}
Gamma_PH={'PopII':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}, 'PopIII':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}, 'QSO':{'HII':np.zeros(n),'HeII':np.zeros(n),'HeIII':np.zeros(n)}}


dnphotdm={'PopII':{'HII':0.0,'HeII':0.0,'HeIII':0.0}, 'PopIII':{'HII':0.0,'HeII':0.0,'HeIII':0.0}}
sigma_PI={'PopII':{'HII':0.0,'HeII':0.0,'HeIII':0.0}, 'PopIII':{'HII':0.0,'HeII':0.0,'HeIII':0.0}, 'QSO':{'HII':0.0,'HeII':0.0,'HeIII':0.0}}
sigma_PH={'PopII':{'HII':0.0,'HeII':0.0,'HeIII':0.0}, 'PopIII':{'HII':0.0,'HeII':0.0,'HeIII':0.0}, 'QSO':{'HII':0.0,'HeII':0.0, 'HeIII':0.0}}


dnphotdz={'H': np.zeros(n), 'He':  np.zeros(n)}

mass_integral={'PopII':{'neut': 0.0, 'ion': 0.0}, 'PopIII':{'neut': 0.0, 'ion': 0.0}}
rhostar={'PopII':{'neut': np.zeros(n), 'ion': np.zeros(n)}, 'PopIII':{'neut': np.zeros(n), 'ion': np.zeros(n)}}
sfr={'PopII':{'neut': np.zeros(n), 'ion': np.zeros(n)}, 'PopIII':{'neut': np.zeros(n), 'ion': np.zeros(n)}}
dfcolldt={'PopII':{'neut':0.0, 'ion': 0.0}, 'PopIII':{'neut': 0.0, 'ion': 0.0}}


escfrac={'PopII':{'HII':0.0,'HeII':0.0,'HeIII':0.0}, 'PopIII':{'HII':0.0,'HeII':0.0,'HeIII':0.0}, 'QSO':{'HII':0.0,'HeII':0.0,'HeIII':0.0}}



esc_II=np.ones(n)
esc_III=np.ones(n)

dtimedz=np.zeros(n)
tdyn_array=np.zeros(n)
lumfun_integral_qso=np.zeros(n)
tau_elsc=np.zeros(n)
Sigma=np.zeros(n)
dNLLdz=np.zeros(n)
nzero=np.zeros(n)
totGamma_PI=np.zeros(3)
totGamma_PH=np.zeros(3)
