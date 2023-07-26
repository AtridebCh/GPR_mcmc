import numpy as np


UVLF_data_file='UVLF.dat'
halofile='halo_mass_new.dat'


from utils import Params

from Likelihood import Likelihood
from CoredataGenerator import CoreModule

#Lymanlimitdatafile='/home/atridebchatterjee/reion_GPR/reion_21cm_code/ObsData/Lyman_limit.dat'
#gammadatafile='/home/atridebchatterjee/reion_GPR/reion_21cm_code/ObsData/gamma_data_all_combined.dat'

Lymanlimitdatafile='/home/atridebchatterjee/Downloads/GPR_mcmc-master_sourav/GPR_mcmc-master/reion_21cm_code/ObsData/Lyman_limit.dat'
gammadatafile='/home/atridebchatterjee/Downloads/GPR_mcmc-master_sourav/GPR_mcmc-master/reion_21cm_code/ObsData/gamma_data_all_combined.dat'

Planck={'H0':67.70, 'ombh2': 0.0223, 'omch2': 0.12, 'sigma8':0.81, 'ns':0.96} 
Astro_dict={'esc_pop_III':0.0}
frac_size=0.001

#init= 0.0036*np.random.rand(6)
#init=np.array([0.00263737, 0.00335895, 0.00196963, 0.00302385, 0.00173669, 0.00318811]) # produced using 0.0036*np.random.rand(6)
'''
init=np.array([3.204587e-03, 3.679605e-03, 2.868110e-03, 2.842103e-03, 6.101025e-04, 3.871218e-03, 5.002051e+00])
params = Params(("epsilon_a0", [init[0], 0.0, 1.0, frac_size]),  #  #start  param_min param_max paramWidth
                ("epsilon_a1", [init[1], 0.0, 1.0, frac_size]),
                ("epsilon_a2", [init[2], 0.0, 1.0, frac_size]),
		("epsilon_a3", [init[3], 0.0, 1.0, frac_size]),
		("epsilon_a4", [init[4], 0.0, 1.0, frac_size]),
		("epsilon_a5", [init[5], 0.0, 1.0, frac_size]),
		("lambda_0",   [init[6], 0.0, 10.0, frac_size]))
				
'''
init=np.array([0.003, 0.003, 0.002, 0.003])
params = Params(("epsilon_a0", [init[0], 1e-6, 1.0, frac_size]),  #  #start  param_min param_max paramWidth
                ("epsilon_a1", [init[1], 1e-6, 1.0, frac_size]),
                ("epsilon_a2", [init[2], 1e-6, 1.0, frac_size]),
		("epsilon_a3", [init[3], 1e-6, 1.0, frac_size]),
		("lambda_0",   [5.0, 1.0, 100.0, 0.1]))


param_latex_name_arr=['\epsilon_{a_0}','\epsilon_{a_1}','\epsilon_{a_2}', '\epsilon_{a_3}', '\lambda_0']

print('params_value', params[:,0])





like = Likelihood(CoreModule, Planck, Astro_dict, params[:,1], params[:,2])
like.setup(gammadatafile, Lymanlimitdatafile)


likelihood, blobs=like(params[:,0], single_run=True)
print('likelihhod=%s, tau=%s, QHII_5point8=%s' %(likelihood, blobs[0], blobs[1]))
