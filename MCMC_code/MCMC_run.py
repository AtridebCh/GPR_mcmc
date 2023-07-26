import sys, os
from os.path import join
import numpy as np
import os.path



from numpy import exp, array, arange


from utils import Params

from sampler import MCMCsampler
from Likelihood import Likelihood
from CoredataGenerator import CoreModule

Lymanlimitdatafile='/home/atridebchatterjee/reion_GPR/reion_21cm_code/ObsData/Lyman_limit.dat'
gammadatafile='/home/atridebchatterjee/reion_GPR/reion_21cm_code/ObsData/gamma_data_all_combined.dat'

Planck={'H0':67.70, 'ombh2': 0.0223, 'omch2': 0.12, 'ns':0.96, 'sigma8':0.81} 
Astro_dict={'esc_pop_III':0.0}
frac_size=0.001

'''
#init=np.array([0.003, 0.004, 0.002, 0.005, 0.02, 0.05, 6.0]) # produced using 0.0036*np.random.rand(6)
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
		("lambda_0",   [5.0, 1e-6, 100.0, 0.1]))


param_latex_name_arr=['\epsilon_{a_0}','\epsilon_{a_1}','\epsilon_{a_2}', '\epsilon_{a_3}', '\lambda_0']

#param_latex_name_arr=['\epsilon_{a_0}','\epsilon_{a_1}','\epsilon_{a_2}', '\epsilon_{a_3}', '\epsilon_{a_4}', '\epsilon_{a_5}', '\lambda_0']

parent_dir="/home/atridebchatterjee/reion_GPR/chain_storage_4_param/" #please change the '/home/atrideb' to the path of the folder where you keep the parent directory
MCMC_dir="chains_Aug18"


chain_storage_path = os.path.join(parent_dir, MCMC_dir)

os.makedirs(chain_storage_path, exist_ok=True)


from datetime import date

today = date.today()

# dd/mm/YY
d1 = today.strftime("%d_%m_%Y")


parent_dir="/home/atridebchatterjee/reion_GPR/chain_storage/" #please change the '/home/atrideb' to the path of the folder where you keep the parent directory
MCMC_dir="4_param_run"+d1

chain_storage_path = os.path.join(parent_dir, MCMC_dir)

os.makedirs(chain_storage_path, exist_ok=True)

file_root='GPR_MCMC'

write_file=Params.write_paramnames_ranges_file(params[:,1], params[:,2], params.keys, param_latex_name_arr, chain_storage_path, file_root)



like = Likelihood(CoreModule, Planck, Astro_dict, params[:,1], params[:,2])
like.setup(gammadatafile, Lymanlimitdatafile)



sampler =MCMCsampler(
                params= params, 
                likelihoodComputation=like, 
                filePrefix=str(d1), 
                chain_storage_path=chain_storage_path, 
                fileroot=file_root,
                walkersRatio=4,  
                sampleIterations=1000000,     #iteration for each worker
				Planck_dict=Planck
                )


print("start sampling")
sampler.startSampling()
print("done!")


