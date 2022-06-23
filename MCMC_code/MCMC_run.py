import sys, os
from os.path import join
import numpy as np
import os.path



from numpy import exp, array, arange


from utils import Params

from sampler import MCMCsampler
from Likelihood import Likelihood
from CoredataGenerator import CoreModule

Lymanlimitdatafile='/home/atrideb/reion_GPR/reion_21cm_code/ObsData/Lyman_limit.dat'
gammadatafile='/home/atrideb/reion_GPR/reion_21cm_code/ObsData/gamma_data_all_combined.dat'

Planck={'H0':67.70, 'ombh2': 0.0223, 'omch2': 0.12, 'ns':0.96, 'sigma8':0.81} 
Astro_dict={'esc_pop_III':0.0}
frac_size=0.001
init=np.array([0.00263737, 0.00335895, 0.00196963, 0.00302385, 0.00173669, 0.00318811]) # produced using 0.0036*np.random.rand(6)
params = Params(("epsilon_a0", [init[0], 0.0, 1.0, frac_size]),  #  #start  param_min param_max paramWidth
                ("epsilon_a1", [init[1], 0.0, 1.0, frac_size]),
                ("epsilon_a2", [init[2], 0.0, 1.0, frac_size]),
				("epsilon_a3", [init[3], 0.0, 1.0, frac_size]),
				("epsilon_a4", [init[4], 0.0, 1.0, frac_size]),
				("epsilon_a5", [init[5], 0.0, 1.0, frac_size]),
				("lambda_0",   [5.0, 0.0, 1.0, frac_size])
				)


param_latex_name_arr=['\epsilon_{a_0}','\epsilon_{a_1}','\epsilon_{a_2}', '\epsilon_{a_3}', '\epsilon_{a_4}', '\epsilon_{a_5}', '\lambda_0']

parent_dir="/home/atrideb/reion_GPR/chain_storage/" #please change the '/home/atrideb' to the path of the folder where you keep the parent directory
MCMC_dir="chains_June22"


chain_storage_path = os.path.join(parent_dir, MCMC_dir)

os.makedirs(chain_storage_path)


file_root='GPR_MCMC'

write_file=Params.write_paramnames_ranges_file(params[:,1], params[:,2], params.keys, param_latex_name_arr, chain_storage_path, file_root)



like = Likelihood(CoreModule, Planck, Astro_dict, params[:,1], params[:,2])
like.setup(gammadatafile, Lymanlimitdatafile)



sampler =MCMCsampler(
                params= params, 
                likelihoodComputation=like, 
                filePrefix="June_22", 
                chain_storage_path=chain_storage_path, 
                fileroot=file_root,
                walkersRatio=4,  
                sampleIterations=1000000,     #iteration for each worker
				Planck_dict=Planck
                )


print("start sampling")
sampler.startSampling()
print("done!")


