# This code has been taken from  CosmoHammer (proper citation to be made in the paper when published) with lots of Modification as required


import emcee
from schwimmbad import MPIPool
from multiprocessing import Pool
import numpy as np
import time
import logging



import file_configure as c
from utils import getLogger

class MCMCsampler():
	
	def __init__(self, params, likelihoodComputation, filePrefix, chain_storage_path, fileroot, walkersRatio,
				sampleIterations,  Planck_dict, threadCount=1, logLevel=logging.INFO, pool=None):
		"""
		MCMC sampler implementation

		"""
		self.params = params
		self.paramValues=self.params[:,0]
		self.paramWidths=self.params[:,3]
		self.likelihoodComputationChain = likelihoodComputation
		self.walkersRatio = walkersRatio
		self.filePrefix = filePrefix
		self.chain_storage_path=chain_storage_path
		self.file_root=fileroot
		self.threadCount = threadCount
		self.paramCount = len(self.paramValues)
		self.nwalkers = self.paramCount*walkersRatio
		self.sampleIterations = sampleIterations

		if not hasattr(self.likelihoodComputationChain, "params"):
			self.likelihoodComputationChain.params = params
        
		# setting up the logging
		self._configureLogging(filePrefix+c.LOG_FILE_SUFFIX, logLevel)
		
		# The sampler object
		self._sampler = self.createEmceeSampler(self.likelihoodComputationChain)




	def InitPosGenerator(self):
		return  [self.paramValues+np.random.normal(size=self.paramCount)*self.paramWidths for i in range(self.nwalkers)]
		

	def startSampling(self):
		for k in range(self.nwalkers):
			f=open(self.chain_storage_path+'/'+self.file_root+ "_"+str(k+1)+".txt", "w+")
			f.write('# Weight, Lnprob, different params\n')
			f.close()

		pos = self.InitPosGenerator()
		prob = None
		rstate = None
		datas = None
		self.log("start sampling, we did not use burn in. We will remove the burn-in step while producing the corner plot")
		start = time.time()
		'''
		# Make sure the thread we're running on is the master...copy and paste this
		with MPIPool() as pool:
			if not pool.is_master():
    				pool.wait()
    				sys.exit(0)
    			self._sampler = self.createEmceeSampler(self.likelihoodComputationChain, pool) replace self._sampler in _init_
			self.sample(pos, prob, rstate, datas)
			end = time.time()
			self.log("sampling done! Took: " + str(round(end-start,4))+"s")
    			
    			'''
		self.sample(pos, prob, rstate, datas)
		end = time.time()
		self.log("sampling done! Took: " + str(round(end-start,4))+"s")




	def createEmceeSampler(self, callable, **kwargs):  # pool inside 
		"""
		create the emcee sampler
		"""
		return emcee.EnsembleSampler(self.nwalkers, 
                                     self.paramCount, 
                                     callable, 
                                     threads=self.threadCount,
                                     **kwargs)
# replace threadcount with pool=pool
	def sample(self, InitPos, burninProb=None, burninRstate=None, datas=None):
		"""
		Starts the sampling process
		"""
		print('MCMC sampling with nwalkers=%s, Iterations=%s, Number of parameters=%s\n'%(self.nwalkers, self.sampleIterations, self.paramCount))

		nblobs=2 #Number of blobs one wants to use in emcee
		samplenumber=self.sampleIterations
		chain_arr = np.empty([self.nwalkers, samplenumber, self.paramCount])
		lnprob_arr = -np.inf * np.ones([self.nwalkers, self.sampleIterations])
		blobs_arr = np.empty([self.nwalkers, self.sampleIterations, nblobs])

		step_index = 0
		save_steps_start = 0
		counter = 1
		save_steps=10
		for pos, prob, stat, blobs in self._sampler.sample(InitPos,iterations=self.sampleIterations):
			chain_arr[:, step_index, :] = pos
			lnprob_arr[:, step_index] = prob         
			blobs_asarray=np.asarray(blobs)
			blobs_arr[:, step_index] =blobs_asarray
			step_index=step_index+1
			if (np.remainder(step_index, save_steps) == 0):
				for k in range(self.nwalkers):
					print('step,save_step',step_index,save_steps)
					f=open(self.chain_storage_path+'/'+self.file_root+ "_"+str(k+1)+".txt", "a")
					for i in range(save_steps_start, save_steps_start + save_steps):
						s = "{0:6d}".format(1)  ### dummy for weight
						s += " " + "{:.6e}".format(-lnprob_arr[k,i])
						for kk in range(self.paramCount):
							s += " " + "{:.6e}".format(chain_arr[k,i,kk])
						for kk in range(len(blobs_arr[k,i,:])):
							s += " " + "{:.6e}".format(blobs_arr[k,i,kk])
						s += "\n"
						f.write(s)

					f.close()
				save_steps_start = save_steps_start + save_steps

                
			if(counter%1==0):
				self.log("Iteration finished with total sample Number " + str(counter*self.nwalkers) + '\n') #Sample number from the MCMC after every step
                
			counter = counter + 1


	def log(self, message, level=logging.INFO):
		"""
		Logs a message to the logfile
		"""
		getLogger().log(level, message)


	def _configureLogging(self, filename, logLevel):
		logger = getLogger()
		logger.setLevel(logLevel)
		fh = logging.FileHandler(filename, "w")
		fh.setLevel(logLevel)
		# create console handler with a higher log level
		ch = logging.StreamHandler()
		ch.setLevel(logging.ERROR)
		# create formatter and add it to the handlers
		formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
		fh.setFormatter(formatter)
		ch.setFormatter(formatter)
		# add the handlers to the logger
		for handler in logger.handlers[:]:
			logger.removeHandler(handler)
		logger.addHandler(fh)
		logger.addHandler(ch)



