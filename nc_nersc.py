#!/usr/local/bin/python

# Import Libraries
from nc_analysis import extract_noise_correlation_dataset, noise_correlation_analysis
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py

t_start = time.time()
# MPI Rank to parallelize on NERSC
#TODO: Implement this
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

# Datasets
##WARNING: LINEAR DISCRIM FAILS ON R19_B11-Spk
root_dir = '/Users/maxd/' #/Users/iMax/'
rat_ids = ['R32_B7','R18_B12','R19_B11','R6_B10','R6_B16']
rat_ids = ['R6_B10','R6_B16']
freq_bands = [u'B', u'G', u'HG', u'UHG', u'Spk']
rat_dir = root_dir + 'data/ToneAnalysisDatasets/'
extract_save_dir = root_dir + 'data/nc_analysis/nc_datasets/'
analysis_save_dir = root_dir + 'data/nc_analysis/results/'
fig_dir = root_dir + 'data/nc_analysis/results/figures/'

rat_ids = ['R32_B7']
freq_bands = ['HG']

nanls = len(rat_ids)*len(freq_bands)# Number of Analyses to perform
print nanls
## TESTING: Hard coded parameters
if len(sys.argv) < 3:
	rat_id = 'R6_B10'
	freq_band = 'HG'
else:
	rat_id = sys.argv[1]
	freq_band = sys.argv[2]

for rat_id in rat_ids:
	for freq_band in freq_bands:
		#Extraction Parameters
		twnd = (40,80)
		FORCE_EXTRACT = 0;

		#Analysis Parameters
		method = 'lr'
		amp_set = [5, 6]
		frq_set = [] #Not currently implemented. Uses full range.

		### Extract response if yet incomplete, or if FORCE_EXTRACT==1
		print(rat_id, freq_band, str(twnd))
		epath = extract_save_dir + rat_id + '_' + freq_band + '_ext_rsp.h5'
		if not os.path.exists(epath) or FORCE_EXTRACT: 
			if not os.path.exists(extract_save_dir): #Create the parent directory if it does not yet exist
				os.makedirs(extract_save_dir)
			extr_rsp = extract_noise_correlation_dataset(rat_dir, rat_id, freq_band)
			#Save extracted response
			hfe = h5py.File(epath,'w')
			final_rsp = hfe.create_dataset('final_rsp',data=extr_rsp[0])
			bf_el = hfe.create_dataset('bf_el',data=extr_rsp[1])
			final_rsp.attrs['rat_id'] = rat_id
			final_rsp.attrs['freq_band'] = freq_band
			final_rsp.attrs['twnd'] = str(twnd)
			final_rsp = np.array(final_rsp) #Convert to np array for analysis
			bf_el = np.array(bf_el) #Convert to np array for analysis
			hfe.close()
			print('final_response saved')
		else:
			#Load extracted response if it already exists
			hfe = h5py.File(epath,'r')
			final_rsp = np.array(hfe.get('final_rsp'))
			bf_el = np.array(hfe.get('bf_el'))
			hfe.close()
			print('final_response loaded')



		### Perform NC Analysis

		print "Analysis Method: %s"%(method)
		results = noise_correlation_analysis(final_rsp, bf_el, method, twnd, amp_set, frq_set, fig_dir)

		#Save Results to HDF5 file
		apath = analysis_save_dir + method + '/' + method + '_' + rat_id +'_' + freq_band + '_' + str(mpi_rank) + '.h5'
		hfa = h5py.File(apath,'w') #h5 noise correlation
		decor = hfa.create_dataset('decor_focused_response',data=results[0])
		scores = hfa.create_dataset('scores',data=results[1])
		scores.attrs['rat_id'] = rat_id
		scores.attrs['freq_band'] = freq_band
		scores.attrs['method'] = method
		scores.attrs['amp_set'] = str(amp_set)
		scores.attrs['frq_set'] = str(frq_set)
		hfa.close()
		print "Runtime: %s seconds"%(str(round(time.time()-t_start)))
		#Plot the results
		discrim_x_examp = results[3]
		discrim_y_examp = results[4]
		test_dict = results[5]
		plt.scatter(x = np.array(discrim_x_examp), y = np.array(discrim_y_examp))
		plt.plot([i for i in test_dict],[np.mean(np.array(test_dict[i])) for i in test_dict])
		plt.axhline(0)
		plt.show(block=False)
		plt.savefig(fig_dir + 'nc_' + method + '_' + rat_id + '_' + freq_band + '.png')
		plt.close()






# print('')












