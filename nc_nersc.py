#!/usr/local/bin/python

# Import Libraries

from nc_analysis2 import extract_noise_correlation_dataset, noise_correlation_analysis
import pdb
import sys
import os
import time
import numpy as np
from scipy import stats
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
#rat_ids = ['R19_B11','R6_B10','R6_B16']
#rat_ids = ['R6_B10','R6_B16']
freq_bands = [u'B', u'G', u'HG', u'UHG', u'MUAR', u'Spk']
freq_bands = [u'B', u'G', u'HG', u'UHG', u'MUAR']
rat_dir = root_dir + 'data/ToneAnalysisDatasets/'
extract_save_dir = root_dir + 'data/nc_analysis/nc_datasets/'
analysis_save_dir = root_dir + 'data/nc_analysis/results/'
fig_dir = root_dir + 'data/nc_analysis/results/figures/40_60/'

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
		twnd = (40,60) #45,53
		FORCE_EXTRACT = 0;

		#Analysis Parameters
		method = 'ld'
		amp_set = [5, 6]
		frq_set = [] #Not currently implemented. Uses full range.

		### Extract response if yet incomplete, or if FORCE_EXTRACT==1
		print(rat_id, freq_band, str(twnd))
		epath = extract_save_dir + rat_id + '_' + freq_band + '_ext_rsp.h5'
		if not os.path.exists(epath) or FORCE_EXTRACT: 
			if not os.path.exists(extract_save_dir): #Create the parent directory if it does not yet exist
				os.makedirs(extract_save_dir)
			extr_rsp = extract_noise_correlation_dataset(rat_dir, rat_id, freq_band,twnd)
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

		#Plot the results
		# discrim_x_examp = results['discrim_x_examp']
		# discrim_y_examp = results['discrim_y_examp']
		# test_dict = results['test_dict']
		# plt.scatter(x = np.array(discrim_x_examp), y = np.array(discrim_y_examp))
		# plt.plot([i for i in test_dict],[np.mean(np.array(test_dict[i])) for i in test_dict])
		# plt.axhline(0)
		# plt.show(block=False)
		# plt.savefig(fig_dir + 'nc_' + method + '_' + rat_id + '_' + freq_band + '.png')
		# plt.close()

		##Fig1: Plot the Corr and Decorr LDs as function of stimulus frequency difference
		f, axarr = plt.subplots(1, 2,figsize=(12,5))
		x = [i for i in results['df_scores']]
		pdb.set_trace()
		org_y = [np.mean(np.array(results['df_scores'][i]['org_score'])) for i in results['df_scores']]
		org_ysd = [np.std(np.array(results['df_scores'][i]['org_score'])) for i in results['df_scores']]
		#org_ysem = [stats.sem(np.array(results['df_scores'][i]['org_score'])) for i in results['df_scores']]
		decor_y = [np.mean(np.array(results['df_scores'][i]['decor_score'])) for i in results['df_scores']]
		decor_ysd = [np.std(np.array(results['df_scores'][i]['decor_score'])) for i in results['df_scores']]
		#decor_ysem = [stats.sem(np.array(results['df_scores'][i]['decor_score'])) for i in results['df_scores']]
		diff_y = [np.mean(np.array(results['df_scores'][i]['diff_score'])) for i in results['df_scores']]
		diff_ysd = [np.std(np.array(results['df_scores'][i]['diff_score'])) for i in results['df_scores']]
		#diff_ysem = [stats.sem(np.array(results['df_scores'][i]['diff_score'])) for i in results['df_scores']]

		axarr[0].plot(x, org_y,'g',linewidth=2)
		axarr[0].errorbar(x,org_y,yerr=org_ysd,fmt='none',ecolor='g',elinewidth=2)
		axarr[0].plot(x, decor_y, 'r',linewidth=1)
		axarr[0].errorbar(x,decor_y,yerr=decor_ysd,fmt='none',ecolor='r',elinewidth=1)
		axarr[0].axhline(0)
		axarr[0].set_title('Corr LD (green), Decorr LD (red) as f(diff. in Stim Frq)')
		axarr[1].plot(x, diff_y, 'b',linewidth=2)
		axarr[1].errorbar(x,diff_y,yerr=diff_ysd,fmt='none',ecolor='b',elinewidth=2)
		axarr[1].axhline(0)
		axarr[1].set_title('(Corr LD)-(Decorr LD) as f(diff. in Stim Frq)')
		#pdb.set_trace()
		f.suptitle(method + '_' + rat_id + '_' + freq_band)
		plt.show(block=False)
		plt.savefig(fig_dir + 'nc_' + method + '_' + rat_id + '_' + freq_band + '.png')
		plt.close()


		### Save Results to HDF5 file
		apath = analysis_save_dir + method + '/' + method + '_' + rat_id +'_' + freq_band + '_' + str(mpi_rank) + '.h5'
		hfa = h5py.File(apath,'w') #h5 noise correlation
		decor = hfa.create_dataset('decor_focused_response',data=results['decor_focused_response'])
		# Save the scores for each electrode and stimulus pair
		sg = hfa.create_group('scores')
		for f1 in results['scores'].keys():
			f1g = sg.create_group(str(f1))
			for f2 in results['scores'][f1].keys():
				f2g = f1g.create_group(str(f2))
				el_pairs = results['scores'][f1][f2]['el_pairs']
				stim_pairs = results['scores'][f1][f2]['stim_pairs']
				org_score = results['scores'][f1][f2]['org_score']
				decor_score = results['scores'][f1][f2]['decor_score']
				f2g.create_dataset('el_pairs',data=el_pairs)
				f2g.create_dataset('stim_pairs',data=stim_pairs)
				f2g.create_dataset('org_score',data=org_score)
				f2g.create_dataset('decor_score',data=decor_score)
		sg.attrs['rat_id'] = rat_id
		sg.attrs['freq_band'] = freq_band
		sg.attrs['method'] = method
		sg.attrs['amp_set'] = str(amp_set)
		sg.attrs['frq_set'] = str(frq_set)

		# Save the scores arranged by the frequency difference between two given stimuli
		dg = hfa.create_group('df_scores')
		for df in results['df_scores'].keys():
			dfg = dg.create_group(str(df))
			diff_score = results['df_scores'][df]['diff_score']
			el_pairs = results['df_scores'][df]['el_pairs']
			decor_score = results['df_scores'][df]['decor_score']
			org_score = results['df_scores'][df]['org_score']
			stim_pairs = results['df_scores'][df]['stim_pairs']
			dfg.create_dataset('diff_score',data=diff_score)
			dfg.create_dataset('el_pairs',data=el_pairs)
			dfg.create_dataset('decor_score',data=decor_score)
			dfg.create_dataset('org_score',data=org_score)
			dfg.create_dataset('stim_pairs',data=stim_pairs)

		#hfa.create_dataset('test_dict',data=test_dict)
		hfa.close()
		print "Runtime: %s seconds"%(str(round(time.time()-t_start)))

		






# print('')












