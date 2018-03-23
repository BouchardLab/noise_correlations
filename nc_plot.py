"""
nc_plot.py

Created on: 3.21.18
"""
# Import Libraries
import pdb
import sys
import os
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py

# Identify data directory, rat IDs, and neural frq bands
root_dir = '/Users/maxd/' #/Users/iMax/'
rat_ids = ['R32_B7','R18_B12','R19_B11','R6_B10','R6_B16']
#rat_ids = ['R19_B11','R6_B10','R6_B16']
#rat_ids = ['R6_B10','R6_B16']
freq_bands = [u'B', u'G', u'HG', u'UHG', u'MUAR', u'Spk']
freq_bands = [u'B', u'G', u'HG', u'UHG', u'MUAR']
cl_bands = ['#663096','g','r','b','.2']
#rat_dir = root_dir + 'data/ToneAnalysisDatasets/'
#extract_save_dir = root_dir + 'data/nc_analysis/nc_datasets/'
data_dir = root_dir + 'data/nc_analysis/results/'
fig_dir = root_dir + 'data/nc_analysis/results/figures/nc_plot/'

#rat_ids = ['R32_B7']
#freq_bands = ['HG']

method = 'ld' #SPECIFY THE METHOD
data_dir = data_dir + method + '/'


# Iterate over each block and freq band and plot the diff_score
leg_lns = []
mx_dfs = [] #maximum ld(C)-ld(D)
se_dfs = [] #standard error at max dfs
b_inds = []
blbls = [] #bar plot labels
bclrs = []
bind = -1
for rat_id in rat_ids:
	bind = bind+1
	for frq_ind, freq_band in enumerate(freq_bands):
		bind = bind+1
		fpath = data_dir + 'ld_' + rat_id + '_' + freq_band + '_0.h5'
		f = h5py.File(fpath,'r')

		x = np.sort([int(i) for i in f['df_scores']])
		mudiff = [np.mean(f['df_scores'][str(i)]['diff_score']) for i in x]
		sddiff = [np.std(f['df_scores'][str(i)]['diff_score']) for i in x]
		semdiff = [stats.sem(f['df_scores'][str(i)]['diff_score']) for i in x]

		plt.plot(x,mudiff,cl_bands[frq_ind],label=freq_band)
		plt.errorbar(x,mudiff,yerr=semdiff,fmt='none',ecolor=cl_bands[frq_ind])
		#Extract maximum mudiff
		try:
			max_i = np.argmax(mudiff)
			mx_dfs.append(mudiff[max_i])
			se_dfs.append(semdiff[max_i])
		except:
			mx_dfs.append(0)
			se_dfs.append(0)
		blbls.append(rat_id + ' ' + freq_band)
		b_inds.append(bind)
		bclrs.append(cl_bands[frq_ind])
	plt.axhline(0,linestyle='dashed',color='0.5')
	plt.xlabel(r'$\Delta StimFrq$')
	plt.ylabel(method + '(C) - ' + method + '(D)' )
	plt.title(method + ' ' + rat_id)
	plt.legend()
	plt.show(block=False)
	plt.savefig(fig_dir + method + '_' + rat_id + '.png')
	plt.close()

plt.figure(figsize=(24,10))
width = 1
x = range(1,len(mx_dfs)+1)
plt.bar(b_inds, mx_dfs, width, yerr=se_dfs, color=bclrs, ecolor='k')
plt.xticks(b_inds,blbls,rotation=45)
plt.ylabel('Max ' + method + '(C) - ' + method + '(D) w/ S.E.' )
plt.show(block=False)
plt.savefig(fig_dir + method + '.png')
plt.close()























