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

img_fmt = '.pdf'

# Identify data directory, rat IDs, and neural frq bands
root_dir = '/Users/maxd/' #/Users/iMax/'
rat_ids = ['R32_B7','R18_B12','R19_B11','R6_B10','R6_B16']
#rat_ids = ['R19_B11','R6_B10','R6_B16']
#rat_ids = ['R6_B10','R6_B16']
freq_bands = [u'B', u'G', u'HG', u'UHG', u'MUAR', u'Spk']
freq_bands = [u'B', u'G', u'HG', u'UHG', u'MUAR']
cl_bands = ['#663096','g','r','b','.2']
#ALTERNATE PURPLE: #9932CC
#rat_dir = root_dir + 'data/ToneAnalysisDatasets/'
#extract_save_dir = root_dir + 'data/nc_analysis/nc_datasets/'
data_dir = root_dir + 'data/nc_analysis/results/'
fig_dir = root_dir + 'data/nc_analysis/results/figures/nc_plot_nca4/'

if not os.path.exists(fig_dir):
	os.makedirs(fig_dir)

rat_ids = ['R32_B7']
#freq_bands = ['HG']
bl_markers = ['o','s','^','D','*']
amp_set = [5,6]

method = 'ld' #SPECIFY THE METHOD
data_dir = data_dir + method + '_nca4_amp' + ''.join([str(i) for i in amp_set]) + '/'





## Figure 1: (Per-block) Plot the [delta]StimFrq discriminability.
leg_lns = []
mx_dfs = [] #maximum ld(C)-ld(D)
se_dfs = [] #standard error at max dfs
sc_mx_dfs = [] #score at maximum mx_dfs
se_mx_dfs = [] #score standard error at maximum mx_dfs
b_inds = []
blbls = [] #bar plot labels
bclrs = []
bmrkrs = []
bind = -1
for rat_ind, rat_id in enumerate(rat_ids): # Iterate over each block and freq band and plot the diff_score
	bind = bind+1
	for frq_ind, freq_band in enumerate(freq_bands):
		bind = bind+1
		#Load the hdf5 file for this rat_id and freq_band
		fpath = data_dir + method + '_' + rat_id + '_' + freq_band + '_0.h5'
		f = h5py.File(fpath,'r')
		#pdb.set_trace()
		x = np.sort([int(i) for i in f['df_scores']])
		mudiff = [np.mean(f['df_scores'][str(i)]['diff_score']) for i in x]
		sddiff = [np.std(f['df_scores'][str(i)]['diff_score']) for i in x]
		semdiff = [stats.sem(f['df_scores'][str(i)]['diff_score']) for i in x]
		muscore = [np.mean(f['df_scores'][str(i)]['org_score']) for i in x]
		semscore = [stats.sem(f['df_scores'][str(i)]['org_score']) for i in x]
		#Plot the mudiff and semdiff as error bars
		plt.plot(x,mudiff,cl_bands[frq_ind],label=freq_band)
		plt.errorbar(x,mudiff,yerr=semdiff,fmt='none',ecolor=cl_bands[frq_ind])
		#Extract maximum mudiff
		try:
			max_i = np.argmax(mudiff)
			mx_dfs.append(mudiff[max_i]) #Maximum performance difference
			se_dfs.append(semdiff[max_i]) #Standard error at maximum performance difference
			sc_mx_dfs.append(muscore[max_i])
			se_mx_dfs.append(semscore[max_i])
		except:
			mx_dfs.append(0)
			se_dfs.append(0)
			sc_mx_dfs.append(0)
			se_mx_dfs.append(0)
		#Update bar chart settings
		blbls.append(rat_id + ' ' + freq_band)
		b_inds.append(bind)
		bclrs.append(cl_bands[frq_ind])
		bmrkrs.append(bl_markers[rat_ind])
	plt.axhline(0,linestyle='dashed',color='0.5')
	plt.xlabel(r'$\Delta StimFrq$')
	plt.ylabel(method + '(C) - ' + method + '(D)' )
	plt.title(method + ' ' + rat_id)
	plt.legend()
	plt.show(block=False)
	plt.savefig(fig_dir + method + '_' + rat_id + img_fmt)
	plt.close()


## Figure 2: (One Plot) Bar plot mean+-s.e at maximum [delta]lin_discrim for each block and neural frequency band
plt.figure(figsize=(24,10))
width = 1
x = range(1,len(mx_dfs)+1)
plt.bar(b_inds, mx_dfs, width, yerr=se_dfs, color=bclrs, ecolor='k')
plt.xticks(b_inds,blbls,rotation=45)
plt.ylabel('Max ' + method + '(C) - ' + method + '(D) w/ S.E.' )
plt.show(block=False)
plt.savefig(fig_dir + method + '_diff' + img_fmt)
plt.close()

## Figure 3: (One Plot) Bar plot mean+-s.e lin_discrim/reg_accuracy at point of maximum [delta]lin_discrim/reg_accuracy
plt.figure(figsize=(24,10))
width=1
x = range(1,len(sc_mx_dfs)+1)
plt.bar(b_inds, sc_mx_dfs, width, yerr=se_mx_dfs, color=bclrs, ecolor='k')
plt.xticks(b_inds,blbls,rotation=45)
plt.ylabel(method + '(C) score at max(' + method + '(C)-' + method + '(D) w/ S.E.' )
plt.show(block=False)
plt.savefig(fig_dir + method + '_score' + img_fmt)
plt.close()


###
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals,'0.5',linestyle='dashed')


def mxbline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals,'b')

## Figure 4: (One Plot) Scatter plot (mean score) vs (mean corr_score-decor_score)
plt.figure(figsize=(12,12))
width=1
nblks = len(rat_ids)
nfrqs = len(freq_bands)
ind = 0
for b in range(nblks):
	si = ind
	se = ind+nfrqs
	x = sc_mx_dfs[si:se]
	print(range(si,se))
	y = mx_dfs[si:se]
	x_err = se_mx_dfs[si:se]
	y_err = se_dfs[si:se]
	#m=[bl_markers[b]]*nfrqs
	s=[75]*len(x)
	#plt.scatter(x=x,y=y,c=bclrs[si:se],s=s,marker=bl_markers[b],label=rat_ids[b])
	plt.scatter(x=x,y=y,c=bclrs[si:se],s=s,marker='o',label=rat_ids[b])
	#pdb.set_trace()
	#for f in range(nfrqs):
		#plt.errorbar(x[f],y[f],yerr=y_err[f],xerr=x_err[f],fmt='none',elinewidth=2,ecolor=cl_bands[f],capsize=3,label=rat_ids[b])


	ind = se
	#pdb.set_trace()
plt.xlabel(method + '(C) score at max(' + method + '(C)-' + method + '(D) w/ S.E.' )
plt.ylabel('Max ' + method + '(C) - ' + method + '(D) w/ S.E.' )
plt.axhline(0,linestyle='dashed',color='0.5')
plt.axvline(0,linestyle='dashed',color='0.5')
plt.xlim((-.5,2.1))
plt.ylim((-.5,2.1))
abline(1,0)
[slope,intercept,rval,pval,stderr] = stats.linregress(sc_mx_dfs,mx_dfs)
r2 = rval*rval
print slope
print intercept
print rval*rval
print pval
mxbline(slope,intercept)
#pdb.set_trace()
#plt.legend()
plt.title('R^2: ' + str(r2) + ' , p= ' + str(pval))
plt.show(block=False)
plt.savefig(fig_dir + method + '_scatter' + img_fmt)
plt.close()
























