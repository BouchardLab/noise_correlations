
"""

Created on: 3.1.18
"""
import time
import numpy as np
import scipy.io
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib
import sklearn.linear_model
import h5py    
from itertools import combinations
import cPickle as pickle
import os
#from __future__ import print_function

##########################################################
def save_object(obj, filename):
    """
    save as filename, increment run order to filename
    
    Parameters
    ----------
    obj : numpy array
    filename : str
    """
    # i = 0
    # while os.path.exists(filename + "_" + str(i) + ".pkl"):
    #     i += 1
    # with open(filename + "_" + str(i) + ".pkl", 'wb') as output:  # Overwrites any existing file.
    #         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    with open(filename + ".pkl", 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filepath):
    """
    load data
    
    Parameters
    ----------
    filepath : str
    """
    if os.path.exists(filepath):
        with open(filepath) as file_input:
            return pickle.load(file_input)
    else:
        return None

### Electrode Pair Linear Discriminability Function
def pair_el_discrim_value(responses,stim_1, stim_2, electrodes, n_samples = 40, cutoff = False,adjust_volume = False):
    ###### COMMENT MORE!!!!!!
    
    #trial avg response to stimulus
    n1 = np.mean(responses[:,electrodes,stim_1],axis = 0)
    n2 = np.mean(responses[:,electrodes,stim_2],axis = 0)
    #print n1, n1.shape
    #print n2, n2.shape
    cov = np.cov(responses[:,electrodes,:][:,:,[stim_1,stim_2]].transpose((0,2,1)).reshape(n_samples*2,-1).transpose())
    #print cov, cov.shape
    #cov_inv = np.linalg.inv(cov)
    #print cov_inv, cov_inv.shape
    volume_factor = 1
    if adjust_volume:
        ### Think about volume factor
        volume_factor = np.sqrt(np.linalg.det(cov)/np.linalg.det(np.diag(np.diag(cov))))
    try:
        value = np.dot((n2-n1).transpose(),np.dot(np.linalg.inv(cov),(n2-n1))) 
    except:
        print electrodes
        print stim_1
        print stim_2
        print responses[:,electrodes,stim_1]
        print n1
        print responses[:,electrodes,stim_2]
        print n2
        print (n2-n1).shape
        print (cov).shape
        raise
    decor_value = np.dot((n2-n1).transpose(),np.dot(np.linalg.inv(np.diag(np.diag(cov*volume_factor))),(n2-n1)))
    return value, decor_value


##########################################################
# Necessary Parameters:
#   Extraction: rat_dir, rat_id, freq_band, dset_ext, final_response_path, 
#
#   Analysis: rat_dir, rat_id, freq_band, final_response_path, 
#      results_dir, tmwnd, amp_set, frq_set, method
#
##############################################

##### Specify datasets and locations #####
##TODO: PARAMETERIZE

rat_ids = ['R32_B7','R18_B12','R19_B11','R6_B10','R6_B16']
rat_dir = "/Users/iMax/data/ToneAnalysisDatasets/"
freq_bands = [u'B', u'G', u'HG', u'UHG', u'MUAR', u'Spk']

#TODO: PARAMETERIZE
rat_id = rat_ids[0] #this is where you would change the rat
freq_band = freq_bands[3] #this is where we would change the high gamma


def extract_noise_correlation_dataset(rat_dir, rat_id, freq_band):
    print "Extracting noise correlation dataset for: %s"%(rat_id)
    print "Neural Frequency Band: %s"%(freq_band)
    rat_file = rat_dir + rat_id + '.WVL_CAR1.TN.RspM_TM.mat' #dset_ext
    f1 = h5py.File(rat_file,'r+') 

    # Select the target rat, neural frequency band, and stimulus parameters
    data = np.array(f1['RspM'][freq_band])
    stim_vals = np.array(f1['stimVls'])

    # Load dataset identifying the "Best Frequency" for each tuned electrode
    #TODO: this is needed for other rats? (for best frequencies)
    f2 = h5py.File(rat_dir + 'TN.FRA.anl.mat','r+')
    data2 = f2['frall'][freq_band][rat_id]
    #best frequencies
    bf_val = np.array(data2['bf_fra'])

    #initialize results, these are saved in save_dir
    results = []


    # Specify a save directory for analysis results
    #TODO: PARAMETERIZE THIS!!!!!!!!!
    #save_dir = '/Users/iMax/data/nc_analysis/results/' + rat_id + '/' + freq_band + '/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # Create sub directory for each rat and frequency band
    #TODO: IS THIS DOING ANYTHING?
    # for dir_1 in rat_ids:
    #     for dir_2 in freq_bands:
    #         directory = 'results/' + dir_1 + '/' + dir_2 + '/runs/'
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)


    #specific electrode
    electrode_number = np.array(data2['fb_tuned'])

    bf_electrodes = [[i[1],i[0]] for i in zip(list(bf_val.flatten()),list(electrode_number.flatten()))]

    #electrode number, frequency with largest response
    #calculated by looking at the freq-amp tuning map for each electrode and taking a weighted average
    bf_el = np.array(bf_electrodes)
    electrode_length = bf_el.shape[0]


    # Get metaparameters from dataset size
    time_length,amp_fq_length, total_electrode_length = data.shape


    #### FIGURE 1: Example neural tone response
    # print data.shape
    # #need to use stim_vals to figure out the corresponding stim
    # sample_number = 4004
    # electrode_number = 19
    # plt.plot(data[:,sample_number,electrode_number])


    # plt.vlines([40,80],-1,1)

    # plt.xlabel('time')
    # plt.ylabel('zscored wavelet decomp amp in '+ freq_band)
    # plt.title("sample timeseries of electrode data 100 timepoints after stimulus (" + str(stim_vals[:,sample_number]) + "} onset and electrode "+ str(electrode_number))
    # plt.show(block=False)
    #############


    ## Find the peak response for each stimulus presentation
    peak_response = np.zeros((4800,total_electrode_length))
    # peak response calculated for 40 to 80 sample points after stimulus onset
    #TODO: PARAMETERIZE THE TIMEWINDOW ???!!!!
    #does 40 to 80!
    for i in xrange(amp_fq_length):
        for j in xrange(total_electrode_length):    
            peak_response[i][j] = np.max(data[40:80,i,j])


    ## Print analysis parameters to verify a run
    #TODO: Make sure this carries input parameter information
    electrodes_used = [int(x) for x in list(bf_el[:,0] -1)]
    uniq_freqs = np.sort(list(set(stim_vals[1,:])))
    uniq_amps = np.sort(list(set(stim_vals[0,:])))
    num_of_freq = len(uniq_freqs)
    num_of_amp = len(uniq_amps)
    num_of_stims = 20
    num_of_electrodes = len(electrodes_used)
    num_of_stims_used = 40
    num_of_samples = 20*2
    print 'bf_el', bf_el.shape, ": 126 electrodes, (electrode number, best frequency)"
    print 'peak_response', peak_response.shape, ": 4800 samples (30 freq * 8 amplitude 20 stim samples), 128 electrodes"
    print 'stim_vals', stim_vals.shape, ": (stim amplitude, stim frequency), 4800 samples (30 freq * 8 amplitude 20 stim samples)"
    print 'electrodes_used', num_of_electrodes, ": list of integer values of electrodes actually used"
    print 'electrode_length: ',electrode_length
    print 'amp_fq_length: ', amp_fq_length
    print 'time_length: ', time_length
    print 'total_electrode_length: ', total_electrode_length
    print 'num_of_amp: ', num_of_amp
    print 'num_of_freq: ', num_of_freq
    print 'num_of_stims: ', num_of_stims
    print 'num_of_stims_used', num_of_stims_used
    print 'num_of_samples', num_of_samples


    ## Generate "Final Response", the cleaned dataset used for noise-correlation analysis
    final_response = np.zeros((num_of_electrodes,num_of_stims,num_of_freq,num_of_amp))
    #TODO: PARAMETERIZE THIS
    #file_path = save_dir + rat_id + '_' + freq_band + '_final_response.pkl'
    #enter the for-loop of doom // to clean
    #if not os.path.exists(file_path):
    #cycle electrodes 
    for electrode_counter, electrode_num in enumerate(electrodes_used):
        stim_counter = np.zeros((num_of_freq,num_of_amp))
        #cycle 4800 samples (30 freq * 8 amplitude 20 stim samples) for stim_vals
        for amp_fq_num in xrange(amp_fq_length):
            #cycle through volume amplitudes
            for amp_counter, amp in enumerate(uniq_amps):
                #cycle through stimulus frequencies
                for freq_counter, freq in enumerate(uniq_freqs):
                    #integrating two data sets from stim_vals and peak_response
                    if stim_vals[0,amp_fq_num]== amp and stim_vals[1,amp_fq_num] == freq:
                        #print electrode_num, stim_counter, freq_counter, amp_counter, amp_fq_num, peak_response[amp_fq_num,electrode_num] 
                        final_response[electrode_counter,int(stim_counter[freq_counter,amp_counter]),freq_counter,amp_counter] = peak_response[amp_fq_num,electrode_num]
                        #this number should equal the number of stimuli, e.g. 20
                        stim_counter[freq_counter,amp_counter] = stim_counter[freq_counter,amp_counter]+1
                        #print stim_counter[freq_counter,amp_counter]
        print "passed electrode", electrode_num
        
    # save_object(final_response,save_dir + rat_id + '_' + freq_band + '_final_response')
    # save_object(bf_el,save_dir + rat_id + '_' + freq_band + '_bf_el')
    #print "final_response saved"
    #else:
        #final_response = load_object(file_path)
        #print "final_response loaded"
    #print len(set(stim_vals[1,:]))
    return (final_response, bf_el)




#####################################
# SPLIT SCRIPT HERE. 
#   ABOVE: Function generates final_response and saves it.
#       Saves: final_response, bf_el
#   BELOW: Function loads final_response and performs nc_analysis
#####################################

def noise_correlation_analysis(final_response, bf_el, method, twnd, amp_set, frq_set, fig_dir):
    num_of_samples = 40
    # Load Final Response
    # final_response = load_object(dataset_path + rat_id + '_' + freq_band + '_final_response.pkl')
    print final_response.shape
    # bf_el = load_object(dataset_path + rat_id + '_' + freq_band + '_bf_el.pkl')



    num_of_electrodes = bf_el.shape[0]
    results = []
    #sorting to get electrodes with corresponding sorted electrodes
    electrode_bestfreq_sort_correspondence = np.argsort(bf_el, axis = 0)
    #print electrode_bestfreq_sort_correspondence

    #electrodes organized by electrode best frequency
    sorted_response = final_response[electrode_bestfreq_sort_correspondence[:,1]]




    ### Generate the focused response and z-score
    #TODO: Potential change try with out Z SCORING !!!!!!!!!!!!
    #TODO: Focused_response hardcodes the stimulus amplitudes. MAKE THIS MORE FLEXIBLE.
    #TODO: PARAMETERIZE THIS.
    print "min/max best freq vals:", np.min(bf_el[:,1]),np.max(bf_el[:,1])
    # rearraged so (126 electrodes, 20 samples, 30 stimuli, 8 attenuations) turn into
    # (40 samples from -10 and -20 attenuations, 126 electrodes, 19 frequecies within electrode best freq vals)
    # number of frequecies within electrode best freq vals
    num_of_stims_used = 40
    b_stim_start = int(np.min(bf_el[:,1]))
    b_stim_end = int(np.max(bf_el[:,1])+1)
    num_of_best_stims = b_stim_end - b_stim_start
    #print 'num_of_stims_used: ', num_of_stims_used
    print 'num_of_best_stims: ', num_of_best_stims

    #collect only -10 and -20 attenuations
    #use only stimuli whose freqs are within electrode best freq values 
    focused_response = np.transpose(final_response,(1,3,0,2))[:,5:7,:,b_stim_start:b_stim_end].reshape(num_of_stims_used,num_of_electrodes,-1)

    #mean and std calculated for each electrode accross all peak responses
    peak_electrode_mean = np.dot(np.mean(final_response[:,:,:,5:7].reshape(num_of_electrodes,-1),axis =1).reshape(num_of_electrodes,1),np.ones([1,num_of_stims_used*num_of_best_stims]))
    peak_electrode_std = np.dot(np.std(final_response[:,:,:,5:7].reshape(num_of_electrodes,-1),axis =1).reshape(num_of_electrodes,1),np.ones([1,num_of_stims_used*num_of_best_stims]))
    peak_electrode_mean = np.transpose(peak_electrode_mean.reshape(num_of_electrodes,40,num_of_best_stims),(1,0,2))
    peak_electrode_std = np.transpose(peak_electrode_std.reshape(num_of_electrodes,40,num_of_best_stims),(1,0,2))

    #z-scored electrode peak response values
    ###### IF YOU DON'T WANT ZSCORE GET RID OF MEAN AND STD
    ########## POTENTIAL CHANGE COMMENT OUT!!!!!
    focused_response = ((focused_response-peak_electrode_mean)/peak_electrode_std)
    #print('FOCUSED RESPONSE IS NOT BEING Z-SCORED')

    ##### Figure 2: electrode pair response to one tone stimulus frequency
    # el_x = 17
    # el_y = 99
    # stim_1 = 5
    # plt.scatter(x=focused_response[:,el_x,stim_1].flatten(),y=focused_response[:,el_y,stim_1].flatten(),c='red',marker='s')
    # plt.xlabel('peak response from electrode ' + str(el_x))
    # plt.ylabel('peak response from electrode ' + str(el_y))
    # plt.title('response from multiple exposures to stimulus ' + str(stim_1))
    # plt.show()
    ###########

    ##### Figure 3: electrode pair response to 2 tone stimulus frequencies
    # el_x = 17
    # el_y = 99
    # stim_1 = 5
    # stim_2 = 17
    # plt.scatter(x=focused_response[:,el_x,stim_1].flatten(),y=focused_response[:,el_y,stim_1].flatten(),c='red',marker='s',label='stim '+str(stim_1))
    # plt.scatter(x=focused_response[:,el_x,stim_2].flatten(),y=focused_response[:,el_y,stim_2].flatten(),c='navy',label = 'stim '+str(stim_2))
    # plt.scatter(x=np.mean(focused_response[:,el_x,stim_1]).flatten(),y=np.mean(focused_response[:,el_y,stim_1].flatten()),marker='x',s=250, c ='red')
    # plt.scatter(x=np.mean(focused_response[:,el_x,stim_2]).flatten(),y=np.mean(focused_response[:,el_y,stim_2].flatten()),marker='x',s=250)
    # plt.legend()
    # plt.xlabel('peak response from electrode ' + str(el_x))
    # plt.ylabel('peak response from electrode ' + str(el_y))
    # plt.show()
    ###################


    ## Permuting accross samples within electrode and stimuli
    decor_focused_response = np.zeros(focused_response.shape)
    for i in range(focused_response.shape[2]):
        for j in range(focused_response.shape[1]):
            sample_perm = np.random.permutation(np.array(range(40)))
            #sample_perm = np.array(range(40))
            for k in range(focused_response.shape[0]):
                decor_focused_response[k,j,i]=focused_response[sample_perm[k],j,i]
    ##### Figure 4: Decorrelated response after shuffling stimulus responses
    # f, axarr = plt.subplots(1, 2,figsize=(12,5))
    # el_x = 17
    # el_y = 99
    # stim_1 = 5
    # stim_2 = 17
    # # add a print with fr_stim_vals[stim_1],fr_stim_vals[stim_2]
    # axarr[0].scatter(x=focused_response[:,el_x,stim_1].flatten(),y=focused_response[:,el_y,stim_1].flatten(),c='red')
    # axarr[0].scatter(x=focused_response[:,el_x,stim_2].flatten(),y=focused_response[:,el_y,stim_2].flatten())
    # axarr[0].scatter(x=np.mean(focused_response[:,el_x,stim_1]),y=np.mean(focused_response[:,el_y,stim_1]),marker='x',s=200, c ='red')
    # axarr[0].scatter(x=np.mean(focused_response[:,el_x,stim_2]),y=np.mean(focused_response[:,el_y,stim_2]),marker='x',s=200)
    # axarr[0].set_title("original data")
    # axarr[1].scatter(x=decor_focused_response[:,el_x,stim_1].flatten(),y=decor_focused_response[:,el_y,stim_1].flatten(),c='red',label='stim '+str(stim_1))
    # axarr[1].scatter(x=decor_focused_response[:,el_x,stim_2].flatten(),y=decor_focused_response[:,el_y,stim_2].flatten(),label='stim '+str(stim_2))
    # axarr[1].scatter(x=np.mean(decor_focused_response[:,el_x,stim_1]).flatten(),y=np.mean(decor_focused_response[:,el_y,stim_1].flatten()),marker='x',s=200, c ='red')
    # axarr[1].scatter(x=np.mean(decor_focused_response[:,el_x,stim_2]).flatten(),y=np.mean(decor_focused_response[:,el_y,stim_2].flatten()),marker='x',s=200)
    # axarr[1].set_title("decorrelated data")
    # plt.xlabel('peak response from electrode ' + str(el_x))
    # plt.ylabel('peak response from electrode ' + str(el_y))
    # plt.legend()
    # plt.show()
    ######


    #transposing focused and decor response
    mod_f_resp = focused_response.transpose((0,2,1))
    dec_mod_f_resp = decor_focused_response.transpose((0,2,1))

    ## Generate index pairs for each stimulus frequency pair and electrode pair
    stim_pairs = []
    electrode_pairs = []
    for (i,j) in combinations(range(num_of_best_stims),2):
        stim_pairs.append((i,j))
    for (i,j) in combinations(range(num_of_electrodes),2):
        electrode_pairs.append((i,j))

    ########################################
    # Initialize result arrays for plotting
    discrim_x_examp = []
    discrim_y_examp = []
    discrim_y_decor_examp = []
    discrim_y_uncor_examp = []
    discrim_y_org_examp = []
    org_y_examp = []
    decor_y_examp = []
    count = 0
    pair_scores_dict = {}
    scores = []
    test_dict = {}
    org_dict = {}
    decor_dict = {}
    #electrode_combinations = [(0,1),(5,10),(3,100),(7,88)]

    # Select analysis method and associated paraeters
    #TODO: PARAMETERIZE THIS!!!
    #meathod = "linear discrim"
    adjust_volume = False
    #meathod = "log regression"

    ## For each stimulus and electrode pair, calculate linear discriminability or perform logistic regression
    t=time.time()
    for (i,j) in stim_pairs:
        if count%10==0:
            # print(count + ', ', end='', flush=True)
            print count
        print "%s : %ss"%(str(count),str(round(time.time()-t)))
        count += 1
        t=time.time()
        stim_1 = i
        stim_2 = j

        org_score = []
        decor_score = []
        for el_comb in electrode_pairs:
        #for el_comb in electrode_combinations:
            if method == "ld": #"linear discrim":
                org_score_val, decor_score_val = pair_el_discrim_value(focused_response,stim_1,stim_2,[el_comb[0],el_comb[1]],adjust_volume=adjust_volume)
                org_score.append(org_score_val)
                decor_score.append(decor_score_val)
            
            if method == "lr": #"logistic regression":
                model = sklearn.linear_model.LogisticRegression()
                model = model.fit(mod_f_resp[:,[stim_1,stim_2],[el_comb[0],el_comb[1]]].reshape(num_of_samples*2,-1),np.array([-10]*num_of_samples+[10]*num_of_samples))
                org_score.append(model.score(mod_f_resp[:,[stim_1,stim_2],[el_comb[0],el_comb[1]]].reshape(num_of_samples*2,-1),np.array([-10]*num_of_samples+[10]*num_of_samples)))

                model = sklearn.linear_model.LogisticRegression()
                model = model.fit(dec_mod_f_resp[:,[stim_1,stim_2],[el_comb[0],el_comb[1]]].reshape(num_of_samples*2,-1),np.array([-10]*num_of_samples+[10]*num_of_samples))
                decor_score.append(model.score(dec_mod_f_resp[:,[stim_1,stim_2],[el_comb[0],el_comb[1]]].reshape(num_of_samples*2,-1),np.array([-10]*num_of_samples+[10]*num_of_samples)))


        discrim_x_examp.append(float(j-i))
        discrim_y_examp.append(np.mean(np.array(org_score)-np.mean(np.array(decor_score))))
        org_y_examp.append(np.mean(np.array(org_score)))
        decor_y_examp.append(np.mean(np.array(decor_score)))
        if not pair_scores_dict.has_key(j-i):
            pair_scores_dict[j-i] = []
        pair_scores_dict[j-i].append([np.mean(np.array(org_score)),np.mean(np.array(decor_score)),np.array(org_score),np.array(decor_score)])
        #
        scores.append([org_score,decor_score])
        if not test_dict.has_key(j-i):
            test_dict[j-i] = []
        test_dict[j-i].append(np.mean(np.array(org_score)-np.array(decor_score)))
        #Save org and decor values as a function of tuning difference
        if not org_dict.has_key(j-i):
            org_dict[j-i] = []
        org_dict[j-i].append(np.mean(np.array(org_score)))
        if not decor_dict.has_key(j-i):
            decor_dict[j-i] = []
        decor_dict[j-i].append(np.mean(np.array(decor_score)))



        discrim_y_decor_examp.append(np.mean(np.array(decor_score)))
        discrim_y_uncor_examp.append(np.mean(np.array(org_score)))
    plt.scatter(x = np.array(discrim_x_examp), y = np.array(discrim_y_examp))
    plt.scatter(x = np.array(discrim_x_examp), y = np.array(org_y_examp),color='g')
    plt.scatter(x = np.array(discrim_x_examp), y = np.array(decor_y_examp),color='r')
    #plt.plot([i for i in pair_scores_dict],[np.mean(np.array(pair_scores_dict[i][0]) - np.array(pair_scores_dict[i][1])) for i in pair_scores_dict])
    #print [i for i in pair_scores_dict]
    #print len(pair_scores_dict[1])
    #print [np.mean(np.array(pair_scores_dict[i][0])) for i in pair_scores_dict]
    #plt.plot([i for i in pair_scores_dict],[np.mean(np.array(pair_scores_dict[i][0])) for i in pair_scores_dict])
    plt.plot([i for i in test_dict],[np.mean(np.array(test_dict[i])) for i in test_dict])
    plt.plot([i for i in org_dict],[np.mean(np.array(org_dict[i])) for i in org_dict],'g')
    plt.plot([i for i in decor_dict],[np.mean(np.array(decor_dict[i])) for i in decor_dict],'r')
    plt.axhline(0)
    plt.show()
    plt.savefig()
    print count
    scores = np.array(scores)


    ## Keep results for this ratid and frequency band
    #TODO: Save these results
    #scores has structure scores[stim_pair][orig,decor][electrode_pair]
    if results == [] or np.any(results[-1][0] != decor_focused_response):
        results.append([decor_focused_response, scores, method])

    return (decor_focused_response, scores, method, discrim_x_examp, discrim_y_examp, test_dict)



