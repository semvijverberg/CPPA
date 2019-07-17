#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load main_era5.py
"""
Created on Mon Dec 10 10:31:42 2018

@author: semvijverberg
"""

import os, sys


if os.path.isdir("/Users/semvijverberg/surfdrive/"):
    basepath = "/Users/semvijverberg/surfdrive/"
    data_base_path = basepath
else:
    basepath = "/home/semvij/"
#    data_base_path = "/p/tmp/semvij/ECE"
    data_base_path = "/p/projects/gotham/semvij"
os.chdir(os.path.join(basepath, 'Scripts/CPPA/CPPA'))
script_dir = os.getcwd()
if script_dir not in sys.path: sys.path.append(script_dir)
if sys.version[:1] == '3':
    from importlib import reload as rel

import numpy as np
import xarray as xr 
import pandas as pd
import matplotlib.pyplot as plt
import func_CPPA
import func_pred
import load_data
import ROC_score
from ROC_score import plotting_timeseries

xarray_plot = func_CPPA.xarray_plot
xrplot = func_CPPA.xarray_plot

# experiments
#import init.EC_t2m_E_US_grouped_hot_days as settings
#import init.era5_t2mmax_W_US_sst as settings

#import init.era5_t2mmax_E_US_sst as settings
#import init.ERAint_t2mmax_E_US_sst as settings
import init.EC_t2m_E_US as settings



# bram equal era 5 mask
#import init.bram_e5mask_era5_t2mmax_E_US_sst as settings
#import init.bram_e5mask_ERAint_t2mmax_E_US_sst as settings
#import init.bram_e5mask_EC_t2m_E_US_sst as settings

ex = settings.__init__()
#ex['RV_aggregation'] = 'RVfullts_mean'

# =============================================================================
# load data (write your own function load_data(ex) )
# =============================================================================
RVtsfull, RV_ts, ex = load_data.load_response_variable(ex)
Prec_reg, ex = load_data.load_precursor(ex)

print_ex = ['RV_name', 'name', 'max_break',
            'min_dur', 'event_percentile',
            'event_thres', 'extra_wght_dur',
            'grid_res', 'startyear', 'endyear', 
            'startperiod', 'endperiod', 'leave_n_out',
            'n_oneyr', 'wghts_accross_lags', 'add_lsm',
            'tfreq', 'lags', 'n_yrs', 'region',
            'rollingmean', 'seed',
            'SCM_percentile_thres', 'FCP_thres', 'perc_yrs_out', 'days_before',
             'distance_eps_init',
            'ROC_leave_n_out', 'method', 'n_boot',
            'RVts_filename', 'path_pp', 'RV_aggregation']


def printset(print_ex=print_ex, ex=ex):
    max_key_len = max([len(i) for i in print_ex])
    
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline)


printset()
n = 0
ex['n'] = n ; lag=0

 


# In[ ]:


# =============================================================================
# Run code with ex settings
# =============================================================================
#ex['lags'] = [0]; ex['method'] = 'no_train_test_split' ; 
l_ds_CPPA, ex = func_CPPA.main(RV_ts, Prec_reg, ex)


# save ex setting in text file
output_dic_folder = ex['output_dic_folder']
#if os.path.isdir(output_dic_folder):
#    answer = input('Overwrite?\n{}\ntype y or n:\n\n'.format(output_dic_folder))
#    if 'n' in answer:
#        assert (os.path.isdir(output_dic_folder) != True)
#    elif 'y' in answer:
#        pass

if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)

# save output in numpy dictionary
filename = 'output_main_dic'
if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)
to_dict = dict( { 'ex'      :   ex,
                 'l_ds_CPPA' : l_ds_CPPA} )
np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)  

# write output in textfile
if 'output_dic_folder' not in print_ex: print_ex.append('output_dic_folder')
txtfile = os.path.join(output_dic_folder, 'experiment_settings.txt')
with open(txtfile, "w") as text_file:
    max_key_len = max([len(i) for i in print_ex])
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline, file=text_file)




subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
total_folder = os.path.join(ex['figpathbase'], subfolder)
if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
# 'group_accros_tests_single_lag' 
# 'group_across_test_and_lags'

if ex['method'] == 'iter': 
    eps = 10
    l_ds_CPPA, ex = func_CPPA.grouping_regions_similar_coords(l_ds_CPPA, ex, 
                     grouping = 'group_accros_tests_single_lag', eps=10)
if ex['method'][:6] == 'random':
 
    eps = 11 ; grouping = 'group_across_test_and_lags'
    if ex['datafolder'] == 'EC': 
        eps = 6 ; grouping = 'group_across_test_and_lags'
    l_ds_CPPA, ex = func_CPPA.grouping_regions_similar_coords(l_ds_CPPA, ex, 
                     grouping = grouping, eps=eps)

 
func_CPPA.plot_precursor_regions(l_ds_CPPA, 10, 'pat_num_CPPA', [0, 5], None, ex)
print('\n\n\nCheck labelling\n\n\n')
func_CPPA.plot_precursor_regions(l_ds_CPPA, 10, 'pat_num_CPPA_clust', [0, 5], None, ex)

if ex['store_timeseries'] == True:
    ex['eps_traintest'] = eps
    func_CPPA.store_ts_wrapper(l_ds_CPPA, RV_ts, Prec_reg, ex)
    

    to_dict = dict( { 'ex'      :   ex,
                     'l_ds_CPPA' : l_ds_CPPA} )
    np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)  
    ex = func_pred.spatial_cov(ex, key1='spatcov_CPPA')


#%% 
RVaggr = 'RVfullts95'
EC_folder = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/random10fold_leave_16_out_2000_2159_tf1_95p_1.125deg_60nyr_95tperc_0.85tc_1rmRV_2019-06-14/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
era5T95   = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2018_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_rng50_2019-06-24/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
erai      = f'/Users/semvijverberg/surfdrive/MckinRepl/ERAint_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2017_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_{RVaggr}_rng50_2019-07-04/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'

### same mask Bram
#EC_folder = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/random10fold_leave_16_out_2000_2159_tf1_95p_1.125deg_60nyr_95tperc_0.85tc_1rmRV_2019-06-12_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
era5      = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2018_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_2019-06-09_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
#erai      = '/Users/semvijverberg/surfdrive/MckinRepl/ERAint_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2017_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_2019-06-09_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'



if ex['datafolder'] == 'ERAint': output_dic_folder = erai
if ex['datafolder'] == 'era5': output_dic_folder = era5T95
if ex['datafolder'] == 'EC': output_dic_folder = EC_folder
#    
    
# =============================================================================
# Load and Generate output in console
# =============================================================================
#output_dic_folder = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/random10fold_leave_16_out_2000_2159_tf1_95p_1.125deg_60nyr_95tperc_0.85tc_1rmRV_2019-05-23/lags[0,10,20,30]Ev1d0p'

from scoringclass import SCORE_CLASS
filename = 'output_main_dic'
dic = np.load(os.path.join(output_dic_folder, filename+'.npy'),  encoding='latin1').item()
# load settings
ex = dic['ex']
# load patterns
try:
    l_ds_CPPA = dic['l_ds_CPPA']
except:
    l_ds_CPPA = dic['l_ds_PEP']
if 'score' in ex.keys():
    SCORE = ex['score']

ex['store_timeseries'] = False
#%%
#ex['grouped'] = True
#ex['lags'] = [0, 20, 40]
#ex['store_timeseries']  = False
#ex['event_percentile']  = 65
#ex['min_dur']           = 4
#ex['max_break']         = 0
#if ex['event_percentile'] == 'std':
#    # binary time serie when T95 exceeds 1 std
#    ex['event_thres'] = RV_ts.mean(dim='time').values + RV_ts.std().values
#else:
#    percentile = ex['event_percentile']
#    ex['event_thres'] = np.percentile(RV_ts.values, percentile)
#tim = func_CPPA.Ev_timeseries(RV_ts, ex['event_thres'], ex, grouped=ex['grouped'])[0].time
#tim.size / RV_ts.size


# write output in textfile
if 'use_ts_logit' in ex.keys() and 'pval_logit_final' in ex.keys():
    predict_folder = '{}{}_ts{}'.format(ex['pval_logit_final'], ex['logit_valid'], ex['use_ts_logit'])
else:
    predict_folder = ''
ex['exp_folder'] = os.path.join(ex['CPPA_folder'], predict_folder)
main_output = os.path.join(ex['figpathbase'], ex['exp_folder'])
if os.path.isdir(main_output) != True : os.makedirs(main_output)

txtfile = os.path.join(main_output, 'experiment_settings.txt')
with open(txtfile, "w") as text_file:
    max_key_len = max([len(i) for i in print_ex])
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline)
        print(printline, file=text_file)

# =============================================================================
# perform prediciton        
# =============================================================================

# write output in textfile
if 'use_ts_logit' in ex.keys():
    if ex['use_ts_logit'] == True:
        predict_folder = '{}{}_ts{}'.format(ex['pval_logit_final'], ex['logit_valid'], ex['use_ts_logit'])
else:
    ex['use_ts_logit'] = False
    predict_folder = ''
ex['exp_folder'] = os.path.join(ex['CPPA_folder'], predict_folder)
if ex['store_timeseries'] == True:
    
    if ex['method'] == 'iter' or ex['method'][:6] == 'random': 
        l_ds_CPPA, ex = func_CPPA.grouping_regions_similar_coords(l_ds_CPPA, ex, 
                         grouping = 'group_accros_tests_single_lag', eps=10)
        key_pattern_num = 'pat_num_CPPA_clust'
    else:
        key_pattern_num = 'pat_num_CPPA'
    func_CPPA.store_ts_wrapper(l_ds_CPPA, RV_ts, Prec_reg, ex)
    ex = func_pred.spatial_cov(ex, key1='spatcov_CPPA')
    ex = ROC_score.func_AUC_wrapper(ex)
else:
    key_pattern_num = 'pat_num_CPPA'
    ex, SCORE = ROC_score.only_spatcov_wrapper(l_ds_CPPA, RV_ts, Prec_reg, ex)
    ex['score'] = SCORE
    filename_2 = 'output_main_dic_with_score'
    to_dict = dict( { 'ex'      :   ex,
                     'l_ds_CPPA' : l_ds_CPPA} )
    np.save(os.path.join(output_dic_folder, filename_2+'.npy'), to_dict) 
if ex['use_ts_logit'] == False: ex.pop('use_ts_logit')

ROC_score.create_validation_plot([output_dic_folder], metric='AUC', getPEP=False)
ROC_score.create_validation_plot([output_dic_folder], metric='brier', getPEP=False)

#%% New scoring func
path_ts = ex['output_ts_folder']
lag_to_load = 5
lags_to_test = [0, 1, 2]
frequencies = [1, 5, 10, 15, 20, 25, 29, 30, 31, 33, 35, 37, 39, 40, 41, 42, 44, 45]
#frequencies = [60, 80]
t = 30
BSS_f = [] 
AUC_f = []
for t in frequencies:
    print('tfreq:', t)
    ex['tfreq'] = t
    ex['lags'] = [l*t for l in lags_to_test]
    SCORE, ex = ROC_score.spatial_cov(RV_ts, ex, path_ts, lag_to_load=lag_to_load, keys=['spatcov_CPPA'])
    BSS = SCORE.brier_logit.loc['BSS'].values
    AUC = SCORE.AUC_spatcov.loc['AUC'].values
    BSS_f.append(BSS)
    AUC_f.append(AUC)
    print(BSS)

#%%
import seaborn as sns

def plot_score_freq(x, y, metric, lags_to_test):
    df = pd.DataFrame(np.swapaxes(BSS_f, 1,0), 
                      index=None, columns = frequencies)
    df['lag'] = pd.Series(lags_to_test, index=df.index)
    
    g = sns.FacetGrid(df, col='lag', size=3, aspect=1.4,sharey=True, col_wrap=3)
    for i, ax in enumerate(g.axes.flatten()):
        l = df['lag'][i]
        ax.scatter(x, np.array(y)[:,l].squeeze())
        ax.set_ylabel(metric) ; ax.set_xlabel('Time Aggregation')
        ax.grid(b=True, which='major')
        ax.set_title('lag {}'.format(i))
        xticks = np.arange(0, max(frequencies), 5) ; xticks[0] = 1
        ax.set_xticks(xticks)
    str_freq = str(x).replace(' ' ,'')  
    lags_str = str(lags_to_test).replace(' ' ,'')        
    f_name = '{}_lags_tfreqs_{}.png'.format(metric, str_freq, lags_str)
    filename = os.path.join(output_dic_folder, f_name)
    plt.savefig(filename, dpi=600)
plot_score_freq(frequencies, BSS_f, 'BSS', lags_to_test)
plot_score_freq(frequencies, AUC_f, 'AUC', lags_to_test)
#%%
RVaggr = 'RVfullts95'
EC_folder = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/random10fold_leave_16_out_2000_2159_tf1_95p_1.125deg_60nyr_95tperc_0.85tc_1rmRV_2019-06-14/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
era5T95   = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2018_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_rng50_2019-06-24/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
erai      = f'/Users/semvijverberg/surfdrive/MckinRepl/ERAint_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2017_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_{RVaggr}_rng50_2019-07-04/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'



## same mask Bram
#EC_folder = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/random10fold_leave_16_out_2000_2159_tf1_95p_1.125deg_60nyr_95tperc_0.85tc_1rmRV_2019-06-12_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
#era5      = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2018_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_2019-06-09_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
#erai      = '/Users/semvijverberg/surfdrive/MckinRepl/ERAint_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2017_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_2019-06-09_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'

if ex['datafolder'] == 'ERAint': output_dic_folder = erai
if ex['datafolder'] == 'era5': output_dic_folder = era5T95
if ex['datafolder'] == 'EC': output_dic_folder = EC_folder

try:
    filename_2 = 'output_main_dic_with_score'
    dic = np.load(os.path.join(output_dic_folder, filename_2+'.npy'),  encoding='latin1').item()
    # load settings
    ex = dic['ex']
    # load patterns
    l_ds_CPPA = dic['l_ds_CPPA']
except:
    pass
if 'score' in ex.keys():
    SCORE = ex['score']

#%% Reliability curve
from sklearn.calibration import calibration_curve
lag = 90

strategy = 'quantile' # 'quantile' or 'uniform'
fraction_of_positives, mean_predicted_value = calibration_curve(SCORE.y_true_test[0], SCORE.predmodel_2[lag], 
                                                                n_bins=5, strategy=strategy)
plt.plot(mean_predicted_value, fraction_of_positives) ; plt.plot(np.arange(0,1+1E-9,0.1),np.arange(0,1+1E-9,0.1))
plt.title(f'Lead time = {lag}')
plt.ylabel('fraction of positives')
plt.xlabel('mean predicted value')
#%%
RVaggr = 'RVfullts95'
EC_folder = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/random10fold_leave_16_out_2000_2159_tf1_95p_1.125deg_60nyr_95tperc_0.85tc_1rmRV_2019-06-14/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
era5T95   = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2018_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_rng50_2019-06-24/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
erai      = f'/Users/semvijverberg/surfdrive/MckinRepl/ERAint_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2017_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_{RVaggr}_rng50_2019-07-04/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'

## same mask Bram
#EC_folder = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/random10fold_leave_16_out_2000_2159_tf1_95p_1.125deg_60nyr_95tperc_0.85tc_1rmRV_2019-06-12_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
#era5      = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2018_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_2019-06-09_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'
#erai      = '/Users/semvijverberg/surfdrive/MckinRepl/ERAint_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2017_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_2019-06-09_bram/lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p'

outdic_folders = [era5T95, erai, EC_folder] # , erai, EC_folder

try:  
    ROC_score.create_validation_plot(outdic_folders, metric='AUC')
    ROC_score.create_validation_plot(outdic_folders, metric='brier')
except:
    pass



#%%
# =============================================================================
#   Plotting
# =============================================================================
lags_plot = [0, 10, 20, 35, 50, 65]
#lags_plot = [0, 10, 20, 30]
try:
    ROC_str_Sem     = ['{} days - AUC score {:.2f}'.format(l, np.round(float(SCORE.AUC_spatcov.loc['AUC'][l].values), 2) ) for l in lags_plot ]
except:
    ROC_str_Sem     = ['{} days'.format(ex['lags'][i]) for i in range(len(ex['lags'])) ]

lats = Prec_reg.latitude
lons = Prec_reg.longitude
array = np.zeros( (len(l_ds_CPPA), len(lags_plot), len(lats), len(lons)) )
patterns_Sem = xr.DataArray(data=array, coords=[range(len(l_ds_CPPA)), lags_plot, lats, lons], 
                      dims=['n_tests', 'lag','latitude','longitude'], 
                      name='{}_tests_patterns_Sem'.format(len(l_ds_CPPA)), attrs={'units':'Kelvin'})


for n in range(len(ex['train_test_list'])):
    name_for_ts = 'CPPA'
        
    if (ex['method'][:6] == 'random'):
        if n == len(l_ds_CPPA):
            # remove empty n_tests
            patterns_Sem = patterns_Sem.sel(n_tests=slice(0,len(l_ds_CPPA)))

    
    upd_pattern = l_ds_CPPA[n]['pattern_' + name_for_ts].sel(lag=lags_plot)
    patterns_Sem[n,:,:,:] = upd_pattern * l_ds_CPPA[n]['std_train_min_lag']


    
kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                    'vmin' : -0.4, 'vmax' : 0.4, 'subtitles' : ROC_str_Sem,
                   'cmap' : plt.cm.RdBu_r, 'column' : 2} )

mean_n_patterns = patterns_Sem.mean(dim='n_tests')
mean_n_patterns = mean_n_patterns.where(l_ds_CPPA[n]['pat_num_CPPA']>0.5)
mean_n_patterns.attrs['units'] = '[K]'
mean_n_patterns.attrs['title'] = 'CPPA - Precursor Pattern'
try:
    mean_n_patterns.name = 'mean_{}_traintest'.format(ex['n_conv'])
except:
    mean_n_patterns.name = '' 
filename = os.path.join('', 'mean_over_{}_tests_lags{}'.format(ex['n_conv'],
                        str(lags_plot).replace(' ' ,'')) )
func_CPPA.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)




#%% Robustness accross training sets
#    ex['lags'] = [5,15,30,50]

lats = patterns_Sem.latitude
lons = patterns_Sem.longitude
array = np.zeros( (ex['n_conv'], len(lags_plot), len(lats), len(lons)) )
wgts_tests = xr.DataArray(data=array, 
                coords=[range(ex['n_conv']), lags_plot, lats, lons], 
                dims=['n_tests', 'lag','latitude','longitude'], 
                name='{}_tests_wghts'.format(ex['n_conv']), attrs={'units':'wghts ['})
for n in range(ex['n_conv']):
#    wgts_tests[n,:,:,:] = l_ds_CPPA[n]['weights'].sel(lag=ex['lags'])
    wgts_tests[n,:,:,:] = l_ds_CPPA[n]['weights'].sel(lag=lags_plot).where(l_ds_CPPA[n]['pat_num_CPPA']>0.5)
    
from matplotlib.colors import LinearSegmentedColormap 
if ex['leave_n_out']:
    n_lags = len(lags_plot)
    n_lats = patterns_Sem.sel(n_tests=0).latitude.size
    n_lons = patterns_Sem.sel(n_tests=0).longitude.size
    
    pers_patt = patterns_Sem.sel(n_tests=0).sel(lag=lags_plot).copy()
#    arrpatt = np.nan_to_num(patterns_Sem.values)
#    mask_patt = (arrpatt != 0)
#    arrpatt[mask_patt] = 1
    wghts = np.zeros( (n_lags, n_lats, n_lons) )
#    plt.imshow(arrpatt[0,0]) ; plt.colorbar()
    for l in lags_plot:
        i = lags_plot.index(l)
        wghts[i] = np.nansum(wgts_tests[:,i,:,:].values, axis=0)
    pers_patt.values = wghts 
    pers_patt = pers_patt.where(pers_patt.values != 0)
    pers_patt -= 1E-9
    size_trainset = ex['n_yrs'] - ex['leave_n_years_out']
    pers_patt.attrs['units'] = 'No. of times in final pattern [0 ... {}]'.format(ex['n_conv'])
    pers_patt.attrs['title'] = ('Robustness SST pattern\n{} different '
                            'training sets (n={} yrs)'.format(ex['n_conv'],size_trainset))
    filename = os.path.join('', 'Robustness_across_{}_training_tests_lags{}'.format(ex['n_conv'],
                            str(lags_plot).replace(' ' ,'')) )
    vmax = ex['n_conv'] 
    extend = ['min','yellow']
    if vmax-20 <= ex['n_conv']: extend = ['min','white']
    
    mean = np.round(pers_patt.mean(dim=('latitude', 'longitude')).values, 1)
    
    std =  np.round(pers_patt.std(dim=('latitude', 'longitude')).values, 0)
    ax_text = ['mean = {}±{}'.format(mean[l],int(std[l])) for l in range(len(lags_plot))]
    colors = plt.cm.magma_r(np.linspace(0,0.7, 20))
    colors[-1] = plt.cm.magma_r(np.linspace(0.99,1, 1))
    cm = LinearSegmentedColormap.from_list('test', colors, N=255)
    kwrgs = dict( {'title' : pers_patt.attrs['title'], 'clevels' : 'notdefault', 
                   'steps' : 11, 'subtitles': ROC_str_Sem, 
                   'vmin' : max(0,vmax-20), 'vmax' : vmax, 'clim' : (max(0,vmax-20), vmax),
                   'cmap' : cm, 'column' : 2, 'extend':extend,
                   'cbar_vert' : 0.07, 'cbar_hght' : 0.01,
                   'adj_fig_h' : 1.25, 'adj_fig_w' : 1., 
                   'hspace' : -0.02, 'wspace' : 0.04, 
                   'title_h': 0.95} )
    func_CPPA.plotting_wrapper(pers_patt, ex, filename, kwrgs=kwrgs)



#%% Weighing features if there are extracted every run (training set)
# weighted by persistence of pattern over
if ex['leave_n_out']:
    kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                    'vmin' : -0.4, 'vmax' : 0.4, 'subtitles' : ROC_str_Sem,
                   'cmap' : plt.cm.RdBu_r, 'column' : 1,
                   'cbar_vert' : 0.04, 'cbar_hght' : -0.025,
                   'adj_fig_h' : 0.9, 'adj_fig_w' : 1., 
                   'hspace' : 0.2, 'wspace' : 0.08,
                   'title_h' : 0.95} )
    # weighted by persistence (all years == wgt of 1, less is below 1)
    final_pattern = patterns_Sem.mean(dim='n_tests') * wghts/np.max(wghts)
    final_pattern = mean_n_patterns.where(wghts!=0.)
    final_pattern['lag'] = ROC_str_Sem

    title = 'Precursor Pattern'
    if final_pattern.sum().values != 0.:
        final_pattern.attrs['units'] = 'Kelvin'
        final_pattern.attrs['title'] = title
                             
        final_pattern.name = ''
        filename = os.path.join('', ('{}_Precursor_pattern_robust_w_'
                             '{}_tests_lags{}'.format(ex['datafolder'], ex['n_conv'],
                              str(lags_plot).replace(' ' ,'')) ))

        func_CPPA.plotting_wrapper(final_pattern, ex, filename, kwrgs=kwrgs)

#%% Cross corr matrix
final_pattern.coords['lag'] = lags_plot


#%% Robustness of training precursor regions

subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
total_folder = os.path.join(ex['figpathbase'], subfolder)
if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
years = range(ex['startyear'], ex['endyear'])

#n_land = np.sum(np.array(np.isnan(Prec_reg.values[0]),dtype=int) )
#n_sea = Prec_reg[0].size - n_land
if ex['method'] == 'iter':
    sorted_idx = np.argsort(ex['n_events'])
    sorted_n_events = ex['n_events'].copy(); sorted_n_events.sort()
    test_set_to_plot = [ex['tested_yrs'][n][0] for n in sorted_idx[:6]]
    [test_set_to_plot.append(ex['tested_yrs'][n][0]) for n in sorted_idx[-5:]]
elif ex['method'][:6] == 'random':
    test_set_to_plot = [set(t[1]['RV'].time.dt.year.values) for t in ex['train_test_list'][::5]]
elif ex['method'][:] == 'no_train_test_split':
    test_set_to_plot = ['None']
#test_set_to_plot = list(np.arange(0,ex['n_conv'],5))
for yr in test_set_to_plot: 
    n = test_set_to_plot.index(yr)
    Robustness_weights = l_ds_CPPA[n]['weights'].sel(lag=lags_plot)
    size_trainset = ex['n_yrs'] - ex['leave_n_years_out']
    Robustness_weights.attrs['title'] = ('Robustness\n test yr(s): {}, single '
                            'training set (n={} yrs)'.format(yr,size_trainset))
    Robustness_weights.attrs['units'] = 'Weights [{} ... 1]'.format(ex['FCP_thres'])
    filename = os.path.join('', Robustness_weights.attrs['title'].replace(
                            ' ','_')+'.png')
    for_plt = Robustness_weights.where(Robustness_weights.values != 0).copy()
#    n_pattern = Prec_reg[0].size - np.sum(np.array(np.isnan(for_plt[0]),dtype=int))
    
    if ex['n_conv'] == 1:
        steps = 19
    else:
        steps = 11
    kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                   'steps' : 11, 'subtitles': ROC_str_Sem, 
                   'vmin' : ex['FCP_thres'], 'vmax' : for_plt.max().values+1E-9, 
                   'cmap' : plt.cm.viridis_r, 'column' : 2,
                   'cbar_vert' : 0.05, 'cbar_hght' : 0.01,
                   'adj_fig_h' : 1.25, 'adj_fig_w' : 1., 
                   'hspace' : 0.02, 'wspace' : 0.08} )
    
    func_CPPA.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
    
#%%

filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
dicRV = np.load(filename,  encoding='latin1').item()
folder = os.path.join(ex['figpathbase'], ex['exp_folder'])
if 'mask' in ex.keys():
    xarray_plot(ex['mask'], path=folder, name='RV_mask', saving=True)
    
func_CPPA.plot_oneyr_events(RV_ts, ex, 2012, ex['output_dic_folder'], saving=True)
## plotting same figure as in paper
#for i in range(2005, 2010):
#    func_CPPA.plot_oneyr_events(RV_ts, ex, i, folder, saving=True)

#%% Plotting prediciton time series vs truth:

#yrs_to_plot = [1983, 1988, 1994, 2002, 2007, 2012, 2015]
ex['n_events'] = []
all_years = np.unique(SCORE.y_true_test.index.year)
for y in all_years:
    n_ev = int(SCORE.y_true_test[0][SCORE.y_true_test[0].index.year==y].sum())
    ex['n_events'].append(n_ev)
if 'n_events' in ex.keys():
    sorted_idx = np.argsort(ex['n_events'])
    sorted_n_events = ex['n_events'].copy(); sorted_n_events.sort()
    yrs_to_plot = [all_years[n] for n in sorted_idx[:6]]
    [yrs_to_plot.append(all_years[n]) for n in sorted_idx[-5:]]

#    test = ex['train_test_list'][0][1]        
plotting_timeseries(SCORE, 'spatcov', yrs_to_plot, ex) 
plotting_timeseries(SCORE, 'logit', yrs_to_plot, ex) 

#%%    
from sklearn.metrics import brier_score_loss
n_steps = 2400
rand_scores = [] 
for p in np.linspace(0, 1, 19):
    n_ev = int(p * n_steps) ; 
    rand_true = np.zeros( (n_steps) ) ; 
    ind = np.random.choice(range(n_steps), n_ev)
    rand_true[ind] = 1
    rand_scores.append(brier_score_loss(rand_true, np.repeat(p, n_steps)))
plt.plot(np.linspace(0, 1, 19), rand_scores)

#%% plot frequency
import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
datesRV = pd.to_datetime(SCORE.y_true_test[0].index)
freq = pd.DataFrame(data= np.zeros(len(ex['all_yrs'])), index = ex['all_yrs'], columns=['freq'])
for i, yr in enumerate(ex['all_yrs']):
    oneyr = SCORE.y_true_test[0].loc[func_CPPA.get_oneyr(datesRV, yr)]
    freq.loc[yr] = oneyr.sum()
plt.figure( figsize=(8,6) )
plt.bar(freq.index, freq['freq'])
plt.ylabel('freq. hot days', fontdict={'fontsize':14})

fname = 'freq_per_year.png'
filename = os.path.join(output_dic_folder, fname)
plt.savefig(filename, dpi=600) 


#%% Initial regions from only composite extraction:
key_pattern_num = 'pat_num_CPPA_clust'
lags_plot = [0, 5, 10, 15, 20, 25]
lags = lags_plot
if ex['leave_n_out']:
    subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
    total_folder = os.path.join(ex['figpathbase'], subfolder)
    if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
    if 'ROC_str_Sem' in globals():
        subtitles = [ROC_str_Sem[lags_plot.index(l)] for l in lags]
    else:
        subtitles = ['{} days'.format(lags_plot[i]) for i in range(len(lags_plot)) ]
    
        
    func_CPPA.plot_precursor_regions(l_ds_CPPA, ex['n_conv'], key_pattern_num, lags, subtitles, ex)
#%%
lags_plot = [30, 35, 40, 50, 60]    
lags = lags_plot
if ex['leave_n_out']:
    subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
    total_folder = os.path.join(ex['figpathbase'], subfolder)
    if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
    if 'ROC_str_Sem' in globals():
        subtitles = [ROC_str_Sem[lags_plot.index(l)] for l in lags]
    else:
        subtitles = ['{} days'.format(lags_plot[i]) for i in range(len(lags_plot)) ]
    
        
    func_CPPA.plot_precursor_regions(l_ds_CPPA, ex['n_conv'], key_pattern_num, lags, subtitles, ex)

#%% cross correlation plots
df = ROC_score.get_ts_matrix(Prec_reg, final_pattern, ex, lag=0)
ROC_score.build_matrix_wrapper(df, ex, lag=[0], wins=[1, 20, 30, 60, 80, 100, 365])


#%% get autocorrlation plots
import datetime    
now = datetime.datetime.now()
fname = now.strftime('%Y-%m-%d_%Hhr')
from load_data import load_1d
west_ts = "era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus2.npy"
east_ts = "era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_new.npy"

tsname = 'RVfullts_mean' # ['RVfullts95', 'RVfullts_mean']
#%%
# import t2mmax
filename_t2m = 't2mmax_US_1979-2018_1jan_31dec_daily_0.25deg.nc'
prec_filename = os.path.join(ex['path_pp'], filename_t2m)
t2m = func_CPPA.import_ds_lazy(prec_filename, ex, loadleap=True)
filename = '/Users/semvijverberg/surfdrive/Scripts/rasterio/mask_North_America_0.25deg.nc'
mask = func_CPPA.import_array(filename, ex)
#%%
if tsname == 'RVfullts95':
    CONUS = t2m.where(mask.values==1).quantile(0.95, dim=('latitude', 'longitude'))
elif tsname == 'RVfullts_mean':
    CONUS = t2m.where(mask.values==1).mean(dim=('latitude', 'longitude'), skipna=True)
#%% autocorr summers

def plot_ac_w_e(ax, x, ac_w, ac_e, ac_US, xlabel):
    ax.plot(x, ac_w[0][:x.size], color='b', label='W-U.S. cluster temperature',
            alpha=0.5, linestyle='dashed')
#    ax.fill_between(x, ac_w[1][:,0][:x.size], ac_w[1][:,1][:x.size], color='b',
#                    alpha=0.2)
    ax.plot(x, ac_e[0][:x.size], label='E-U.S. cluster temperature', color='r')
    ax.fill_between(x, ac_e[1][:,0][:x.size], ac_e[1][:,1][:x.size], color='r',
                    alpha=0.3)
    ax.plot(x, ac_US[0][:x.size], label='CON-U.S. temperature', color='g',
            alpha=0.5, linestyle='dashed')
#    ax.fill_between(x, ac_US[1][:,0][:x.size], ac_US[1][:,1][:x.size], color='g',
#                    alpha=0.2)
    ax.set_ylim((-0.2,1.0))
    ax.tick_params(labelsize=15)
    ax.set_xticks(np.linspace(0,round(np.max(x)), 11, dtype=int))
    ax.grid(which='both', axis='both') 
    ax.set_xlabel(xlabel, fontsize=15)
    ax.legend(fontsize=12)

season = 'JJA'

W_ts = load_1d(os.path.join(ex['RV1d_ts_path'], west_ts), ex, tsname)[0]
E_ts = load_1d(os.path.join(ex['RV1d_ts_path'], east_ts), ex, tsname)[0]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,10))
#ac_w = ROC_score.autocorrelation(W_ts.sel(time=W_ts['time.season'] == season))[:100]
#ac_e = ROC_score.autocorrelation(E_ts.sel(time=E_ts['time.season'] == season))[:100]
fig.text(0.5, 0.98, f'autocorrelation {season}', fontsize=15,
               fontweight='heavy', transform=fig.transFigure,
               horizontalalignment='center',verticalalignment='top')
for i, ax in enumerate(axes):
    if i == 0:
        n = 92
        ac_w = ROC_score.autocorr_sm(W_ts.sel(time=W_ts['time.season'] == season), max_lag=n)
        ac_e = ROC_score.autocorr_sm(E_ts.sel(time=E_ts['time.season'] == season), max_lag=n)
        ac_US = ROC_score.autocorr_sm(CONUS.sel(time=CONUS['time.season'] == season), max_lag=n)
        x = np.arange(0, n)/92 ; xlabel = 'JJA periods [years]'
    elif i == 1:
        n = 20*92
        ac_w = ROC_score.autocorr_sm(W_ts.sel(time=W_ts['time.season'] == season), max_lag=n)
        ac_e = ROC_score.autocorr_sm(E_ts.sel(time=E_ts['time.season'] == season), max_lag=n)
        ac_US = ROC_score.autocorr_sm(CONUS.sel(time=CONUS['time.season'] == season), max_lag=n)
        x = np.arange(0, ac_e[0].size)/92 ;  xlabel = 'JJA periods [years]'
    plot_ac_w_e(ax, x, ac_w, ac_e, ac_US, xlabel=xlabel)
plt.savefig(os.path.join('/Users/semvijverberg/surfdrive/McKinRepl/', 
                         'autocorr', fname + f'_summers_{tsname}.png'), dpi=600)

#%% autocorr fullyear
    
now = datetime.datetime.now()
fname = now.strftime('%Y-%m-%d_%Hhr')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
fig.text(0.5, 0.98, f'autocorrelation', fontsize=15,
               fontweight='heavy', transform=fig.transFigure,
               horizontalalignment='center',verticalalignment='top')
for i, ax in enumerate(axes):
    if i == 0:
        ac_w  = ROC_score.autocorr_sm(W_ts, max_lag=n)
        ac_e  = ROC_score.autocorr_sm(E_ts, max_lag=n)
        ac_US = ROC_score.autocorr_sm(CONUS, max_lag=n)
        x = np.arange(0, 101) ; xlabel = 'days'
    elif i == 1:
        n = 20*365
        ac_w  = ROC_score.autocorr_sm(W_ts, max_lag=n)
        ac_e  = ROC_score.autocorr_sm(E_ts, max_lag=n)
        ac_US = ROC_score.autocorr_sm(CONUS, max_lag=n)
        x = np.arange(0, ac_e[0].size)/365 ;  xlabel = 'years'
    plot_ac_w_e(ax, x, ac_w, ac_e, ac_US, xlabel=xlabel)
plt.savefig(os.path.join('/Users/semvijverberg/surfdrive/McKinRepl/', 
                         'autocorr', fname + f'_fullyear_{tsname}.png'), dpi=600)


#%% autocorr subset

dates_to_sel = func_CPPA.get_oneyr(pd.to_datetime(E_ts.time.values), 1988, 2012, 2011, 1983)
dates_to_sel = dates_to_sel[np.logical_or(dates_to_sel.month==6, dates_to_sel.month==7, dates_to_sel.month==8)]
now = datetime.datetime.now()
fname = now.strftime('%Y-%m-%d_%Hhr')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
fig.text(0.5, 0.98, f'autocorrelation', fontsize=15,
               fontweight='heavy', transform=fig.transFigure,
               horizontalalignment='center',verticalalignment='top')
for i, ax in enumerate(axes):
    if i == 0:
        ac_w  = ROC_score.autocorrelation(W_ts.sel(time=dates_to_sel))[:101]
        ac_e  = ROC_score.autocorrelation(E_ts.sel(time=dates_to_sel))[:101]
        ac_US = ROC_score.autocorrelation(CONUS.sel(time=dates_to_sel))[:101]
        x = range(0, 101) ; xlabel = 'days'
    elif i == 1:
        n = 3000
        ac_w  = ROC_score.autocorrelation(W_ts.sel(time=dates_to_sel))[:n]
        ac_e  = ROC_score.autocorrelation(E_ts.sel(time=dates_to_sel))[:n]
        ac_US = ROC_score.autocorrelation(CONUS.sel(time=dates_to_sel))[:n]
        x = np.arange(0, ac_e.size)/92 ;  xlabel = 'years'
    plot_ac_w_e(ax, x, ac_w, ac_e, ac_US, xlabel=xlabel)
plt.savefig(os.path.join('/Users/semvijverberg/surfdrive/McKinRepl/', 
                         'autocorr', fname + '_subsetyears.png'), dpi=600)

#%%

#now = datetime.datetime.now()
#from load_data import load_1d
#east_ts = "era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4.npy"
#E_ts = load_1d(os.path.join(ex['RV1d_ts_path'], east_ts), ex)[0]
#W_ts = Prec_reg.mean(dim=('latitude', 'longitude'), skipna=True)
#
#def plot_ac_w_e(ax, x, ac_w, ac_e, xlabel=None):
#    ax.plot(x, ac_w, label='W-U.S. cluster temperature')
##    ax.plot(x, ac_e, label='E-U.S. cluster temperature')
#    ax.tick_params(labelsize=15)
#    ax.set_xticks(np.linspace(0,round(np.max(x)), 11, dtype=int))
#    ax.grid(which='both', axis='both') 
#    if xlabel == None:
#        ax.set_xlabel(xlabel, fontsize=15)
#    ax.legend(fontsize=12)
#
#fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
#ac_w = ROC_score.autocorrelation(W_ts)
#ac_e = ROC_score.autocorrelation(E_ts)
#fig.text(0.5, 0.98, f'autocorrelation', fontsize=15,
#               fontweight='heavy', transform=fig.transFigure,
#               horizontalalignment='center',verticalalignment='top')
#for i, ax in enumerate(axes):
#    if i == 0:
#        ac_w = ROC_score.autocorrelation(W_ts)[:101]
#        ac_e = ROC_score.autocorrelation(E_ts)[:101]
#        x = range(0, 101) ; xlabel = 'days'
#    elif i == 1:
#        n = W_ts.size
#        ac_w = ROC_score.autocorrelation(W_ts)[:n]
#        ac_e = ROC_score.autocorrelation(E_ts)[:n]
#        x = np.arange(0, n)/365 ;  xlabel = 'years'
#    plot_ac_w_e(ax, x, ac_w, ac_e, xlabel=xlabel)
    
    
    
        

# # End of code

# In[ ]:


# %run func_CPPA.py
# rel(func_CPPA)



# if (ex['method'] == 'no_train_test_split') : ex['n_conv'] = 1
# if ex['method'][:5] == 'split' : ex['n_conv'] = 1
# if ex['method'][:6] == 'random' : ex['n_conv'] = int(ex['n_yrs'] / int(ex['method'][6:]))
# if ex['method'] == 'iter': ex['n_conv'] = ex['n_yrs'] 


# if ex['ROC_leave_n_out'] == True or ex['method'] == 'no_train_test_split': 
#     print('leave_n_out set to False')
#     ex['leave_n_out'] = False
# else:
#     ex['tested_yrs'] = []


# rmwhere, window = ex['rollingmean']
# if rmwhere == 'all' and window != 1:
#     Prec_reg = rolling_mean_time(Prec_reg, ex, center=False)

# train_test_list  = []
# l_ds_CPPA        = []  
# n = 0
# train_all_test_n_out = (ex['ROC_leave_n_out'] == True) & (n==0) 
# ex['n'] = n
# # do single run    
# # =============================================================================
# # Create train test set according to settings 
# # =============================================================================
# train, test, ex = train_test_wrapper(RV_ts, Prec_reg, ex)  

# Prec_train = Prec_reg.isel(time=train['Prec_train_idx'])
# # Prec_train = Prec_train.astype('float64')
# lats = Prec_train.latitude
# lons = Prec_train.longitude

# array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
# pattern_CPPA = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
#                       dims=['lag','latitude','longitude'], name='communities_composite',
#                       attrs={'units':'Kelvin'})


# array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
# pat_num_CPPA = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
#                       dims=['lag','latitude','longitude'], name='commun_numb_init', 
#                       attrs={'units':'Precursor regions'})

# array = np.empty( (len(ex['lags']), len(lats), len(lons)) )
# std_train_min_lag = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
#                       dims=['lag','latitude','longitude'], name='std_train_min_lag', 
#                       attrs={'units':'std [-]'})

# pat_num_CPPA.name = 'commun_numbered'

# weights     = pattern_CPPA.copy()
# weights.name = 'weights'


# RV_event_train = Ev_timeseries(train['RV'], ex['event_thres'], ex)[0]
# RV_event_train = pd.to_datetime(RV_event_train.time.values)

# RV_dates_train = pd.to_datetime(train['RV'].time.values)
# all_yrs_set = list(set(RV_dates_train.year.values))
# comp_years = list(RV_event_train.year.values)
# mask_chunks = get_chunks(all_yrs_set, comp_years, ex)
# lag = 0

# idx = ex['lags'].index(lag)

# events_min_lag = func_dates_min_lag(RV_event_train, lag)[1]
# dates_train_min_lag = func_dates_min_lag(RV_dates_train, lag)[1]
# event_idx = [list(dates_train_min_lag.values).index(E) for E in events_min_lag.values]
# binary_events = np.zeros(RV_dates_train.size)    
# binary_events[event_idx] = 1





# std_train_min_lag[idx] = Prec_train.where(Prec_train.mask==False).sel(time=dates_train_min_lag).std(dim='time', skipna=True)
# #%%
# # extract precursor regions composite approach
# # composite_p1, xrnpmap_p1, wghts_at_lag = extract_regs_p1(Prec_train, mask_chunks, events_min_lag, 
# #                                      dates_train_min_lag, std_train_lag, ex)  
#%%
for yr in range(2000, 2010):
    dates = func_CPPA.get_oneyr(ex['dates_RV'], yr) ; 
    plt.figure() ; plt.plot(RV_ts_era5_Bram.sel(time=dates), label='bram') ; 
    dates = func_CPPA.get_oneyr(RV_ts_era5_Sem.time.values, yr) ; 
    plt.plot(((RV_ts_era5_Sem-RV_ts_era5_Sem.mean())/RV_ts_era5_Sem.std()).sel(time=dates), label='Sem') ; 
    plt.legend()