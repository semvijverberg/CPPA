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
    data_base_path = "/p/projects/gotham/semvij/"
os.chdir(os.path.join(basepath, 'Scripts/CPPA/CPPA'))
script_dir = os.getcwd()
sys.path.append(script_dir)
if sys.version[:1] == '3':
    from importlib import reload as rel

import numpy as np
import xarray as xr 
import pandas as pd
import matplotlib.pyplot as plt
import func_CPPA
import func_pred
import load_data
from ROC_score import ROC_score_wrapper
from ROC_score import only_spatcov_wrapper
from ROC_score import plotting_timeseries

xarray_plot = func_CPPA.xarray_plot
xrplot = func_CPPA.xarray_plot


datafolder = 'era5'
path_pp  = os.path.join(data_base_path, 'Data_'+datafolder +'/input_pp') # path to netcdfs
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)


# =============================================================================
# General Settings
# =============================================================================
ex = {'datafolder'  :       datafolder,
      'grid_res'    :       1.0,
     'startyear'    :       1979,
     'endyear'      :       2018,
     'path_pp'      :       path_pp,
     'startperiod'  :       '06-24', #'1982-06-24',
     'endperiod'    :       '08-22', #'1982-08-22',
     'figpathbase'  :       os.path.join(basepath, 'McKinRepl/'),
     'RV1d_ts_path' :       os.path.join(basepath, 'MckinRepl/RVts'),
     'RVts_filename':       "comp_v_spclus4of4_tempclus2_AgglomerativeClustering_smooth15days_clus1_daily.npy",
     'RV_name'      :       'v_E-US',
     'name'         :       'sst',
     'add_lsm'      :       False,
     'region'       :       'Northern',
     'lags'         :       [0, 5, 10, 30], #[0, 5, 10, 15, 20, 30, 40, 50, 60], #[5, 15, 30, 50] #[10, 20, 30, 50] 
     'plot_ts'      :       True,
     'exclude_yrs'  :      []
     }
# =============================================================================
# Settings for event timeseries
# =============================================================================
ex['tfreq']                 =       1 
ex['max_break']             =       0   
ex['min_dur']               =       1
ex['event_percentile']      =       'std'
# =============================================================================
# Settins for precursor / CPPA
# =============================================================================
ex['filename_precur']       =       '{}_{}-{}_1jan_31dec_daily_{}deg.nc'.format(
                                    ex['name'], ex['startyear'], ex['endyear'], ex['grid_res'])
ex['rollingmean']           =       ('RV', 7)
ex['extra_wght_dur']        =       False
ex['prec_reg_max_d']        =       1
ex['SCM_percentile_thres']  =       95
ex['FCP_thres']             =       0.80
ex['min_perc_area']         =       0.02 # min size region - in % of total prec area [m2]
ex['wghts_accross_lags']    =       False
ex['perc_yrs_out']          =       [5,7.5,10,12.5,15] #[5, 10, 12.5, 15, 20] 
ex['days_before']           =       [0, 7, 14]
ex['store_timeseries']      =       False
# =============================================================================
# Settings for validation     
# =============================================================================
ex['leave_n_out']           =       True
ex['ROC_leave_n_out']       =       False
ex['method']                =       'iter' #'iter' or 'no_train_test_split' or split#8 or random3  
ex['n_boot']                =       0
# =============================================================================
# load data (write your own function load_data(ex) )
# =============================================================================
RV_ts, Prec_reg, ex = load_data.load_data(ex)

if ex['RVts_filename'].split('_')[1] == "spclus4of4" and ex['RV_name'][-1]=='S':
    ex['RV_name'] += '_' +ex['RVts_filename'].split('_')[-1][:-4] 
ex['exppathbase'] = '{}_{}_{}_{}'.format(datafolder, ex['RV_name'],ex['name'],
                      ex['region'])
ex['figpathbase'] = os.path.join(ex['figpathbase'], ex['exppathbase'])
if os.path.isdir(ex['figpathbase']) == False: os.makedirs(ex['figpathbase'])


print_ex = ['RV_name', 'name', 'max_break',
            'min_dur', 'event_percentile',
            'event_thres', 'extra_wght_dur',
            'grid_res', 'startyear', 'endyear', 
            'startperiod', 'endperiod', 'leave_n_out',
            'n_oneyr', 'wghts_accross_lags', 'add_lsm',
            'tfreq', 'lags', 'n_yrs', 'region',
            'rollingmean', 
            'SCM_percentile_thres', 'FCP_thres', 'perc_yrs_out', 'days_before',
            'min_perc_area', 'prec_reg_max_d', 
            'ROC_leave_n_out', 'method', 'n_boot',
            'RVts_filename', 'path_pp']


def printset(print_ex=print_ex, ex=ex):
    max_key_len = max([len(i) for i in print_ex])
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline)


printset()
n = 1
ex['n'] = n ; lag=0
#%% 

# =============================================================================
# Run code with ex settings
# =============================================================================

l_ds_CPPA, ex = func_CPPA.main(RV_ts, Prec_reg, ex)
   


# save ex setting in text file
output_dic_folder = ex['output_dic_folder']
if os.path.isdir(output_dic_folder):
    answer = input('Overwrite?\n{}\ntype y or n:\n\n'.format(output_dic_folder))
    if 'n' in answer:
        assert (os.path.isdir(output_dic_folder) != True)
    elif 'y' in answer:
        pass

if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)

# save output in numpy dictionary
filename = 'output_main_dic'
if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)
to_dict = dict( { 'ex'      :   ex,
                 'l_ds_CPPA' : l_ds_CPPA} )
np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)  

# write output in textfile
print_ex.append('output_dic_folder')
txtfile = os.path.join(output_dic_folder, 'experiment_settings.txt')
with open(txtfile, "w") as text_file:
    max_key_len = max([len(i) for i in print_ex])
    for key in print_ex:
        key_len = len(key)
        expand = max_key_len - key_len
        key_exp = key + ' ' * expand
        printline = '\'{}\'\t\t{}'.format(key_exp, ex[key])
        print(printline, file=text_file)



if ex['store_timeseries'] == True:
    if ex['method'] == 'iter': 
        eps = 10
        l_ds_CPPA, ex = func_CPPA.grouping_regions_similar_coords(l_ds_CPPA, ex, 
                         grouping = 'group_accros_tests_single_lag', eps=10)
    if ex['method'][:6] == 'random':
        eps = 11
        l_ds_CPPA, ex = func_CPPA.grouping_regions_similar_coords(l_ds_CPPA, ex, 
                         grouping = 'group_across_test_and_lags', eps=eps)
        
    func_CPPA.plot_precursor_regions(l_ds_CPPA, 2, 'pat_num_CPPA', [0], [''], ex)
    print('\n\n\nCheck labelling\n\n\n')
    func_CPPA.plot_precursor_regions(l_ds_CPPA, 10, 'pat_num_CPPA_clust', [0], [''], ex)
    
    
    func_CPPA.store_ts_wrapper(l_ds_CPPA, RV_ts, Prec_reg, ex)
    
    
    ex = func_pred.spatial_cov(ex, key1='spatcov_CPPA')
    ex = ROC_score_wrapper(ex)
    args = ['python output_wrapper.py {}'.format(output_dic_folder)]
    func_CPPA.kornshell_with_input(args, ex)





#%% 

# =============================================================================
# Load and Generate output in console
# =============================================================================

filename = 'output_main_dic'
dic = np.load(os.path.join(output_dic_folder, filename+'.npy'),  encoding='latin1').item()
#%%
# load settings
ex = dic['ex']
# load patterns
l_ds_CPPA = dic['l_ds_CPPA']

#ex['min_perc_area'] = ex['min_perc_prec_area']
#ex['n_boot'] = 0
#ex['SCM_percentile_thres'] = ex['perc_map']
#ex['FCP_thres'] = ex['comp_perc']


# write output in textfile
if 'use_ts_logit' in ex.keys() and 'pval_logit_final' in ex.keys():
    predict_folder = '{}{}_ts{}'.format(ex['pval_logit_final'], ex['logit_valid'], ex['use_ts_logit'])
else:
    predict_folder = 'spatcov_CPPA'
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






#%%
# =============================================================================
# perform prediciton        
# =============================================================================

# write output in textfile
if 'use_ts_logit' in ex.keys():
    if ex['use_ts_logit'] == True:
        predict_folder = '{}{}_ts{}'.format(ex['pval_logit_final'], ex['logit_valid'], ex['use_ts_logit'])
else:
    ex['use_ts_logit'] = False
    predict_folder = 'spatcov_CPPA'
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
    ex = ROC_score_wrapper(ex)
else:
    key_pattern_num = 'pat_num_CPPA'
    ex = only_spatcov_wrapper(l_ds_CPPA, RV_ts, Prec_reg, ex)
if ex['use_ts_logit'] == False: ex.pop('use_ts_logit')

score_AUC       = np.round(ex['score'][-1][0], 2)
ROC_str_Sem      = ['{} days - ROC score {}'.format(ex['lags'][i], score_AUC[i]) for i in range(len(ex['lags'])) ]
ROC_boot = [np.round(np.percentile(ex['score'][-1][1][i],99), 2) for i in range(len(ex['lags']))]

ex['score_AUC']   = score_AUC
ex['ROC_boot_99'] = ROC_boot

filename = 'output_main_dic'
to_dict = dict( { 'ex'      :   ex,
                 'l_ds_CPPA' : l_ds_CPPA} )
np.save(os.path.join(output_dic_folder, filename+'.npy'), to_dict)  

#%%
# =============================================================================
#   Plotting
# =============================================================================


lats = Prec_reg.latitude
lons = Prec_reg.longitude
array = np.zeros( (len(l_ds_CPPA), len(ex['lags']), len(lats), len(lons)) )
patterns_Sem = xr.DataArray(data=array, coords=[range(len(l_ds_CPPA)), ex['lags'], lats, lons], 
                      dims=['n_tests', 'lag','latitude','longitude'], 
                      name='{}_tests_patterns_Sem'.format(len(l_ds_CPPA)), attrs={'units':'Kelvin'})


for n in range(len(ex['train_test_list'])):
    name_for_ts = 'CPPA'
        
    if (ex['method'][:6] == 'random'):
        if n == len(l_ds_CPPA):
            # remove empty n_tests
            patterns_Sem = patterns_Sem.sel(n_tests=slice(0,len(l_ds_CPPA)))

    
    upd_pattern = l_ds_CPPA[n]['pattern_' + name_for_ts].sel(lag=ex['lags'])
    patterns_Sem[n,:,:,:] = upd_pattern * l_ds_CPPA[n]['std_train_min_lag']


    
kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                    'vmin' : -0.4, 'vmax' : 0.4, 'subtitles' : ROC_str_Sem,
                   'cmap' : plt.cm.RdBu_r, 'column' : 1} )

mean_n_patterns = patterns_Sem.mean(dim='n_tests')
mean_n_patterns.attrs['units'] = 'mean over {} runs'.format(ex['n_conv'])
mean_n_patterns.attrs['title'] = 'CPPA - Precursor Pattern'
mean_n_patterns.name = 'ROC {}'.format(score_AUC)
filename = os.path.join('', 'mean_over_{}_tests'.format(ex['n_conv']) )
func_CPPA.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)

print(ROC_boot)


#%% Robustness accross training sets
#    ex['lags'] = [5,15,30,50]

lats = patterns_Sem.latitude
lons = patterns_Sem.longitude
array = np.zeros( (ex['n_conv'], len(ex['lags']), len(lats), len(lons)) )
wgts_tests = xr.DataArray(data=array, 
                coords=[range(ex['n_conv']), ex['lags'], lats, lons], 
                dims=['n_tests', 'lag','latitude','longitude'], 
                name='{}_tests_wghts'.format(ex['n_conv']), attrs={'units':'wghts ['})
for n in range(ex['n_conv']):
#    wgts_tests[n,:,:,:] = l_ds_CPPA[n]['weights'].sel(lag=ex['lags'])
    wgts_tests[n,:,:,:] = l_ds_CPPA[n]['weights'].sel(lag=ex['lags']).where(l_ds_CPPA[n]['pat_num_CPPA']>0.5)
    
from matplotlib.colors import LinearSegmentedColormap 
if ex['leave_n_out']:
    n_lags = len(ex['lags'])
    n_lats = patterns_Sem.sel(n_tests=0).latitude.size
    n_lons = patterns_Sem.sel(n_tests=0).longitude.size
    
    pers_patt = patterns_Sem.sel(n_tests=0).sel(lag=ex['lags']).copy()
#    arrpatt = np.nan_to_num(patterns_Sem.values)
#    mask_patt = (arrpatt != 0)
#    arrpatt[mask_patt] = 1
    wghts = np.zeros( (n_lags, n_lats, n_lons) )
#    plt.imshow(arrpatt[0,0]) ; plt.colorbar()
    for l in ex['lags']:
        i = ex['lags'].index(l)
        wghts[i] = np.nansum(wgts_tests[:,i,:,:].values, axis=0)
    pers_patt.values = wghts 
    pers_patt = pers_patt.where(pers_patt.values != 0)
    pers_patt -= 1E-9
    size_trainset = ex['n_yrs'] - ex['leave_n_years_out']
    pers_patt.attrs['units'] = 'No. of times in final pattern [0 ... {}]'.format(ex['n_conv'])
    pers_patt.attrs['title'] = ('Robustness SST pattern\n{} different '
                            'training sets (n={} yrs)'.format(ex['n_conv'],size_trainset))
    filename = os.path.join('', 'Robustness_across_{}_training_tests'.format(ex['n_conv']) )
    vmax = ex['n_conv'] 
    mean = np.round(pers_patt.mean(dim=('latitude', 'longitude')).values, 1)
#    mean = pers_patt.quantile(0.80, dim=('latitude','longitude')).values
    std =  np.round(pers_patt.std(dim=('latitude', 'longitude')).values, 0)
    ax_text = ['mean = {}±{}'.format(mean[l],int(std[l])) for l in range(len(ex['lags']))]
    colors = plt.cm.magma_r(np.linspace(0,0.7, 20))
    colors[-1] = plt.cm.magma_r(np.linspace(0.99,1, 1))
    cm = LinearSegmentedColormap.from_list('test', colors, N=255)
    kwrgs = dict( {'title' : pers_patt.attrs['title'], 'clevels' : 'notdefault', 
                   'steps' : 11, 'subtitles': ROC_str_Sem, 
                   'vmin' : max(0,vmax-20), 'vmax' : vmax, 'clim' : (max(0,vmax-20), vmax),
                   'cmap' : cm, 'column' : 2, 'extend':['min','yellow'],
                   'cbar_vert' : 0.05, 'cbar_hght' : 0.01,
                   'adj_fig_h' : 1.25, 'adj_fig_w' : 1., 
                   'hspace' : 0.02, 'wspace' : 0.08
                    } )
    func_CPPA.plotting_wrapper(pers_patt, ex, filename, kwrgs=kwrgs)



#%% Weighing features if there are extracted every run (training set)
# weighted by persistence of pattern over
if ex['leave_n_out']:
    kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                    'vmin' : -0.4, 'vmax' : 0.4, 'subtitles' : ROC_str_Sem,
                   'cmap' : plt.cm.RdBu_r, 'column' : 1,
                   'cbar_vert' : 0.02, 'cbar_hght' : -0.01,
                   'adj_fig_h' : 0.9, 'adj_fig_w' : 1., 
                   'hspace' : 0.2, 'wspace' : 0.08,
                   'title_h' : 0.95} )
    # weighted by persistence (all years == wgt of 1, less is below 1)
    mean_n_patterns = patterns_Sem.mean(dim='n_tests') * wghts/np.max(wghts)
    mean_n_patterns['lag'] = ROC_str_Sem

    title = 'Precursor Pattern'
    if mean_n_patterns.sum().values != 0.:
        mean_n_patterns.attrs['units'] = 'Kelvin'
        mean_n_patterns.attrs['title'] = title
                             
        mean_n_patterns.name = 'ROC {}'.format(score_AUC)
        filename = os.path.join('', ('{}_Precursor_pattern_robust_w_'
                             '{}_tests'.format(ex['datafolder'], ex['n_conv']) ))

        func_CPPA.plotting_wrapper(mean_n_patterns, ex, filename, kwrgs=kwrgs)




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
    Robustness_weights = l_ds_CPPA[n]['weights'].sel(lag=ex['lags'])
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
xarray_plot(ex['mask'], path=folder, name='RV_mask', saving=True)
    
func_CPPA.plot_oneyr_events(RV_ts, ex, 2012, ex['output_dic_folder'], saving=True)
## plotting same figure as in paper
#for i in range(2005, 2010):
#    func_CPPA.plot_oneyr_events(RV_ts, ex, i, folder, saving=True)

#%% Plotting prediciton time series vs truth:
if ex['method'] == 'iter':
    yrs_to_plot = [1983, 1988, 1994, 2002, 2007, 2012, 2015]
    if 'n_events' in ex.keys():
        sorted_idx = np.argsort(ex['n_events'])
        sorted_n_events = ex['n_events'].copy(); sorted_n_events.sort()
        yrs_to_plot = [ex['tested_yrs'][n][0] for n in sorted_idx[:6]]
        [yrs_to_plot.append(ex['tested_yrs'][n][0]) for n in sorted_idx[-5:]]

    test = ex['train_test_list'][0][1]        
    plotting_timeseries(test, yrs_to_plot, ex) 


#%% Initial regions from only composite extraction:

lags = ex['lags']
if ex['leave_n_out']:
    subfolder = os.path.join(ex['exp_folder'], 'intermediate_results')
    total_folder = os.path.join(ex['figpathbase'], subfolder)
    if os.path.isdir(total_folder) != True : os.makedirs(total_folder)
    subtitles = [ROC_str_Sem[ex['lags'].index(l)] for l in lags]
    
        
    func_CPPA.plot_precursor_regions(l_ds_CPPA, ex['n_conv'], key_pattern_num, lags, subtitles, ex)

#%%
for i, yr in enumerate(ex['all_yrs']):
    print(yr, ex['n_events'][i])

#%%
#pd.DataFrame(ex['test_ts_prec'][0]).plot.kde(bw_method=0.1)
#pd.DataFrame(ex['test_RV'][0]).plot.kde(bw_method=0.1)
#pd.DataFrame(ex['test_RV'][0]*ex['test_ts_prec'][0]).plot()
#pd.DataFrame(ex['test_ts_prec'][0]).plot()
