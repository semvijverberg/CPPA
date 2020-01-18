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
RGCPD_dir  = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/RGCPD'
if script_dir not in sys.path: sys.path.append(script_dir)
if RGCPD_dir not in sys.path: sys.path.append(RGCPD_dir)
if sys.version[:1] == '3':
    from importlib import reload as rel

import cartopy.crs as ccrs
import numpy as np
import xarray as xr 
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import func_CPPA
import functions_pp
import load_data
import find_precursors
import plot_maps
import climate_indices


xarray_plot = func_CPPA.xarray_plot
xrplot = func_CPPA.xarray_plot



import init.era5_t2mmax_E_US_sst as settings
#import init.ERAint_t2mmax_E_US_sst as settings
#import init.EC_t2m_E_US as settings

# experiments
#import init.EC_t2m_E_US_grouped_hot_days as settings
#import init.era5_t2mmax_W_US_sst as settings

# bram equal era 5 mask
#import init.bram_e5mask_era5_t2mmax_E_US_sst as settings
#import init.bram_e5mask_ERAint_t2mmax_E_US_sst as settings
#import init.bram_e5mask_EC_t2m_E_US_sst as settings

ex = settings.__init__()
#ex['RV_aggregation'] = 'RVfullts_mean'

# =============================================================================
# load data (write your own function load_data(ex) )
# =============================================================================
RV, ex = load_data.load_response_variable(ex)

precur_arr, ex = load_data.load_precursor(ex)

print_ex = ['RV_name', 'name', 'kwrgs_events',
            'event_thres', 
            'grid_res', 'startyear', 'endyear', 
            'startperiod', 'endperiod', 
            'n_oneyr', 'add_lsm',
            'tfreq', 'lags', 'n_yrs', 'region',
            'rollingmean', 'seed',
            'SCM_percentile_thres', 'FCP_thres', 'perc_yrs_out', 'days_before',
            'method', 
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
 



#%%

# =============================================================================
# Run code with ex settings
# =============================================================================
#ex['lags'] = np.array([0]) ; ex['method'] = 'no_train_test_split' ; 
today = datetime.datetime.today().strftime("%d-%m-%y_%Hhr")
df_splits = functions_pp.rand_traintest_years(RV, method=ex['method'], 
                                                  seed=ex['seed'], 
                                                  kwrgs_events=ex['kwrgs_events'])
#%%
kwrgs_CPPA = {'perc_yrs_out':[5, 7.5, 10, 12.5, 15],
              'days_before':[0, 7, 14],
              'FCP_thres':0.8,
              'SCM_percentile_thres':95}
lags_i=np.array([10])

CPPA_prec = func_CPPA.get_robust_precursors(precur_arr, RV, df_splits, 
                                            lags_i=lags_i,
                                            kwrgs_CPPA=kwrgs_CPPA)
#%%
actor = func_CPPA.act('sst', CPPA_prec, precur_arr)
kwrgs_cluster = {'distance_eps' : 500,
                 'min_area_in_degrees2' : 5} # minimal size to become precursor region (core sample)
actor = find_precursors.cluster_DBSCAN_regions(actor, **kwrgs_cluster)
CPPA_prec['prec_labels'] = actor.prec_labels
if np.isnan(actor.prec_labels.values).all() == False:
    plot_maps.plot_labels(actor.prec_labels.copy())
actor.ts_corr = find_precursors.spatial_mean_regions(actor)    
# get PEP
PEP_xr = func_CPPA.get_PEP(precur_arr, RV, df_splits, lags_i)

ds = xr.Dataset({'sst' : CPPA_prec, 'PEP' : PEP_xr})
fname = '{}_{}_output.nc'.format(ex['datafolder'], today)
ds.to_netcdf(os.path.join(ex['path_data_out'], fname))                   
dict_ds = {'sst' : ds}  



    

    
def ts_lag(actor, lag):
    n_spl = actor.ts_corr.shape[0]
    df_ts_s = np.zeros( (n_spl) , dtype=object)
    
    for s in range(n_spl):
        all_keys = actor.ts_corr[s].columns
        cols = [k for k in all_keys if int(k.split('..')[0]) == lag]
        df_ts_s[s] = actor.ts_corr[s][cols]
    df_ts = pd.concat(list(df_ts_s), keys= range(n_spl), sort=True)
    return df_ts

def add_spactov(dict_ds, lag, df_data_ts, actor):
    n_spl =df_data_ts.index.levels[0].size
    df_sp_s   = np.zeros( (n_spl) , dtype=object)
    outdic_actors = {'sst' : actor}
    for s in range(n_spl):
        df_split = df_data_ts.loc[s]
        all_cols = list(actor.ts_corr[0].columns)
        cols = [col for col in all_cols if int(col.split('..')[0]) == lag]
        [cols.append(k) for k in ['TrainIsTrue', 'RV_mask'] ]
        df_split_lag = df_split[cols]
        df_sp_s[s] = func_CPPA.get_spatcovs(dict_ds, df_split_lag, s, lag, outdic_actors, normalize=False)
    df_sp = pd.concat(list(df_sp_s), keys= range(n_spl))
    return df_sp.merge(df_data_ts, left_index=True, right_index=True)

def add_RV(df_data_lag, RV):
    n_spl = df_data_lag.index.levels[0].size
    df_RV_s   = np.zeros( (n_spl) , dtype=object)
    for s in range(n_spl):
        df_RV_s[s] = pd.DataFrame(RV.fullts.values, 
                                columns=[ex['RV_name']],
                                index=df_data_lag.loc[0].index)
    df_RV = pd.concat(list(df_RV_s), keys= range(n_spl))
    return df_RV.merge(df_data_lag, left_index=True, right_index=True)

#store_data(dict_ds, actor, ex)

for l, lag in enumerate(lags_i):
    
    
    df_data_ts = ts_lag(actor, lag)
    df_data_ts = df_data_ts.merge(df_splits, left_index=True, right_index=True)
    df_data_lag = add_spactov(dict_ds, lag, df_data_ts, actor)
    if l == 0:
        # PDO and ENSO do not depend on lag
        filepath = os.path.join(ex['path_pp'], ex['filename_precur'])
#        if ex['datafolder'] == 'EC':
#            df_PDO, PDO_patterns = climate_indices.PDO(actor.precur_arr, df_splits)            
#        else:
        df_PDO, PDO_patterns = climate_indices.PDO(filepath, df_splits)
        df_data_lag = df_PDO.merge(df_data_lag, left_index=True, right_index=True)
        
        df_ENSO_34  = climate_indices.ENSO_34(filepath, df_splits)
        df_data_lag = df_ENSO_34.merge(df_data_lag, left_index=True, right_index=True)
    df_data_lag = add_RV(df_data_lag, RV)

    
    dict_of_dfs = {'df_data':df_data_lag}
    fname = '{}_{}_lag_{}.h5'.format(ex['datafolder'], today, lag)
    file_path = os.path.join(ex['path_data_out'], fname)
    functions_pp.store_hdf_df(dict_of_dfs, file_path)

#actor.ts_corr[ex['RV_name']] = pd.Series(RV.RVfullts.values, index=actor.ts_corr[0].index)
central_lon_plots = 200
map_proj = ccrs.LambertCylindrical(central_longitude=central_lon_plots)
kwrgs_corr = {'clim' : (-0.5, 0.5), 'hspace':-0.6}
pdfs_folder = os.path.join(ex['path_fig'], 'pdfs')
if os.path.isdir(pdfs_folder) != True : os.makedirs(pdfs_folder)


f_format = '.png'
lags_to_plot = [0, 20, 50]
contour_mask = (CPPA_prec['prec_labels'] > 0).sel(lag=lags_to_plot).astype(bool)
plot_maps.plot_corr_maps(CPPA_prec.sel(lag=lags_to_plot), 
                         contour_mask, 
                         map_proj, **kwrgs_corr)
lags_str = str(lags_to_plot).replace(' ','').replace('[', '').replace(']','').replace(',','_')
fig_filename = 'corr_{}_vs_{}_{}'.format(ex['RV_name'], 'sst', lags_str) + f_format

if f_format == '.pdf':
    plt.savefig(os.path.join(pdfs_folder, fig_filename),
            bbox_inches='tight')
elif f_format == '.png':
    plt.savefig(os.path.join(ex['path_fig'], fig_filename),
            bbox_inches='tight')    

#%%


## =============================================================================
## Experiment diff tfreq
## =============================================================================
###frequencies = np.array(np.arange(0, 140., 2.5),dtype=int) ; frequencies[0]=int(1)
##frequencies = np.array(np.arange(0, 90., 2.5),dtype=int) ; frequencies[0]=int(1)
##frequencies = frequencies[~np.logical_and(frequencies>=65, frequencies <70)]
##lags_to_test = [0, 1] 
## =============================================================================
## Experiment best tfreq
## =============================================================================
#frequencies = [1]
#lags_to_test = np.arange(0,100/frequencies[0],max(1,int(5/frequencies[0])))
#
##keys = ['spatcov_CPPA'] ; ext_ts = None
##keys = ['spatcov_CPPA', '2', '3','4'] ; ext_ts = None
#
#keys = ['spatcov_CPPA'] 
#ext_ts = [tuple( [os.path.join(ex['output_ts_folder'], 'sm.csv'), tuple(['sm_rm20']) ] )]
#
#'''     for ERA5 [2,3,4,5,6,7]
#        East-Pac = 3
#        CARIBEAN = 6
#        MONSOON = 5
#        LAKES = 2
#        ICELAND = 7
#        Mid-Pac = 4 
#        '''
#        
#if keys != ['spatcov_CPPA']:
#    output_dic_folder = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/random10fold_leave_4_out_1979_2018_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_1rmRV_rng50_2019-06-24/different_keys/' 
#    if os.path.isdir(output_dic_folder) != True : os.makedirs(output_dic_folder)
#else:
#    output_dic_folder = ex['output_dic_folder']
#
#scores_ = ['BSS', 'BSS_low', 'BSS_high', 'AUC', 'AUC_low', 'AUC_high']
#data=np.zeros( (len(frequencies), len(lags_to_test), len(scores_)), 
#              dtype=float)
#summary = xr.DataArray(data, coords=[frequencies, lags_to_test, scores_], dims=['frequency', 'lag', 'score'])
#dict_tfreq = {}
#for i, t in enumerate(frequencies):
#    print('tfreq:', t)
#    ex['tfreq'] = int(t)
#    ex['lags'] = [l*t for l in lags_to_test]
#    SCORE, ex = ROC_score.CV_wrapper(RV_ts, ex, path_ts, lag_to_load=lag_to_load, 
#                                     keys=keys, ext_ts=ext_ts)
#    dict_tfreq[t] = SCORE
#    summary[i,:,0] = np.array(SCORE.brier_logit.loc['BSS'])
#    summary[i,:,1] = np.array(SCORE.brier_logit.loc['BSS_high'])
#    summary[i,:,2] = np.array(SCORE.brier_logit.loc['BSS_low'])
#    summary[i,:,3] = np.array(SCORE.AUC_spatcov.loc['AUC'])
#    summary[i,:,4] = np.array(SCORE.AUC_spatcov.loc['con_low'])
#    summary[i,:,5] = np.array(SCORE.AUC_spatcov.loc['con_high'])
##    summary[i,:,6] = np.array(SCORE.other_metrics.loc['precision']) # tp / (tp + fp)
##    summary[i,:,7] = np.array(SCORE.other_metrics.loc['recall']) # tp / (tp + fn)
##    summary[i,:,8] = np.array(SCORE.other_metrics.loc['FPR']) # fp / (fp + tn)
##    summary[i,:,9] = np.array(SCORE.other_metrics.loc['f1_score']) # 2 * PREC * REC / (PREC + REC)
##    summary[i,:,10] = np.array(SCORE.other_metrics.loc['Accuracy']) # correct pred / all preds
#
#precursors = '_'.join(ex['pred_names'])
#filename_3 = 'list_tfreqs_{}_{}_lag{:.0f}-{:.0f}_trh{}_nb{}_{}'.format(frequencies[0], frequencies[-1],
#                          lags_to_test[0],lags_to_test[-1], 
#                          ex['event_percentile'], ex['n_boot'], precursors)
#to_dict = dict( { 'dict_tfreq'      :   dict_tfreq,
#                     'ex' : ex } )
#np.save(os.path.join(output_dic_folder, filename_3+'.npy'), to_dict) 
#
#
#
##%%
#
#to_dict = np.load(os.path.join(output_dic_folder, filename_3+'.npy'),  encoding='latin1').item()
#dict_tfreq = to_dict['dict_tfreq'] ; ex = to_dict['ex']
#frequencies = list(dict_tfreq.keys())
#precursors = '_'.join(ex['pred_names'])
#
##%%
#
#metric = 'BSS'
#x = list(dict_tfreq.keys())
#ROC_score.plot_score_freq(dict_tfreq, 'BSS', lags_to_test)
#lags_str = str(lags_to_test).replace(' ' ,'')        
#f_name = '{}_tfreqs_{}-{}_lag{}_thr{}_nb{}.png'.format(metric, x[0], x[-1], 
#          lags_str, ex['event_percentile'], ex['n_boot'])
#filename = os.path.join(output_dic_folder, 'AUC_and_BSS', f_name)
#plt.savefig(filename, dpi=600, bbox_inches='tight')
#plt.show()
#
#metric = 'AUC'
#x = list(dict_tfreq.keys())
#ROC_score.plot_score_freq(dict_tfreq, 'AUC', lags_to_test)
#lags_str = str(lags_to_test).replace(' ' ,'')        
#f_name = '{}_tfreqs_{}-{}_lag{}_thr{}_nb{}.png'.format(metric, x[0], x[-1], 
#          lags_str, ex['event_percentile'], ex['n_boot'])
#filename = os.path.join(output_dic_folder, 'AUC_and_BSS', f_name)
#plt.savefig(filename, dpi=600, bbox_inches='tight')
#plt.show()
##%%
#
#
##%%
#to_dict = np.load(os.path.join(output_dic_folder, filename_3+'.npy'),  encoding='latin1').item()
#dict_tfreq = to_dict['dict_tfreq'] ; ex = to_dict['ex']
#f = frequencies[-1]
#n_shuffle = 0
##summary.loc[freq][0].to_dataframe('Summary')
#SCORE = dict_tfreq[f]
#df_sum = ROC_score.add_scores_wrt_random(SCORE, n_shuffle=n_shuffle)
#f_name = '{}_lag{:.0f}-{:.0f}_tf{}_thr{}_ns{}_{}'.format('valid', SCORE._lags[0], 
#          SCORE._lags[-1], SCORE.tfreq,
#          ex['event_percentile'], n_shuffle, precursors)
#df_sum.to_excel(os.path.join(output_dic_folder, f_name+ '.xlsx'))
#%%
filename = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_19-09-19_12hr_output.nc'
#filename = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/ran_strat10_s30/data/EC_16-09-19_19hr_output.nc'
ds = xr.open_dataset(filename)    
    
CPPA_prec = ds['sst']
# LSM mask
#LSM = np.isnan(CPPA_prec)
#mask = (('latitude', 'longitude', LSM.values))
#CPPA_prec['mask'] = LSM
#%%
# =============================================================================
#   Plotting
# =============================================================================
central_lon_plots = 200
map_proj = ccrs.LambertCylindrical(central_longitude=central_lon_plots)
f_format = '.png'
lags_plot = [0, 20, 50]
n_splits = CPPA_prec.split.size


    
mean_n_patterns = CPPA_prec.sel(lag=lags_plot).mean(dim='split')
sig_mask = (CPPA_prec['prec_labels'] > 0).sum(dim='split')
mean_mask       = sig_mask > 0.5 * n_splits # in more then 50% of splits

mean_n_patterns = mean_n_patterns.where(CPPA_prec['lsm'])
mean_n_patterns.attrs['units'] = '[K]'
mean_n_patterns.attrs['title'] = 'CPPA - Precursor Pattern'
subtitles = []
subtitles.append(['{} days'.format(l) for l in lags_plot  ])
subtitles = np.array(subtitles).T
mean_n_patterns.name = 'sst_mean_{}_traintest'.format(n_splits)


kwrgs_corr = {'row_dim':'lag', 'col_dim':'split', 'hspace':-0.3, 
              'size':3, 'cbar_vert':-0.025, 'clim':(-0.4,0.4),
              'subtitles' : subtitles, 'lat_labels':False}

# mcKinnon PEP box
west_lon = -215; east_lon = -130; south_lat = 20; north_lat = 50
kwrgs_corr['drawbox'] = ['all', (west_lon, east_lon, south_lat, north_lat)]

plot_maps.plot_corr_maps(mean_n_patterns, mean_mask, map_proj, **kwrgs_corr)
fig_filename = os.path.join('', 'mean_over_{}_tests_lags{}'.format(n_splits,
                        str(lags_plot).replace(' ' ,'')) ) + f_format

if f_format == '.pdf':
    plt.savefig(os.path.join(pdfs_folder, fig_filename),
            bbox_inches='tight')
elif f_format == '.png':
    plt.savefig(os.path.join(ex['path_fig'], fig_filename),
            bbox_inches='tight')  
plt.show()

#%% Robustness accross training sets

f_format = '.png'
lats = CPPA_prec.latitude
lons = CPPA_prec.longitude
array = np.zeros( (n_splits, len(lags_plot), len(lats), len(lons)) )
wgts_tests = CPPA_prec['weights'].sel(lag=lags_plot).sum(dim='split')
wgts_tests.attrs = {'units':'wghts [-]'}


    
from matplotlib.colors import LinearSegmentedColormap 

pers_patt = wgts_tests
pers_patt = pers_patt.where(pers_patt.values != 0)
pers_patt -= 1E-9
size_trainset = int(ex['n_yrs'] - (ex['n_yrs'] / ds.split.size))
pers_patt.attrs['units'] = 'No. of times in final pattern [0 ... {}]'.format(n_splits)
pers_patt.attrs['title'] = ('Robustness\n{} different '
                        'training sets (n={} yrs)'.format(n_splits,size_trainset))
fig_filename = os.path.join('', 'Robustness_across_{}_training_tests_lags{}'.format(n_splits,
                        str(lags_plot).replace(' ' ,'')) )
vmax = n_splits 
extend = ['min','yellow']
if vmax-20 <= n_splits: extend = ['min','white']

mean = np.round(pers_patt.mean(dim=('latitude', 'longitude')).values, 1)

std =  np.round(pers_patt.std(dim=('latitude', 'longitude')).values, 0)

ax_text = ['mean = {}±{}'.format(mean[l],int(std[l])) for l in range(len(lags_plot))]
colors = plt.cm.magma_r(np.linspace(0,0.7, 20))
colors[-1] = plt.cm.magma_r(np.linspace(0.99,1, 1))
cm = LinearSegmentedColormap.from_list('test', colors, N=255)
kwrgs = dict( {'title' : pers_patt.attrs['title'], 'clevels' : 'notdefault', 
               'steps' : 11, 'subtitles': subtitles, 
               'size' : 3,
               'vmin' : max(0,vmax-20), 'vmax' : vmax, 'clim' : (max(0,vmax-20), vmax),
               'cmap' : cm, 'column' : 1, 'extend':extend,
               'cbar_vert' : 0.04, 'cbar_hght' : -0.01,
               'adj_fig_h' : 1, 'adj_fig_w' : 1., 
               'hspace' : -0.1, 'wspace' : 0.04, 
               'title_h': 0.95} )
func_CPPA.plotting_wrapper(pers_patt, ex, filename, kwrgs=kwrgs, map_proj=map_proj)

if f_format == '.pdf':
    plt.savefig(os.path.join(pdfs_folder, fig_filename),
            bbox_inches='tight')
elif f_format == '.png':
    plt.savefig(os.path.join(ex['path_fig'], fig_filename),
            bbox_inches='tight') 


#%% Weighing features if there are extracted every run (training set)
# weighted by persistence of pattern over

kwrgs = dict( {'title' : '', 'clevels' : 'notdefault', 'steps':17,
                'vmin' : -0.4, 'vmax' : 0.4, 'subtitles' : subtitles,
               'cmap' : plt.cm.RdBu_r, 'column' : 1,
               'cbar_vert' : 0.04, 'cbar_hght' : -0.01,
               'adj_fig_h' : 1, 'adj_fig_w' : 1., 
               'hspace' : -0.1, 'wspace' : 0.04, 
               'title_h' : 0.95} )
# weighted by persistence (all years == wgt of 1, less is below 1)
final_pattern = mean_n_patterns * pers_patt/np.max(pers_patt)
final_pattern = mean_n_patterns.where(~np.isnan(pers_patt))
final_pattern['lag'] = subtitles

title = 'Precursor Pattern'
if final_pattern.sum().values != 0.:
    final_pattern.attrs['units'] = 'Kelvin'
    final_pattern.attrs['title'] = title
                         
    final_pattern.name = ''
    filename = os.path.join('', ('{}_Precursor_pattern_robust_w_'
                         '{}_tests_lags{}'.format(ex['datafolder'], n_splits,
                          str(lags_plot).replace(' ' ,'')) ))

    func_CPPA.plotting_wrapper(final_pattern, ex, filename, kwrgs, map_proj)
    if f_format == '.pdf':
        plt.savefig(os.path.join(pdfs_folder, fig_filename),
            bbox_inches='tight')
    elif f_format == '.png':
        plt.savefig(os.path.join(ex['path_fig'], fig_filename),
            bbox_inches='tight') 

#%% plot precursor regions
f_format = '.png'
dpi = 300
lags_to_plot = [0, 20, 50]
prec_labels = CPPA_prec['prec_labels'].sel(lag=lags_to_plot)

# colors of cmap are dived over min to max in n_steps. 
# We need to make sure that the maximum value in all dimensions will be 
# used for each plot (otherwise it assign inconsistent colors)
max_N_regs = min(20, int(prec_labels.max() + 0.5))
label_weak = np.nan_to_num(prec_labels.values) >=  max_N_regs
contour_mask = None
prec_labels.values[label_weak] = max_N_regs
steps = max_N_regs+1
cmap = plt.cm.tab20
prec_labels.values = prec_labels.values-0.5
clevels = np.linspace(0, max_N_regs,steps)

kwrgs_corr = {'row_dim':'split', 'col_dim':'lag', 'hspace':-0.35, 
              'size':3, 'cbar_vert':-0.025, 'clevels':clevels,
              'subtitles' : None, 'lat_labels':True, 
              'cticks_center':True,
              'cmap':cmap}        



plot_maps.plot_corr_maps(prec_labels, 
                 contour_mask, 
                 map_proj, **kwrgs_corr)

lags_str = str(lags_to_plot).replace(' ','').replace('[', '').replace(']','').replace(',','_')
fig_filename = 'labels_{}_vs_{}_{}'.format(ex['RV_name'], 'sst', lags_str) + f_format

if f_format == '.pdf':
    plt.savefig(os.path.join(pdfs_folder, fig_filename),
            bbox_inches='tight')
elif f_format == '.png':
    plt.savefig(os.path.join(ex['path_fig'], fig_filename),
            bbox_inches='tight', dpi=dpi)    

    
#%% plot ENSO / PDO maps
path_fig = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/figures'
path_data = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-24.h5'
df_ENSO_34 = func_fc.load_hdf5(path_data)['df_data'].loc[0]['0_900_ENSO34']
dt = (df_ENSO_34.index[1] - df_ENSO_34.index[0]).days
ENSO_5_months = int((5 * 30) / dt)
PDO = PDO_patterns.mean(dim='split')
ENSO_ts = df_ENSO_34.rolling(window=ENSO_5_months, min_periods=1).mean() #5month rm
ENSO_d = ENSO_ts[(ENSO_ts > 0.4).values].index
ENSO = precur_arr.sel(time=ENSO_d).mean(dim='time')

map_proj = ccrs.PlateCarree(central_longitude=220)

class MidpointNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)
    
            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))
            
fig = plt.figure(figsize=(20,10) )
ax1 = plt.subplot(1, 2, 1, projection=map_proj) 
#ax.set_xlim(-130, -65)
ax1.set_ylim(-10, 80)
colormap = plt.cm.coolwarm
vmin = -0.8; vmax=0.8
clevels = np.arange(-0.8, 0.8+1E-9, 0.2)
clevels = np.linspace(-0.8, 0.8, 17)
norm = MidpointNormalize(midpoint=0, vmin=clevels[0],vmax=clevels[-1])

im = ENSO.plot.contourf(ax=ax1, transform=ccrs.PlateCarree(), 
                          subplot_kws={'projection': map_proj},
                          cmap=colormap, center=0,
                          clevels=clevels, 
                          vmin=vmin, vmax=vmax, add_colorbar=False)
fontdict = dict({'fontsize'     : 18,
             'fontweight'   : 'bold'})
ax1.set_title('El Nino', fontdict=fontdict, loc='center')
ax1.coastlines()


ax2 = plt.subplot(1, 2, 2, projection=map_proj) 
ax2.set_ylim(20, 65)

PDO.plot.contourf(ax=ax2, transform=ccrs.PlateCarree(), 
                          subplot_kws={'projection': map_proj},
                          cmap=im.cmap, center=0,
                          clevels=clevels,
                          vmin=vmin, vmax=vmax, add_colorbar=False)
ax2.coastlines()

ax2.set_title('PDO pattern (positive phase)', fontdict=fontdict, loc='center')
cbar_ax = fig.add_axes([0.25, 0.35, 0.5, 0.02])
                              
plt.colorbar(im, cax=cbar_ax , orientation='horizontal', norm=norm,
                 label='[K]', ticks=clevels[::8], extend='neither')    
fig_filename = 'ENSO_PDO_patterns' + f_format

if f_format == '.pdf':
    plt.savefig(os.path.join(pdfs_folder, fig_filename),
            bbox_inches='tight')
elif f_format == '.png':
    plt.savefig(os.path.join(ex['path_fig'], fig_filename),
            bbox_inches='tight', dpi=dpi)  

#filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
#dicRV = np.load(filename,  encoding='latin1').item()
#folder = os.path.join(ex['figpathbase'], ex['exp_folder'])
#if 'mask' in ex.keys():
#    xarray_plot(ex['mask'], path=folder, name='RV_mask', saving=True)
#    
#func_CPPA.plot_oneyr_events(RV_ts, ex, 2012, ex['output_dic_folder'], saving=True)
## plotting same figure as in paper
#for i in range(2005, 2010):
#    func_CPPA.plot_oneyr_events(RV_ts, ex, i, folder, saving=True)

##%% Plotting prediciton time series vs truth:
#
##yrs_to_plot = [1983, 1988, 1994, 2002, 2007, 2012, 2015]
#ex['n_events'] = []
#all_years = np.unique(SCORE.y_true_test.index.year)
#for y in all_years:
#    n_ev = int(SCORE.y_true_test[0][SCORE.y_true_test[0].index.year==y].sum())
#    ex['n_events'].append(n_ev)
#if 'n_events' in ex.keys():
#    sorted_idx = np.argsort(ex['n_events'])
#    sorted_n_events = ex['n_events'].copy(); sorted_n_events.sort()
#    yrs_to_plot = [all_years[n] for n in sorted_idx[:6]]
#    [yrs_to_plot.append(all_years[n]) for n in sorted_idx[-5:]]
#
##    test = ex['train_test_list'][0][1]        
#plotting_timeseries(SCORE, 'spatcov', yrs_to_plot, ex) 
#plotting_timeseries(SCORE, 'logit', yrs_to_plot, ex) 

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
import validation as valid
import valid_plots as dfplots
valid.plot_freq_per_yr(RV)

fname = 'freq_per_year.png'
filename = os.path.join(ex['fig_path'], fname)
plt.savefig(filename) 

#%% get timeseries:

ERA5_filename = 'era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19.npy'
GHCND_filename = "PEP-T95TimeSeries.txt"


RV, ex = load_data.load_response_variable(ex)
T95_ERA5 = RV.RV_ts
T95_GHCND, GHCND_dates = load_data.read_T95(GHCND_filename, ex)
dates = functions_pp.get_oneyr(RV.dates_RV, 2012)
shared_dates = functions_pp.get_oneyr(RV.dates_RV, *list(range(1982, 2016)))
#%%
data = np.stack([T95_GHCND.sel(time=shared_dates).values, T95_ERA5.loc[shared_dates].values.squeeze()], axis=1)
df = pd.DataFrame(data, columns=['GHCND', 'ERA-5'], index=shared_dates)

dfplots.plot_oneyr_events(df, 'std', 2012)
plt.savefig(os.path.join(ex['path_fig'], 'timeseries_ERA5_GHCND.png'),
            bbox_inches='tight')
#%% Keep forecast the same, change event definition
import func_fc
import validation as valid
import classes
import pandas as pd

path_fig = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/figures'
path_data = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-24.h5'
df_data = func_fc.load_hdf5(path_data)['df_data']
splits  = df_data.index.levels[0]
RVfullts = pd.DataFrame(df_data[df_data.columns[0]][0])
RV_ts    = pd.DataFrame(df_data[df_data.columns[0]][0][df_data['RV_mask'][0]] )
thresholds = [1,10,20,30,40,50,60,70,80,90,99]
blocksize = valid.get_bstrap_size(RV.RVfullts, plot=False)
metric = 'AUC-ROC'
df_list = [] ; scores = []; 

for r in thresholds:
    kwrgs_events = {'event_percentile': r,
                    'min_dur' : 1,
                    'max_break' : 0,
                    'grouped' : False}
    
    y_pred = pd.DataFrame(df_data['0_101_PEPspatcov'][0])
    y_pred = y_pred[df_data['RV_mask'].loc[0].values]
    RV = classes.RV_class(RVfullts, RV_ts, kwrgs_events, 
                          fit_model_dates=None)
    RV.TrainIsTrue = df_data['TrainIsTrue']
    RV.RV_mask = df_data['RV_mask']
    y_true = RV.RV_bin.values
    y_pred_c = func_fc.get_obs_clim(RV)
    
    df_ = valid.get_metrics_sklearn(RV, y_pred, y_pred_c, alpha=0.05, n_boot=0, blocksize=blocksize)[0]
    df_list.append(df_)
    scores.append(float(df_.loc[metric].loc[metric].values))




df_AUC_ = pd.DataFrame(scores, index=thresholds, columns=['PEP']) 
ax = df_AUC_.plot()
ax.set_ylabel(metric)
if metric == 'AUC-ROC':
    ax.set_ylim(0.5, 1.0)
    ax.vlines(100-13.6, 0.53, 1, alpha=0.5, linewidth=0.75)
    ax.text(100-13.6, 0.5, '+1std', 
        horizontalalignment='center', alpha=0.7)
elif metric == 'EDI':
        ax.set_ylim(-1., 1.0)
ax.set_xlabel('Event Percentile\n(1 if value > p else 0)')


plt.savefig(os.path.join(path_fig, f'diff_event_threshold_{metric}.png'),
            bbox_inches='tight')


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


