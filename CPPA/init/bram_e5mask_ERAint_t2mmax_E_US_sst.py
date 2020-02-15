#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:07:55 2019

@author: semvijverberg
"""

import os


if os.path.isdir("/Users/semvijverberg/surfdrive/"):
    basepath = "/Users/semvijverberg/surfdrive/"
    data_base_path = basepath
else:
    basepath = "/home/semvij/"
#    data_base_path = "/p/tmp/semvij/ECE"
    data_base_path = "/p/projects/gotham/semvij"
os.chdir(os.path.join(basepath, 'Scripts/CPPA/CPPA'))

datafolder = 'ERAint'
path_pp  = os.path.join(data_base_path, 'Data_'+datafolder +'/input_pp') # path to netcdfs
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)


# "comp_v_spclus4of4_tempclus2_AgglomerativeClustering_smooth15days_clus1_daily.npy"
# "era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4.npy"
# "ERAint_t2mmax_US_1979-2017_averAggljacc0.75d_tf1_n4__to_t2mmax_US_tf1.npy"
# "heatwave_ECE.csv"
# =============================================================================
# General Settings
# =============================================================================
def __init__():
    ex = {'datafolder'  :       datafolder,
          'grid_res'    :       1.0,
         'startyear'    :       1979,
         'endyear'      :       2017,
         'path_pp'      :       path_pp,
         'startperiod'  :       '06-24', #'1982-06-24',
         'endperiod'    :       '08-22', #'1982-08-22',
         'figpathbase'  :       os.path.join(basepath, 'McKinRepl/'),
         'RV1d_ts_path' :       os.path.join(basepath, 'MckinRepl/RVts'),
         'RVts_filename':       "T95_ERA_INTERIM.csv", 
         'RV_name'      :       't2mmax',
         'name'         :       'sst',
         'add_lsm'      :       False,
         'region'       :       'Northern',
         'lags'         :       [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75], #[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
         'plot_ts'      :       True,
         'exclude_yrs'  :       []
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

    ex['filename_precur']   =       '{}_{}-{}_1jan_31dec_daily_{}deg.nc'.format(
                                            ex['name'], ex['startyear'], ex['endyear'], ex['grid_res'])

    
        
        
    ex['rollingmean']           =       ('RV', 1)
    ex['extra_wght_dur']        =       False
    ex['prec_reg_max_d']        =       1
    ex['SCM_percentile_thres']  =       95
    ex['FCP_thres']             =       0.80
    ex['min_perc_area']         =       0.02 # min size region - in % of total prec area [m2]
    ex['min_area_in_degrees2']  =       3
    ex['distance_eps_init']     =       275 # km apart from cores sample, standard = 300
    ex['wghts_accross_lags']    =       False
    ex['perc_yrs_out']          =       [5,7.5,10,12.5,15] #[5, 10, 12.5, 15, 20] 
    ex['days_before']           =       [0, 7, 14]
    ex['store_timeseries']      =       True
    # =============================================================================
    # Settings for validation     
    # =============================================================================
    ex['leave_n_out']           =       True
    ex['ROC_leave_n_out']       =       False
    ex['method']                =       'random10fold' #'iter' or 'no_train_test_split' or split#8 or random3  
    ex['n_boot']                =       1
    
    if ex['RVts_filename'].split('_')[1] == "spclus4of4" and ex['RV_name'][-1]=='S':
        ex['RV_name'] += '_' +ex['RVts_filename'].split('_')[-1][:-4] 
    ex['exppathbase'] = '{}_{}_{}_{}'.format(ex['datafolder'], ex['RV_name'],ex['name'],
                          ex['region'])
    ex['figpathbase'] = os.path.join(ex['figpathbase'], ex['exppathbase'])
    if os.path.isdir(ex['figpathbase']) == False: os.makedirs(ex['figpathbase'])
    return ex