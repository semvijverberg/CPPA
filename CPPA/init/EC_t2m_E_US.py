#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:50:24 2019

@author: semvijverberg
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:07:55 2019

@author: semvijverberg
"""

import os
import numpy as np

if os.path.isdir("/Users/semvijverberg/surfdrive/"):
    basepath = "/Users/semvijverberg/surfdrive/"
    data_base_path = basepath
else:
    basepath = "/home/semvij/"
#    data_base_path = "/p/tmp/semvij/ECE"
    data_base_path = "/p/projects/gotham/semvij"
os.chdir(os.path.join(basepath, 'Scripts/CPPA/CPPA'))

datafolder = 'EC'
path_pp  = os.path.join(data_base_path, 'Data_'+datafolder +'/input_pp') # path to netcdfs
path_raw  = os.path.join(data_base_path, 'Data_'+datafolder +'/input_raw') # path to netcdfs
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
          'grid_res'    :       1.125,
         'startyear'    :       2000,
         'endyear'      :       2159,
         'path_pp'      :       path_pp,
         'path_raw'     :       path_raw,
         'input_freq'   :       'daily',
         'startperiod'  :       '06-24', #'1982-06-24',
         'endperiod'    :       '08-22', #'1982-08-22',
         'sstartdate'   :       '01-01', # precursor period
         'senddate'     :       '09-30', # precursor period
         'figpathbase'  :       os.path.join(basepath, 'McKinRepl/'),
         'RV1d_ts_path' :       "/Users/semvijverberg/surfdrive/output_RGCPD/easternUS_EC/958dd_ran_strat10_s30",
         # 'RVts_filename':       "EC_tas_2000-2159_averAggljacc1.125d_tf1_n4__to_tas_tf1_selclus2.npy", 
         'RVts_filename':       "tf1_n_clusters5_q95_dendo_958dd.nc", 
         'RV_name'      :       'tas',
         'name'         :       'tos',
         'add_lsm'      :       False,
         'region'       :       'Northern',
         'lags'         :       np.array([0, 10, 20, 50]), # [0, 10, 20, 30], 
         'plot_ts'      :       True,
         'exclude_yrs'  :       [],
         'verbosity'    :       1,         
         }
    # =============================================================================
    # Settings for event timeseries
    # =============================================================================
    ex['tfreq']                 =       1 
    ex['kwrgs_events']          =   { 'event_percentile':'std',
                                      'max_break' : 0,
                                      'min_dur'   : 1,
                                      'grouped'   : False }
    ex['RV_aggregation']        =       'q90tail'
    # =============================================================================
    # Settins for precursor / CPPA
    # =============================================================================
    if ex['add_lsm'] == True:
        ex['path_mask'] = path_raw
        ex['mask_file'] = 'EC_earth2.3_LSM_T159.nc'
        
    ex['filename_precur']   =       'sst_2000-2159_with_lsm.nc'

    ex['selbox'] 				=  {'lo_min':	-180, 'lo_max':360, 'la_min':-10, 'la_max':80}
    # =============================================================================
    # settings precursor region selection
    # =============================================================================   
    ex['distance_eps'] = 500 # proportional to km apart from a core sample, standard = 1000 km
    ex['min_area_in_degrees2'] = 5 # minimal size to become precursor region (core sample)
    ex['group_split'] = 'together' # choose 'together' or 'seperate'  
        
        
    ex['rollingmean']           =       ('RV', 1)
    ex['SCM_percentile_thres']  =       95
    ex['FCP_thres']             =       0.85
    ex['perc_yrs_out']          =       [10, 20, 30, 40] #[7.5,10,12.5,15]  
    ex['days_before']           =       [0, 7, 14]
    ex['store_timeseries']      =       False
    # =============================================================================
    # Settings for validation     
    # =============================================================================   
    ex['method'] = 'ran_strat10' ; ex['seed'] = 30
    # settings for output
    ex['folder_sub_1'] = f"{ex['method']}_s{ex['seed']}"
    ex['params'] = ''
    ex['file_type2'] = 'png'
    
    if ex['RVts_filename'].split('_')[1] == "spclus4of4" and ex['RV_name'][-1]=='S':
        ex['RV_name'] += '_' +ex['RVts_filename'].split('_')[-1][:-4] 
    ex['folder_sub_0'] = '{}_{}_{}_{}'.format(ex['datafolder'], ex['RV_name'],ex['name'],
                          ex['region'])
    return ex


    