#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:07:55 2019

@author: semvijverberg
"""

import os
import numpy as np
import datetime


if os.path.isdir("/Users/semvijverberg/surfdrive/"):
    basepath = "/Users/semvijverberg/surfdrive/"
    data_base_path = basepath
else:
    basepath = "/home/semvij/"
#    data_base_path = "/p/tmp/semvij/ECE"
    data_base_path = "/p/projects/gotham/semvij"
os.chdir(os.path.join(basepath, 'Scripts/CPPA/CPPA'))

datafolder = 'ERA5'
path_pp  = os.path.join(data_base_path, datafolder +'/input_raw/preprocessed') # path to netcdfs
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)

# orig paper1 v1, 20-03-2020
# 'era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19_Xzkup1.npy'

# "comp_v_spclus4of4_tempclus2_AgglomerativeClustering_smooth15days_clus1_daily.npy"
# "era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4.npy"
# "ERAint_t2mmax_US_1979-2017_averAggljacc0.75d_tf1_n4__to_t2mmax_US_tf1.npy"
# "heatwave_ECE.csv"
# =============================================================================
# General Settings
# =============================================================================
def __init__():
    #%%
    ex = {'datafolder'  :       datafolder,
          'grid_res'    :       1.0,
         'startyear'    :       1979,
         'endyear'      :       2018,
         'path_pp'      :       path_pp,
         'startperiod'  :       '06-24', #'1982-06-24',
         'endperiod'    :       '08-22', #'1982-08-22',
         'sstartdate'   :       '01-01', # precursor period
         'senddate'     :       '12-31', # precursor period
         'figpathbase'  :       os.path.join(basepath, 'McKinRepl/'),
         'RV1d_ts_path' :       "/Users/semvijverberg/surfdrive/output_RGCPD/easternUS/",
         'RVts_filename':       "tf1_n_clusters4_q90_dendo_ff393.nc", 
         'RV_name'      :       'mx2t',
         'name'         :       'sst',
         'add_lsm'      :       False,
         'region'       :       'Northern',
         'lags'         :       np.array([0, 10, 20, 50]), #[0, 10, 20, 35, 50, 65]
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

    ex['filename_precur']   	=  '{}_{}-{}_1jan_31dec_daily_{}deg.nc'.format(
                                    ex['name'], ex['startyear'], ex['endyear'], ex['grid_res'])

    ex['selbox'] 				=  {'lo_min':	-180, 'lo_max':360, 'la_min':-10, 'la_max':80}
    
        
        
    ex['rollingmean']           =       ('RV', 1)
    ex['extra_wght_dur']        =       False
    ex['SCM_percentile_thres']  =       95
    ex['FCP_thres']             =       0.85
    ex['wghts_accross_lags']    =       False
    ex['perc_yrs_out']          =       [5,10,15,20,25] #[5, 10, 12.5, 15, 20] 
    ex['days_before']           =       [0, 7, 14]
    ex['store_timeseries']      =       False

    # =============================================================================
    # settings precursor region selection
    # =============================================================================   
    ex['distance_eps'] = 500 # proportional to km apart from a core sample, standard = 1000 km
    ex['min_area_in_degrees2'] = 5 # minimal size to become precursor region (core sample)
    ex['group_split'] = 'together' # choose 'together' or 'seperate'
    # =============================================================================
    # Train test split
    # =============================================================================
    ###options###
    # (1) random{int}   :   with the int(ex['method'][6:8]) determining the amount of folds
    # (2) ran_strat{int}:   random stratified folds, stratified based upon events, 
    #                       requires kwrgs_events.    
    # (3) leave{int}    :   chronologically split train and test years.
    # (4) split{int}    :   split dataset into single train and test set
    # (5) no_train_test_split
    
    ex['method'] = 'ran_strat10' ; ex['seed'] = 30
    # settings for output
    ex['folder_sub_1'] = f"{ex['method']}_s{ex['seed']}"
    ex['params'] = ''
    ex['file_type2'] = 'png'
    
    if ex['RVts_filename'].split('_')[1] == "spclus4of4" and ex['RV_name'][-1]=='S':
        ex['RV_name'] += '_' +ex['RVts_filename'].split('_')[-1][:-4] 
    ex['folder_sub_0'] = '{}_{}_{}_{}'.format(ex['datafolder'], ex['RV_name'],ex['name'],
                          ex['region'])


#    ex['exp_folder'] = sub_output  + '/figures/' 

    
    #%%
    return ex


seed = 30
test_years = np.array([[1981, 1982, 2000, 2002],
                         [1983, 1994, 2006, 2012],
                         [1990, 2015, 2017, 2018],
                         [1997, 2001, 2004, 2005],
                         [1979, 1989, 2008, 2010],
                         [1986, 1991, 1995, 1999],
                         [1980, 1993, 1996, 2009],
                         [1987, 2011, 2013, 2016],
                         [1984, 1985, 1992, 1998],
                         [1988, 2003, 2007, 2014]])