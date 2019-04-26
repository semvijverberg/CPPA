#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:39:41 2019

@author: semvijverberg
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
import func_CPPA



def load_data(ex):
    #%%
    #'Mckinnonplot', 'U.S.', 'U.S.cluster', 'PEPrectangle', 'Pacific', 'Whole', 'Northern', 'Southern'
    def oneyr(datetime):
        return datetime.where(datetime.year==datetime.year[0]).dropna()
    
  

    # load ERA-i Time series
    print('\nimportRV_1dts is true, so the 1D time serie given with name \n'
              '{} is imported.'.format(ex['RVts_filename']))
    filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
    dicRV = np.load(filename,  encoding='latin1').item()
    RVtsfull = dicRV['RVfullts95']
    if ex['datafolder'] == 'ERAint':
        ex['mask'] = dicRV['RV_array']['mask']
    elif ex['datafolder'] == 'era5':
        ex['mask'] = dicRV['mask']
    if ex['datafolder'] == 'ERAint' or ex['datafolder'] == 'era5':
        func_CPPA.xarray_plot(ex['mask'])
        lpyr = False
    else:
        lpyr = True
    RVhour   = RVtsfull.time[0].dt.hour.values
    datesRV = func_CPPA.make_datestr(pd.to_datetime(RVtsfull.time.values), ex, 
                                    ex['startyear'], ex['endyear'], lpyr=lpyr)
    ex['dates_RV'] = datesRV
    # add RVhour to daily dates
    datesRV = datesRV + pd.Timedelta(int(RVhour), unit='h')
    ex['endyear'] = int(datesRV[-1].year)
    
    # Selected Time series of T95 ex['sstartdate'] until ex['senddate']
    RVts = RVtsfull.sel(time=datesRV)
    ex['n_oneyr'] = oneyr(datesRV).size
    
    RV_ts, dates = func_CPPA.time_mean_bins(RVts, ex)
    #expanded_time = func_mcK.expand_times_for_lags(dates, ex)
    
    if ex['event_percentile'] == 'std':
        # binary time serie when T95 exceeds 1 std
        ex['event_thres'] = RV_ts.mean(dim='time').values + RV_ts.std().values
    else:
        percentile = ex['event_percentile']
        ex['event_thres'] = np.percentile(RV_ts.values, percentile)

    
    # Load in external ncdf
    
    #filename_precur = 'sm2_1979-2017_2jan_31okt_dt-1days_{}deg.nc'.format(ex['grid_res'])
    #path = os.path.join(ex['path_raw'], 'tmpfiles')
    # full globe - full time series
    varfullgl = func_CPPA.import_array(ex['filename_precur'], ex)
    if varfullgl.longitude.min() < -175 and varfullgl.longitude.max() > 175:
        varfullgl = func_CPPA.convert_longitude(varfullgl, 'only_east')

    Prec_reg = func_CPPA.find_region(varfullgl, region=ex['region'])[0]
    
    if ex['tfreq'] != 1:
        Prec_reg, datesvar = func_CPPA.time_mean_bins(Prec_reg, ex)


    
    ## filter out outliers 
    if ex['name'][:2]=='sm':
        Prec_reg = Prec_reg.where(Prec_reg.values < 5.*Prec_reg.std(dim='time').values)
    
    if ex['add_lsm'] == True:
        base_path_lsm = '/Users/semvijverberg/surfdrive/Scripts/rasterio/'
        mask = func_CPPA.import_array(ex['mask_file'].format(ex['grid_res']), ex,
                                     base_path_lsm)
        mask_reg = func_CPPA.find_region(mask, region=ex['region'])[0]
        mask_reg = mask_reg.to_array().squeeze()
        mask = (('latitude', 'longitude'), mask_reg.values)
        Prec_reg.coords['mask'] = mask
        Prec_reg.values = Prec_reg * mask_reg
        


    ex['n_yrs'] = len(set(RV_ts.time.dt.year.values))
    
    #%%
    return RV_ts, Prec_reg, ex

