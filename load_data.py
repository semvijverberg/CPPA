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
from dateutil.relativedelta import relativedelta as date_dt
flatten = lambda l: [item for sublist in l for item in sublist]


def load_data(ex):
    #%%
    #'Mckinnonplot', 'U.S.', 'U.S.cluster', 'PEPrectangle', 'Pacific', 'Whole', 'Northern', 'Southern'  
  

    # load ERA-i Time series
    if 'RV_aggregation' not in ex.keys():
        ex['RV_aggregation']  = 'RVfullts95'
    else:
        ex['RV_aggregation'] = ex['RV_aggregation']
    
    print('\nimportRV_1dts is true, so the 1D time serie given with name \n'
              '{} is imported, importing {}.'.format(ex['RVts_filename'],
               ex['RV_aggregation']))
    
    filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])

    RVtsfull, lpyr = load_1d(filename, ex, ex['RV_aggregation'])
    RVhour   = RVtsfull.time[0].dt.hour.values

    datesRV = func_CPPA.make_datestr(pd.to_datetime(RVtsfull.time.values), ex, 
                            ex['startyear'], ex['endyear'], lpyr=lpyr)
    # =============================================================================
    # Load Precursor     
    # =============================================================================
    prec_filename = os.path.join(ex['path_pp'], ex['filename_precur'])
    if ex['datafolder'] == 'EC':
        try:
            datesRV = func_CPPA.make_datestr(pd.to_datetime(RVtsfull.time.values), ex, 
                            ex['startyear'], ex['endyear'], lpyr=False)
            dates_prec = subset_dates(datesRV, ex)
            varfullgl = func_CPPA.import_ds_lazy(prec_filename, ex, seldates=dates_prec)
        except:
            datesRV = func_CPPA.make_datestr(pd.to_datetime(RVtsfull.time.values), ex, 
                                    ex['startyear'], ex['endyear'], lpyr=True)
            dates_prec = subset_dates(datesRV, ex)
            varfullgl = func_CPPA.import_ds_lazy(prec_filename, ex, seldates=dates_prec)
    else:
        varfullgl = func_CPPA.import_ds_lazy(prec_filename, ex, loadleap=True)
    
    # =============================================================================
    # Ensure same longitude  
    # =============================================================================
    if varfullgl.longitude.min() < -175 and varfullgl.longitude.max() > 175:
        varfullgl = func_CPPA.convert_longitude(varfullgl, 'only_east')

    # =============================================================================
    # Select a focus region  
    # =============================================================================
    Prec_reg = func_CPPA.find_region(varfullgl, region=ex['region'])[0]
    
    if ex['tfreq'] != 1:
        Prec_reg, datesvar = func_CPPA.time_mean_bins(Prec_reg, ex)


    ## filter out outliers 
    if ex['name'][:2]=='sm':
        Prec_reg = Prec_reg.where(Prec_reg.values < 5.*Prec_reg.std(dim='time').values)
    
    if ex['add_lsm'] == True:
        filename = os.path.join(ex['path_mask'], ex['mask_file'])
        mask = func_CPPA.import_array(filename, ex)
                                    
        if len(mask.shape) == 3:
            mask = mask[0].squeeze()
            
        if 'latitude' and 'longitude' not in mask.dims:
            mask = mask.rename({'lat':'latitude',
                       'lon':'longitude'})
    
        mask_reg = func_CPPA.find_region(mask, region=ex['region'])[0]
        mask_reg = mask_reg.squeeze()
        mask_reg = np.array(mask_reg.values < 0.35, dtype=bool)

        mask = (('latitude', 'longitude'), mask_reg)
        Prec_reg.coords['mask'] = mask
        Prec_reg = Prec_reg.where(mask_reg==True)
#        xarray_plot(Prec_reg[0])
    
    if ex['rollingmean'][0] == 'RV' and ex['rollingmean'][1] != 1:
        RVtsfull = func_CPPA.rolling_mean_time(RVtsfull, ex, center=True)
    
    if 'exclude_yrs' in ex.keys():
        print('excluding yr(s): {} from analysis'.format(ex['exclude_yrs']))
        dates_prec = pd.to_datetime(Prec_reg.time.values)
        all_yrs = np.unique(dates_prec.year)
        yrs_keep = [y for y in all_yrs if y not in ex['exclude_yrs']]
        idx_yrs =  [i for i in np.arange(dates_prec.year.size) if dates_prec.year[i] in yrs_keep]
        mask    = np.zeros(dates_prec.size, dtype=bool)
        mask[idx_yrs] = True
        dates_excl_yrs = dates_prec[mask]
        Prec_reg = Prec_reg.sel(time= dates_excl_yrs)
        idx_yrs =  [i for i in np.arange(datesRV.year.size) if datesRV.year[i] in yrs_keep]
        mask    = np.zeros(datesRV.size, dtype=bool)
        mask[idx_yrs] = True
        datesRV = datesRV[mask]
        
    ex['dates_RV'] = datesRV
    # add RVhour to daily dates
    datesRV = datesRV + pd.Timedelta(int(RVhour), unit='h')
    ex['endyear'] = int(datesRV[-1].year)
    
    # Selected Time series of T95 ex['sstartdate'] until ex['senddate']
    RV_ts = RVtsfull.sel(time=datesRV)
    ex['n_oneyr'] = func_CPPA.get_oneyr(datesRV).size
    
    if ex['tfreq'] != 1:
        RV_ts, dates = func_CPPA.time_mean_bins(RV_ts, ex)
    #expanded_time = func_mcK.expand_times_for_lags(dates, ex)
    
    if ex['RVts_filename'][:8] == 'nino3.4_' and 'event_thres' in ex.keys():
        ex['event_thres'] = ex['event_thres']
    else:
        if ex['event_percentile'] == 'std':
            # binary time serie when T95 exceeds 1 std
            ex['event_thres'] = RV_ts.mean(dim='time').values + RV_ts.std().values
        else:
            percentile = ex['event_percentile']
            ex['event_thres'] = np.percentile(RV_ts.values, percentile)

    ex['n_yrs'] = len(set(RV_ts.time.dt.year.values))
    

    #%%
    return RV_ts, Prec_reg, ex

    
def subset_dates(datesRV, ex):
    oneyr = func_CPPA.get_oneyr(datesRV)
    newstart = (oneyr[0] - pd.Timedelta(max(ex['lags']), 'd') \
                - pd.Timedelta(31, 'd') )
    newend   = (oneyr[-1] + pd.Timedelta(31, 'd') )
    newoneyr = pd.DatetimeIndex(start=newstart, end=newend,
                                freq=datesRV[1] - datesRV[0])
    newoneyr = func_CPPA.remove_leapdays(newoneyr)
    return make_dates(datesRV, newoneyr, breakyr=None)

def make_dates(datetime, start_yr, breakyr=None):
    if breakyr == None:
        breakyr = datetime.year.max()
        
    nyears = (breakyr - datetime.year[0])+1
    next_yr = start_yr
    for yr in range(0,nyears-1):
        next_yr = pd.to_datetime([date + date_dt(years=1) for date in next_yr])
        start_yr = start_yr.append(next_yr)
        if next_yr[-1].year == breakyr:
            break
    return start_yr

def csv_to_xarray(ex, path, delim_whitespace=True, header='infer'):
    '''ATTENTION: This only works if values are in last column'''
   # load data from csv file and save to .npy as xarray format
   
#    path = os.path.join(ex['path_pp'], 'RVts', ex['RVts_filename'])
    table = pd.read_csv(path, sep=',', delim_whitespace=delim_whitespace, header=header )
    if str(table.columns[0]).lower() == 'year':
        dates = pd.to_datetime(['{}-{}-{}'.format(r[0],r[1],r[2]) for r in table.iterrows()])
    elif len(table.iloc[0][0].split('-')) >= 2:
        dates = pd.to_datetime(table.values.T[0])
        
    y_val = np.array(table.values[:,-1], dtype='float32')  

    xrdata = xr.DataArray(data=y_val, coords=[dates], dims=['time'])
    return xrdata

def load_1d(filename, ex, name='RVfullts95'):
    if ex['RVts_filename'][-4:] == '.npy':
        
        dicRV = np.load(filename,  encoding='latin1', allow_pickle=True).item()
        try:    
            RVtsfull = dicRV[name]
        except:
            RVtsfull = dicRV['RVfullts']
        if ex['datafolder'] == 'ERAint':
            try:
                ex['mask'] = dicRV['RV_array']['mask']
            except:
                ex['mask'] = dicRV['mask']
        elif ex['datafolder'] == 'era5':
            ex['mask'] = dicRV['mask']
        if ex['datafolder'] == 'ERAint' or ex['datafolder'] == 'era5':
            func_CPPA.xarray_plot(ex['mask'])

        
        lpyr = False
    else:  
        lpyr = True
        RVtsfull = csv_to_xarray(ex, filename, delim_whitespace=False, header=None)
    return RVtsfull, lpyr
