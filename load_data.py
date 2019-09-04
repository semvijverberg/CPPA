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
import func_fc
from dateutil.relativedelta import relativedelta as date_dt
flatten = lambda l: [item for sublist in l for item in sublist]


def load_response_variable(ex):
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

    RVfullts, lpyr = load_1d(filename, ex, ex['RV_aggregation'])
    if ex['tfreq'] != 1:
        RVfullts, dates = func_CPPA.time_mean_bins(RVfullts, ex, ex['tfreq'])
    
    RVhour   = RVfullts.time[0].dt.hour.values
    dates_all = pd.to_datetime(RVfullts.time.values)
    
    
    datesRV = func_CPPA.make_datestr(dates_all, ex, 
                            ex['startyear'], ex['endyear'], lpyr=lpyr)
 

    if ex['rollingmean'][0] == 'RV' and ex['rollingmean'][1] != 1:
        RVfullts = func_CPPA.rolling_mean_time(RVfullts, ex, center=True)
    
    if 'exclude_yrs' in ex.keys():
        print('excluding yr(s): {} from analysis'.format(ex['exclude_yrs']))
        
        all_yrs = np.unique(dates_all.year)
        yrs_keep = [y for y in all_yrs if y not in ex['exclude_yrs']]
        idx_yrs =  [i for i in np.arange(dates_all.year.size) if dates_all.year[i] in yrs_keep]
#        dates_all = dates_all[idx_yrs]
        mask_all    = np.zeros(dates_all.size, dtype=bool)
        mask_all[idx_yrs] = True
        
        idx_yrs =  [i for i in np.arange(datesRV.year.size) if datesRV.year[i] in yrs_keep]
        mask_RV    = np.zeros(datesRV.size, dtype=bool)
        mask_RV[idx_yrs] = True
        
        
        dates_all = dates_all[mask_all]
        datesRV = datesRV[mask_RV]
    
    ex['dates_all']  = pd.to_datetime(np.array(dates_all, dtype='datetime64[D]'))
    ex['dates_RV'] = pd.to_datetime(np.array(datesRV, dtype='datetime64[D]'))
    # add RVhour to daily dates
    datesRV = datesRV + pd.Timedelta(int(RVhour), unit='h')
    ex['endyear'] = int(datesRV[-1].year)
    
    # Selected Time series of T95 ex['sstartdate'] until ex['senddate']
    RV_ts = RVfullts.sel(time=datesRV)
    ex['n_oneyr'] = func_CPPA.get_oneyr(datesRV).size
    

    #expanded_time = func_mcK.expand_times_for_lags(dates, ex)
    
    if ex['RVts_filename'][:8] == 'nino3.4_' and 'event_thres' in ex.keys():
        ex['event_thres'] = ex['event_thres']
    else:
        event_percentile = ex['kwrgs_events']['event_percentile']
        ex['event_thres'] = func_fc.Ev_threshold(RV_ts, event_percentile)

    ex['n_yrs'] = int(len(set(RV_ts.time.dt.year.values)))
    ex['endyear'] = int(datesRV[-1].year)
    
    
    RV = RV_class(RVfullts, RV_ts, kwrgs_events=ex['kwrgs_events'])

    #%%
    return RV, ex

def load_precursor(ex):
    #%%
    dates_all = ex['dates_all']
   # =============================================================================
    # Load Precursor     
    # =============================================================================
    prec_filename = os.path.join(ex['path_pp'], ex['filename_precur'])
#    if ex['datafolder'] == 'EC':
#        try:
#            datesRV = func_CPPA.make_datestr(dates_all, ex, 
#                            ex['startyear'], ex['endyear'], lpyr=False)
#            dates_prec = subset_dates(datesRV, ex)
#            varfullgl = func_CPPA.import_ds_lazy(prec_filename, ex, seldates=dates_prec)
#        except:
#            datesRV = func_CPPA.make_datestr(dates_all, ex, 
#                                    ex['startyear'], ex['endyear'], lpyr=True)
#            dates_prec = subset_dates(datesRV, ex)
#            varfullgl = func_CPPA.import_ds_lazy(prec_filename, ex, seldates=dates_prec)
#    else:
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
    Prec_reg = Prec_reg.sel(time=dates_all)
    
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
    
    
    
    if 'exclude_yrs' in ex.keys():
        if len(ex['exclude_yrs']) != 0:
            print('excluding yr(s): {} from analysis'.format(ex['exclude_yrs']))
            
            all_yrs = np.unique(dates_all.year)
            yrs_keep = [y for y in all_yrs if y not in ex['exclude_yrs']]
            idx_yrs =  [i for i in np.arange(dates_all.year.size) if dates_all.year[i] in yrs_keep]
    #        dates_all = dates_all[idx_yrs]
            mask_all    = np.zeros(dates_all.size, dtype=bool)
            mask_all[idx_yrs] = True
            dates_excl_yrs = dates_all[mask_all]
            Prec_reg = Prec_reg.sel(time= dates_excl_yrs)
       

    #%%
    return Prec_reg, ex

class RV_class:
    def __init__(self, RVfullts, RV_ts, kwrgs_events=None):
        self.RV_ts = RV_ts
        self.RVfullts = RVfullts
        if type(RVfullts) == type(xr.DataArray([0])):
            self.dfRV_ts = RV_ts.drop('quantile').to_dataframe(name='RVfullts')
            self.dfRVfullts = RVfullts.drop('quantile').to_dataframe(name='RVfullts')
        self.dates_all = pd.to_datetime(self.dfRVfullts.index)
        self.dates_RV = pd.to_datetime(self.dfRV_ts.index)
        self.n_oneRVyr = self.dates_RV[self.dates_RV.year == self.dates_RV.year[0]].size
        if kwrgs_events is not None:
            self.threshold = func_fc.Ev_threshold(self.dfRV_ts, 
                                              kwrgs_events['event_percentile'])
#            self.RV_b_full = func_fc.Ev_timeseries(self.RVfullts, 
#                               threshold=self.threshold , 
#                               min_dur=kwrgs_events['min_dur'],
#                               max_break=kwrgs_events['max_break'], 
#                               grouped=kwrgs_events['grouped'])[0]
            self.RV_bin   = func_fc.Ev_timeseries(self.dfRV_ts, 
                               threshold=self.threshold , 
                               min_dur=kwrgs_events['min_dur'],
                               max_break=kwrgs_events['max_break'], 
                               grouped=kwrgs_events['grouped'])[0]
            self.freq      = func_fc.get_freq_years(self)
        
        

def load_data(ex):
    RVfullts, RV_ts, ex = load_response_variable(ex)
    Prec_reg, ex = load_precursor(ex)
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
            RVfullts = dicRV[name]
        except:
            RVfullts = dicRV['RVfullts']
        if ex['datafolder'] == 'ERAint':
            try:
                ex['mask'] = dicRV['RV_array']['mask']
            except:
                ex['mask'] = dicRV['mask']
        elif ex['datafolder'] == 'era5' and 'mask' in dicRV.keys():
            ex['mask'] = dicRV['mask']
        if ex['datafolder'] == 'ERAint' or ex['datafolder'] == 'era5':
            func_CPPA.xarray_plot(ex['mask'])


        
        lpyr = False
    else:  
        lpyr = True
        RVfullts = csv_to_xarray(ex, filename, delim_whitespace=False, header=None)
    return RVfullts, lpyr
