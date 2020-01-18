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
import functions_pp
import core_pp
import func_fc
import class_RV
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
    ex['hash'] = filename.split('_')[-1].split('.')[0]
    RVfullts, lpyr = load_1d(filename, ex, ex['RV_aggregation'])
    if ex['tfreq'] != 1:
        RVfullts, dates = functions_pp.time_mean_bins(RVfullts, ex, ex['tfreq'])
    
    RVhour   = RVfullts.time[0].dt.hour.values
    dates_all = pd.to_datetime(RVfullts.time.values)
    
    start_end_TVdate = (ex['startperiod'], ex['endperiod'])
    datesRV = core_pp.get_subdates(dates_all, 
                                          start_end_TVdate, lpyr=lpyr)
    
    
    RVfullts, dates_all = functions_pp.timeseries_tofit_bins(RVfullts, 
                                                             to_freq=1)
 

    if ex['rollingmean'][0] == 'RV' and ex['rollingmean'][1] != 1:
        RVfullts = func_CPPA.rolling_mean_time(RVfullts, ex, center=True)
    
    if 'exclude_yrs' in ex.keys():
#        print('excluding yr(s): {} from analysis'.format(ex['exclude_yrs']))
        
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
    RV_ts = RVfullts.sel(time=pd.to_datetime(datesRV))
    ex['n_oneyr'] = func_CPPA.get_oneyr(datesRV).size
    

    #expanded_time = func_mcK.expand_times_for_lags(dates, ex)
    
    if ex['RVts_filename'][:8] == 'nino3.4_' and 'event_thres' in ex.keys():
        ex['event_thres'] = ex['event_thres']
    else:
        event_percentile = ex['kwrgs_events']['event_percentile']
        ex['event_thres'] = func_fc.Ev_threshold(RV_ts, event_percentile)

    ex['n_yrs'] = int(len(set(RV_ts.time.dt.year.values)))
    ex['endyear'] = int(datesRV[-1].year)
    
    df_RVfullts = pd.DataFrame(RVfullts.values, columns=['RVfullts'],
                               index = dates_all)
    df_RV_ts = pd.DataFrame(RV_ts.values, columns=['RV_ts'],
                            index = datesRV)
    RV = class_RV.RV_class(df_RVfullts, df_RV_ts, kwrgs_events=ex['kwrgs_events'])
    
    ex['path_data_out']    = os.path.join(ex['figpathbase'], ex['folder_sub_0'], 
                                  ex['hash']+'_'+ex['folder_sub_1'], 'data')
    if os.path.isdir(ex['path_data_out']) == False: os.makedirs(ex['path_data_out'])

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
##            varfullgl = func_CPPA.import_ds_lazy(prec_filename, ex, seldates=dates_prec)
#        except:
#            datesRV = func_CPPA.make_datestr(dates_all, ex, 
#                                    ex['startyear'], ex['endyear'], lpyr=True)
#            dates_prec = subset_dates(datesRV, ex)
#            varfullgl = func_CPPA.import_ds_lazy(prec_filename, ex, seldates=dates_prec)
#    else:
    Prec_reg = functions_pp.import_ds_timemeanbins(prec_filename, ex['tfreq'],
                                             loadleap=True, to_xarr=False, 
                                             seldates=ex['dates_all'])
    Prec_reg = core_pp.convert_longitude(Prec_reg, 'only_east')
    if ex['add_lsm']:
        kwrgs_2d = {'selbox' : ex['selbox'], 'format_lon':'only_east'}
        lsm_filename = os.path.join(ex['path_mask'], ex['mask_file'])
        lsm = core_pp.import_ds_lazy(lsm_filename, **kwrgs_2d)   
        
        Prec_reg['lsm'] = (('latitude', 'longitude'), (lsm < 0.3).values)
        Prec_reg = Prec_reg.where(Prec_reg['lsm'])

    
    
    
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
    newoneyr = core_pp.remove_leapdays(newoneyr)
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

def read_T95(T95name, ex):
    filepath = os.path.join(ex['RV1d_ts_path'], T95name)
    if filepath[-3:] == 'txt':
        data = pd.read_csv(filepath)
        datelist = []
        values = []
        for r in data.values:
            year = int(r[0][:4])
            month = int(r[0][5:7])
            day = int(r[0][7:11])
            string = '{}-{}-{}'.format(year, month, day)
            values.append(float(r[0][10:]))
            datelist.append( pd.Timestamp(string) )
    elif filepath[-3:] == 'csv':
        data = pd.read_csv(filepath, sep='\t')
        datelist = []
        values = []
        for r in data.iterrows():
            year = int(r[1]['Year'])
            month = int(r[1]['Month'])
            day =   int(r[1]['Day'])
            string = '{}-{}-{}T00:00:00'.format(year, month, day)
            values.append(float(r[1]['T95(degC)']))
            datelist.append( pd.Timestamp(string) )
    dates = pd.to_datetime(datelist)
    RVts = xr.DataArray(values, coords=[dates], dims=['time'])
    return RVts, dates