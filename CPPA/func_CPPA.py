#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:29:47 2018

@author: semvijverberg
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
from netCDF4 import num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib as mpl
from shapely.geometry.polygon import LinearRing
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import datetime, calendar
import find_precursors
import plot_maps

flatten = lambda l: [item for sublist in l for item in sublist]
import functions_pp

def get_oneyr(pddatetime, *args):
    dates = []
    pddatetime = pd.to_datetime(pddatetime)
    year = pddatetime.year[0]

    for arg in args:
        year = arg
        dates.append(pddatetime.where(pddatetime.year==year).dropna())
    dates = pd.to_datetime(flatten(dates))
    if len(dates) == 0:
        dates = pddatetime.where(pddatetime.year==year).dropna()
    return dates

# restore default
mpl.rcParams.update(mpl.rcParamsDefault)

from matplotlib import cycler
colors_nice = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors_nice)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='black')
plt.rc('ytick', direction='out', color='black')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

mpl.rcParams['figure.figsize'] = [7.0, 5.0]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 400

mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'



def get_robust_precursors(precur_arr, RV, df_splits, lags_i=np.array([1]), 
                          kwrgs_CPPA={}):
    #%%
#    v = ncdf ; V = array ; RV.RV_ts = ts of RV, time_range_all = index range of whole ts
    """
    This function calculates the correlation maps for precur_arr for different lags. 
    Field significance is applied to test for correltion.
    This function uses the following variables (in the ex dictionary)
    prec_arr: array
    time_range_all: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
    lag_steps: number of lags
    time_cycle: time cycyle of dataset, =12 for monthly data...
    RV_period: indices that matches the response variable time series
    alpha: significance level

    """
 

    n_spl = df_splits.index.levels[0].size
    # make new xarray to store results
    xrcorr = precur_arr.isel(time=0).drop('time').copy()
    # add     
    list_xr = [xrcorr.expand_dims('lag', axis=0) for i in range(lags_i.size)]
    xrcorr = xr.concat(list_xr, dim = 'lag')
    xrcorr['lag'] = ('lag', lags_i)
    # add train test split     
    list_xr = [xrcorr.expand_dims('split', axis=0) for i in range(n_spl)]
    xrcorr = xr.concat(list_xr, dim = 'split')           
    xrcorr['split'] = ('split', range(n_spl))
    
    print('\n{} - finding robust regions'.format(precur_arr.name))
    np_data = np.zeros_like(xrcorr.values)
    np_mask = np.zeros_like(xrcorr.values)
    np_wght = np.zeros_like(xrcorr.values)
    
   

    for s in xrcorr.split.values:
        progress = int(100 * (s+1) / n_spl)
        # =============================================================================
        # Split train test methods ['random'k'fold', 'leave_'k'_out', ', 'no_train_test_split']        
        # =============================================================================
        RV_mask     = df_splits.loc[s]['RV_mask']
        TrainIsTrue = df_splits.loc[s]['TrainIsTrue']
        RV_train = TrainIsTrue[np.logical_and(RV_mask, TrainIsTrue)].index
        RV_bin = RV.RV_bin.loc[RV_train]
        precur = precur_arr.sel(time=TrainIsTrue[TrainIsTrue].index)
#        dates_RV  = pd.to_datetime(RV_ts.time.values)
        n = RV_train.size ; r = int(100*n/RV.dates_RV.size )
        print(f"\rProgress traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")
#        dates_all = pd.to_datetime(precur.time.values)
#        string_RV = list(dates_RV.strftime('%Y-%m-%d'))
#        string_full = list(dates_all.strftime('%Y-%m-%d'))
#        RV_period = [string_full.index(date) for date in string_full if date in string_RV]
        
        
        CPPA_prec, weights = CPPA_single_split(RV_bin, precur, lags_i=lags_i,
                                               kwrgs_CPPA=kwrgs_CPPA)
        

        np_data[s] = CPPA_prec.data
        np_mask[s] = CPPA_prec.mask
        np_wght[s] = weights
        
    print("\n")
    xrcorr.values = np_data
    mask = (('split', 'lag', 'latitude', 'longitude'), np_mask )
    xrcorr.coords['mask'] = mask
    weights = (('split', 'lag', 'latitude', 'longitude'), np_wght )
    xrcorr.coords['weights'] = weights
    xrcorr.name = 'sst'
#    xrcorr['lsm'] = precur_arr['mask']
    #%%
    return xrcorr


def CPPA_single_split(RV_bin, precur, lags_i, kwrgs_CPPA):
    #%%  
    lats = precur.latitude
    lons = precur.longitude
    
    weights     = np.zeros( (len(lags_i), len(lats), len(lons)) )

       
    events = RV_bin[RV_bin==1].dropna().index
    RV_dates_train = RV_bin.index
    all_yrs_set = list(set(RV_dates_train.year.values))
    comp_years = list(events.year.values)
    mask_chunks = get_chunks(all_yrs_set, comp_years, kwrgs_CPPA['perc_yrs_out'])
    #%%
    Comp_robust = np.ma.zeros( (len(lats) * len(lons), len(lags_i)) )
    
    max_lag_dates = func_dates_min_lag(RV_dates_train, max(lags_i))[1]
    dates_lags  = sorted(np.unique(np.concatenate([max_lag_dates, RV_dates_train])))
    dates_lags  = pd.to_datetime(dates_lags)
    std_train_lag = precur.sel(time=dates_lags).std(dim='time', skipna=True)
    
    for idx, lag in enumerate(lags_i):
        
        events_min_lag = func_dates_min_lag(events, lag)[1]
        dates_train_min_lag = func_dates_min_lag(RV_dates_train, lag)[1]
    #        std_train_min_lag[idx] = precur.sel(time=dates_train_min_lag).std(dim='time', skipna=True)
    #        std_train_lag = std_train_min_lag[idx]
        
        _kwrgs_CPPA = {k:i for k, i in kwrgs_CPPA.items() if k != 'perc_yrs_out'}
        # extract precursor regions composite approach
        Comp_robust[:,idx], weights[idx] = extract_regs_p1(precur, mask_chunks, events_min_lag, 
                                             dates_train_min_lag, std_train_lag, **_kwrgs_CPPA)  

    Composite = Comp_robust[:,:].reshape( (len(lats), len(lons), len(lags_i)) )
    Composite = Composite.swapaxes(0,-1).swapaxes(1,2)
    Composite = Composite * weights
    #%%
    return Composite, weights

class act:
    def __init__(self, name, corr_xr, precur_arr):
        self.name = name
        self.corr_xr = corr_xr
        self.precur_arr = precur_arr
        self.lat_grid = precur_arr.latitude.values
        self.lon_grid = precur_arr.longitude.values
        self.area_grid = find_precursors.get_area(precur_arr)
        self.grid_res = abs(self.lon_grid[1] - self.lon_grid[0])

def get_PEP(precur_arr, RV, df_splits, lags_i=np.array([1])):
    
    
    n_spl = df_splits.index.levels[0].size
    lags = np.array(lags_i, dtype=int)   

    precur_mck = precur_arr#find_region(precur_arr, 'Mckinnonplot')
    # make new xarray to store results
    PEP_xr = precur_mck.isel(time=0).drop('time').drop('mask').copy()
    # add lags
    list_xr = [PEP_xr.expand_dims('lag', axis=0) for i in range(lags.size)]
    PEP_xr = xr.concat(list_xr, dim = 'lag')
    PEP_xr['lag'] = ('lag', lags)
    # add train test split     
    list_xr = [PEP_xr.expand_dims('split', axis=0) for i in range(n_spl)]
    PEP_xr = xr.concat(list_xr, dim = 'split')           
    PEP_xr['split'] = ('split', range(n_spl))
    
    np_data = np.zeros_like(PEP_xr.values)
    
    

    for s in PEP_xr.split.values:
        progress = int(100 * (s+1) / n_spl)
        # =============================================================================
        # Split train test methods ['random'k'fold', 'leave_'k'_out', ', 'no_train_test_split']        
        # =============================================================================
        RV_mask     = df_splits.loc[s]['RV_mask']
        TrainIsTrue = df_splits.loc[s]['TrainIsTrue']
        RV_train = TrainIsTrue[np.logical_and(RV_mask, TrainIsTrue)].index
        RV_bin = RV.RV_bin.loc[RV_train]
        precur = precur_mck.sel(time=TrainIsTrue[TrainIsTrue].index)
#        dates_RV  = pd.to_datetime(RV_ts.time.values)
        n = RV_train.size ; r = int(100*n/RV.dates_RV.size )
        print(f"\rGetting PEP pattern {progress}%, trainsize=({n}dp, {r}%)", end="")
#        dates_all = pd.to_datetime(precur.time.values)
#        string_RV = list(dates_RV.strftime('%Y-%m-%d'))
#        string_full = list(dates_all.strftime('%Y-%m-%d'))
#        RV_period = [string_full.index(date) for date in string_full if date in string_RV]
        
        
        PEP_np = PEP_single_split(RV_bin, precur, lags_i)
        

        np_data[s] = PEP_np
    
    print("\n")
    PEP_xr.values = np_data
    
    mask_lon_PEP = np.logical_and(PEP_xr.longitude > 145, PEP_xr.longitude < 230); 
    mask_lat_PEP = np.logical_and(PEP_xr.latitude > 20, PEP_xr.latitude < 50); 
    PEP_mask = PEP_xr.where(mask_lon_PEP).where(mask_lat_PEP)
    PEP_xr['PEP_mask'] = PEP_mask
    
    return PEP_xr    


def PEP_single_split(RV_bin, precur, lags_i):
    lats = precur.latitude
    lons = precur.longitude
    np_data = np.zeros( (len(lags_i), len(lats), len(lons)) )   
    events = RV_bin[RV_bin==1].dropna().index
    for idx, lag in enumerate(lags_i):
        
        events_min_lag = func_dates_min_lag(events, lag)[1]
        np_data[idx] = precur.sel(time=events_min_lag).mean(dim='time')          

    return np_data


def get_spatcovs(dict_ds, df_split, s, lag, outdic_actors, normalize=True):
    #%%

    TrainIsTrue = df_split['TrainIsTrue']
    times = df_split.index
#    n_time = times.size
    options = [f'{int(lag)}..CPPAsv', f'{int(lag)}..PEPsv']

    data = np.zeros( (len(options), times.size) )
    df_sp_s = pd.DataFrame(data.T, index=times, columns=options)
    dates_train = TrainIsTrue[TrainIsTrue.values].index
#    dates_test  = TrainIsTrue[TrainIsTrue.values==False].index
    for var, actor in outdic_actors.items():
        ds = dict_ds[var]
        for i, select in enumerate(['sst', 'PEP']):
            # spat_cov over test years using corr fields from training
#            for col in options:
            if select == 'sst':
                col = f'{int(lag)}..CPPAsv'
                mask_name = 'prec_labels'
            elif select == 'PEP':
                col = f'{int(lag)}..PEPsv'
                mask_name = 'PEP_mask'

            lag = int(col.split('..')[0])
            full_timeserie = actor.precur_arr
            corr_vals = ds[select].sel(split=s).sel(lag=lag)
            mask = ds[select].sel(split=s).sel(lag=lag)[mask_name]
            pattern = corr_vals.where(~np.isnan(mask))
            if np.isnan(pattern.values).all():
                # no regions of this variable and split
                pass
            else:
                if normalize == True:
                    spatcov_full = find_precursors.calc_spatcov(full_timeserie, pattern)
                    mean = spatcov_full.sel(time=dates_train).mean(dim='time')
                    std = spatcov_full.sel(time=dates_train).std(dim='time')
                    spatcov_test = ((spatcov_full - mean) / std)
                elif normalize == False:
                    spatcov_test = find_precursors.calc_spatcov(full_timeserie, pattern)
                pd_sp = pd.Series(spatcov_test.values, index=times)
#                    col = options[i]
                df_sp_s[col] = pd_sp
                
    #%%
    return df_sp_s

# =============================================================================
# =============================================================================
# Core functions
# =============================================================================
# =============================================================================

def get_chunks(all_yrs_set, comp_years, perc_yrs_out=[5, 7.5, 10, 12.5, 15]):

    n_yrs = len(all_yrs_set)
    years_n_out = list(set([int(np.round(n_yrs*p/100., decimals=0)) for p in perc_yrs_out]))
    chunks = []
    for n_out in years_n_out:    
        chunks, count = create_chunks(all_yrs_set, n_out, chunks)
    
    mask_chunk = np.zeros( (len(chunks), len(comp_years)) , dtype=bool)
    for n, chnk in enumerate(chunks):
        mask_true_idx = [i for i in range(len(comp_years)) if comp_years[i] not in chnk] 
        mask_chunk[n][mask_true_idx] = True
        
    count = np.zeros( (len(all_yrs_set)) )
    for yr in all_yrs_set:
        idx = all_yrs_set.index(yr)
        count[idx] = np.sum( [chnk.count(yr) for chnk in chunks] )
        
    return mask_chunk


def extract_regs_p1(precur, mask_chunks, events_min_lag, dates_train_min_lag, 
                    std_train_lag, days_before=[0, 7, 14], FCP_thres=0.8,
                    SCM_percentile_thres=95):
    #%% 
#    T, pval, mask_sig = Welchs_t_test(sample, full, alpha=0.01)
#    threshold = np.reshape( mask_sig, (mask_sig.size) )
#    mask_threshold = threshold 
#    plt.figure()
#    plt.imshow(mask_sig)
    # divide train set into train-feature and train-weights part:
#    start = time.time()   
    
    lats = precur.latitude
    lons = precur.longitude    
    lsm  = np.isnan(precur[0])
    comp_train_stack = np.empty( (len(days_before), events_min_lag.size, lats.size* lons.size), dtype='int16')
    for i, d in enumerate(days_before):
        Prec_RV_train = precur.sel(time=dates_train_min_lag - pd.Timedelta(d, 'd'))
        comp_train_i = Prec_RV_train.sel(time=events_min_lag - pd.Timedelta(d, 'd'))
        comp_train_n = np.array((comp_train_i/std_train_lag)*1000, dtype='int16')
        comp_train_n[:,lsm] = 0
        
        comp_train_n = np.reshape(np.nan_to_num(comp_train_n), 
                              (events_min_lag.size,lats.size*lons.size))
        comp_train_stack[i] = comp_train_n
        
    assert ~any(np.isnan(comp_train_stack).flatten()), 'Nans in var {comp_train_stack}'
    
    import numba # conda install -c conda-forge numba=0.43.1
    
    
    def _make_composites(mask_chunks, comp_train_stack, iter_regions):
        
        
        for subset_i in range(comp_train_stack.shape[0]):
            comp_train_n = comp_train_stack[subset_i, :, :]
            for idx in range(mask_chunks.shape[0]):
#                comp_train_stack[subset_i, mask_chunks[idx], :]
                comp_subset = comp_train_n[mask_chunks[idx], :]
    
                sumcomp = np.zeros( comp_subset.shape[1] )
                for i in range(comp_subset.shape[0]):
                    sumcomp += comp_subset[i]
                mean = sumcomp / comp_subset.shape[0]
    
#                threshold = np.nanpercentile(mean, 95)
#                threshold = np.percentile(mean[~np.isnan(mean)], 95)
#                threshold = np.percentile(mean, 95)
                threshold = sorted(mean)[int(0.95 * mean.size)]
#                mean[np.isnan(mean)] = 0
                idx += subset_i * mask_chunks.shape[0]
#
                iter_regions[idx] = np.abs(mean) > ( threshold )

            
        return iter_regions

    jit_make_composites = numba.jit(nopython=True, parallel=True)(_make_composites)
    
    iter_regions = np.zeros( (comp_train_stack.shape[0]*len(mask_chunks), comp_train_stack[0,0].size), dtype='int8')
    iter_regions = jit_make_composites(mask_chunks, comp_train_stack, iter_regions)
    del jit_make_composites
    
#    iter_regions = make_composites(mask_chunks, comp_train_stack, iter_regions)
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(np.sum(iter_regions, axis=0), (lats.size, lons.size))) ; plt.colorbar()

    mask_final = ( np.sum(iter_regions, axis=0) < int(FCP_thres * iter_regions.shape[0]))
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(np.array(mask_final, dtype=int), (lats.size, lons.size))) ; plt.colorbar()
    weights = np.sum(iter_regions, axis=0)
    weights[mask_final==True] = 0.
    sum_count = np.reshape(weights, (lats.size, lons.size))
    weights = sum_count / np.max(sum_count)
    
    composite_p1 = precur.sel(time=events_min_lag).mean(dim='time', skipna=True)
    nparray_comp = np.reshape(np.nan_to_num(composite_p1.values), (composite_p1.size))
#    nparray_comp = np.nan_to_num(composite_p1.values)
    Comp_robust_lag = np.ma.MaskedArray(nparray_comp, mask=mask_final)
    #%%
    return Comp_robust_lag, weights

def create_chunks(all_yrs_set, n_out, chunks):
    #%%
    '''yr_prior are years which have priority to be part of the chunk '''
    # determine priority in random sampling
    # count what years are the least taken out of the composite
    def find_priority(chunks, all_yrs_set):
        count = np.zeros( (len(all_yrs_set)) )
        for yr in all_yrs_set:
            idx = all_yrs_set.index(yr)
            count[idx] = np.sum( [chnk.count(yr) for chnk in chunks] )
        list_idx_1 = list(np.argwhere(count == count.min()))
        list_idx_2 = list(np.argwhere(count != count.max()))
        
        all_yrs = all_yrs_set.copy()    
        yr_prior_1 = [all_yrs[int(i)] for i in list_idx_1]
        yr_prior_2 = [all_yrs[int(i)] for i in list_idx_2 if i not in list_idx_1]
        return yr_prior_1, yr_prior_2, count
    
    yr_prior_1, yr_prior_2, count = find_priority(chunks, all_yrs_set)
    
    
    for yr in all_yrs_set:    
        # yr is always going to be part of chunk
        if len(yr_prior_1) != 0 and yr in yr_prior_1:
            yr_prior_1.remove(yr)
        
        # resplenish every time half the years have passed
        if all_yrs_set.index(yr) < 0.5*len(all_yrs_set) or len(yr_prior_1) == 0:
            yr_prior_1, yr_prior_2 = find_priority(chunks, all_yrs_set)[:2]
            if len(yr_prior_1) != 0 and yr in yr_prior_1:
                yr_prior_1.remove(yr)
        
        years_to_add = [[yr]]
        n_choice = n_out - 1
        
        if len(yr_prior_1) >= n_choice:
            
            # yr of for loop iteration is always in list, 
            # give priority to yr_prior_1
            
            yrs_to_list  = list(np.random.choice(yr_prior_1, n_choice, replace=False))
            years_to_add.append(yrs_to_list)
#                years_to_add = flatten(years_to_add)
            # the year that is added has now reduced priority
            yr_prior_1 = [yr for yr in yr_prior_1 if yr not in yrs_to_list]
            if len(flatten(years_to_add)) != n_out:
#                print(yr_prior_1)
#                print(yr_prior_2)
#                print(years_to_add)
#                print('first')
                break
            
        elif len(yr_prior_1) < n_choice:# and len(yr_prior_1) != 0:
            
            yr_prior_1, yr_prior_2 = find_priority(chunks, all_yrs_set)[:2]
            if len(yr_prior_1) != 0 and yr in yr_prior_1:
                yr_prior_1.remove(yr)
            
            n_out_part = n_choice - len(yr_prior_1)
            
            # if there are still sufficient years left in yr_prior_1, just pick them
            if n_out_part < len(yr_prior_1) and n_out_part - len(yr_prior_1) >= 0:
                try:
                    yrs_to_list  = list(np.random.choice(yr_prior_1, n_choice, replace=False)) 
                except ValueError:
                    print(f"{yr_prior_1}\n")
                    print(f"{n_choice}")
                # replace n_out_part by n_choice, because n_out_part can be negative
                yrs_to_list  = list(np.random.choice(yr_prior_1, n_choice, replace=False)) 
            # if not, add what is left in yr_prior_1
            else:
                yrs_to_list = yr_prior_1
            years_to_add.append(yrs_to_list)
#                years_to_add = flatten(years_to_add)
            
            # the year that is added has now reduced priority
            yr_prior_1 = [yr for yr in yr_prior_1 if yr not in yrs_to_list]
            
            # what is left is sampled from yr_prior_2
            
            n_out_left = n_out - len(flatten(years_to_add))
            
            # ensure that there will be years in prior_2
            if len(yr_prior_2) < n_out_left and len(yr_prior_2) != 0:
                # add years that are left in yr_prior_2
                years_to_add.append(yr_prior_2)
                n_out_left = n_out - len(years_to_add)
#                    years_to_add = flatten(years_to_add)
            elif len(yr_prior_2) == 0 or n_out_left != 0:
                # create new yr_prior_2
                yr_prior_2 = [yr for yr in all_yrs_set.copy() if yr not in flatten(years_to_add)]
            
                yrs_to_list  = list(np.random.choice(yr_prior_2, n_out_left, replace=False))
                years_to_add.append( yrs_to_list )
#                    years_to_add = flatten(years_to_add)

            
            if len(flatten(years_to_add)) != n_out:
#                print('second')
#                print(yr_prior_1)
#                print(yr_prior_2)
#                print(n_out_left)
#                print(n_out)
#                print(years_to_add)
                break
            
        chunks.append( flatten(years_to_add) )
        
            
    yr_prior_1, yr_prior_2, count = find_priority(chunks, all_yrs_set)
    #%%
    return chunks, count


def make_datestr(dates, ex, startyr, endyr, lpyr=False):
    
    sstartdate = str(startyr) + '-' + ex['startperiod']
    senddate   = str(startyr) + '-' + ex['endperiod']
    
    # Find nearest matching date from dates (available)
    idx_start = abs(dates - pd.to_datetime(sstartdate)).argmin()
    sstartdate = dates[idx_start]
    idx_end   = abs(dates - pd.to_datetime(senddate)).argmin()
    senddate = dates[idx_end]
    
    start_yr = pd.date_range(start=sstartdate, end=senddate, 
                                freq=(dates[1] - dates[0]))
    if lpyr==True and calendar.isleap(startyr):
        start_yr -= pd.Timedelta( '1 days')
    else:
        pass
    breakyr = endyr
    datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
    nyears = (endyr - startyr)+1
    startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
    endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
    firstyear = startday[:4]
    def plusyearnoleap(curr_yr, startday, endday, incr):
        startday = startday.replace(firstyear, str(curr_yr+incr))
        endday = endday.replace(firstyear, str(curr_yr+incr))
        
        next_yr = pd.date_range(start=startday, end=endday, 
                        freq=(dates[1] - dates[0]))
        if lpyr==True and calendar.isleap(curr_yr+incr):
            next_yr -= pd.Timedelta( '1 days')
        elif lpyr == False:
            # excluding leap year again
            noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
            next_yr = next_yr[noleapdays].dropna(how='all')
        return next_yr
    

    for yr in range(0,nyears):
        curr_yr = yr+startyr
        next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
        nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
        datesstr = datesstr + nextstr

        if next_yr.year[0] == breakyr:
            break
    dates_period = pd.to_datetime(datesstr)
    return dates_period


def find_region(data, region='Pacific_US'):
    if region == 'Pacific_US':
        west_lon = -240; east_lon = -40; south_lat = -10; north_lat = 80

    elif region ==  'U.S.soil':
        west_lon = -130; east_lon = -60; south_lat = 0; north_lat = 60
    elif region ==  'U.S.cluster':
        west_lon = -100; east_lon = -70; south_lat = 20; north_lat = 50
    elif region ==  'Pacific':
        west_lon = -215; east_lon = -120; south_lat = 19; north_lat = 60
    elif region ==  'global':
        west_lon = -360; east_lon = -0.1; south_lat = -80; north_lat = 80
    elif region ==  'Northern':
        west_lon = -360; east_lon = -0.1; south_lat = -10; north_lat = 80
    elif region ==  'Southern':
        west_lon = -360; east_lon = -0.1; south_lat = -80; north_lat = -10
    elif region ==  'Tropics':
        west_lon = -360; east_lon = -0.1; south_lat = -15; north_lat = 30 
    elif region ==  'elnino3.4':
        west_lon = -170; east_lon = -120; south_lat = -5; north_lat = 5 
    elif region ==  'PEPrectangle':
        west_lon = -215; east_lon = -130; south_lat = 20; north_lat = 50
    if region == 'Mckinnonplot':
        west_lon = -240; east_lon = -40; south_lat = -10; north_lat = 80
    elif region ==  'PDO':
        west_lon = -250; east_lon = -110; south_lat = 20; north_lat = 70
        west_lon = -200; east_lon = -130; south_lat = 20; north_lat = 50
        # -250W (or 110E) tot -110W (or 250E)

    region_coords = [west_lon, east_lon, south_lat, north_lat]
    out = find_region_core(data, region_coords)
    
    return out 


def find_region_core(data, region_coords):
    
    west_lon, east_lon, south_lat, north_lat = region_coords
    import numpy as np
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)
#    if data.longitude.values[-1] > 180:
#        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
#        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
#        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)
        
    if west_lon <0 and east_lon > 0:
        # left_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon)))
        # right_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))
        # all_values = np.concatenate((np.reshape(left_of_meridional, (np.size(left_of_meridional))), np.reshape(right_of_meridional, np.size(right_of_meridional))))
        lon_idx = np.concatenate(( np.arange(find_nearest(data['longitude'], 360 + west_lon), len(data['longitude'])),
                              np.arange(0,find_nearest(data['longitude'], east_lon), 1) ))
        
        north_idx = find_nearest(data['latitude'],north_lat)
        south_idx = find_nearest(data['latitude'],south_lat)
        if north_idx > south_idx:
            lat_idx = np.arange(south_idx,north_idx,1)
            all_values = data.sel(latitude=slice(south_lat, north_lat), 
                                  longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
        elif south_idx > north_idx:
            lat_idx = np.arange(north_idx,south_idx,1)
            all_values = data.sel(latitude=slice(north_lat, south_lat), 
                                  longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
    if west_lon < 0 and east_lon < 0:
        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
        
        north_idx = find_nearest(data['latitude'],north_lat)
        south_idx = find_nearest(data['latitude'],south_lat)
        if north_idx > south_idx:
            lat_idx = np.arange(south_idx,north_idx,1)
            all_values = data.sel(latitude=slice(south_lat, north_lat), 
                                  longitude=slice(360+west_lon, 360+east_lon))
        elif south_idx > north_idx:
            lat_idx = np.arange(north_idx,south_idx,1)
            all_values = data.sel(latitude=slice(north_lat, south_lat), 
                                  longitude=slice(360+west_lon, 360+east_lon))     
        
#        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
#        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)

    return all_values


def xarray_plot(data, path='default', name = 'default', saving=False):
    #%%
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import numpy as np
    if type(data) == type(xr.Dataset()):
        data = data.to_array().squeeze()

    # some lon values > 180
    if len(data.longitude[np.where(data.longitude > 180)[0]]) != 0:
        # if 0 is in lon values
        if data.longitude.where(data.longitude==0).dropna(dim='longitude', how='all').size != 0.:
            print('hoi')   
            data = functions_pp.convert_longitude(data, 'only_east') 
    else:
        pass
    
    # get lonlat labels
    def evenly_spaced(ticks, steps=[5, 6, 7, 4, 3, 4]):
        def myround(x, base=5):
            return np.array(base * np.round(x/base,0), dtype=int)
        size = []
        tmp  = []
        for s in steps:
            labels = np.linspace(np.min(ticks), np.max(ticks), s, dtype=int)
            labels = np.array(sorted(list(set(myround(labels, 5)))))
            tmp.append([labels[l]-labels[l+1] for l in range(s-1)])
            size.append(np.unique([labels[l]-labels[l+1] for l in range(s-1)]).size)
            if size[-1]==1:
                break
        return labels

    longitude_labels = evenly_spaced(data.longitude.values)
    latitude_labels = evenly_spaced(data.latitude.values, steps=[4,3,5])
    
    
    if data.ndim != 2:
        print("number of dimension is {}, printing first element of first dimension".format(np.squeeze(data).ndim))
        data = data[0]
    else:
        pass
    if 'mask' in list(data.coords.keys()):
        lons = data.where(data.mask==True, drop=True).longitude
        lats = data.where(data.mask==True, drop=True).latitude
        cen_lon = int(lons.mean())
        data = data.where(data.mask==True, drop=True)
    else:
        lons = data.longitude
        lats = data.latitude
        cen_lon = (lons.mean())
#    data = data.sortby(lons)
    
    
    fig = plt.figure( figsize=(15,11) ) 
    proj = ccrs.PlateCarree(central_longitude=cen_lon)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(color='black', alpha=0.3, facecolor='grey')
    ax.add_feature(cfeature.LAND, facecolor='grey', alpha=0.3)
#    ax.tick_params(axis='both', labelsize='small')
#    ax.set_xlabel([str(el) for el in longitude_labels])
    if proj.proj4_params['proj'] in ['merc', 'eqc']:
        
        ax.set_xticks(longitude_labels[:], crs=ccrs.PlateCarree())
        ax.set_xticklabels(longitude_labels[:], fontsize=12)
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        
        ax.set_yticks(latitude_labels, crs=ccrs.PlateCarree())
        ax.set_yticklabels(latitude_labels, fontsize=12)
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
    
    vmin = np.round(float(data.min())-0.01,decimals=2) 
    vmax = np.round(float(data.max())+0.01,decimals=2) 
    vmin = -max(abs(vmin),vmax) ; vmax = max(abs(vmin),vmax)
    ax.set_extent([lons[0],lons[-1], lats[0], lats[-1]], ccrs.PlateCarree())

    data.attrs['long_name'] = ''
    if 'mask' in list(data.coords.keys()):
        for_plt = data.copy().where(data.mask==True)
        plot = for_plt.plot.pcolormesh('longitude', 'latitude', ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True,
                             vmin=vmin, vmax=vmax, subplot_kws={'projection': proj},
                             cbar_kwargs={'orientation' : 'horizontal', 
                                          'fraction':0.10,
                                          'pad':0.05})
    else:
        plot = data.plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True,
                             vmin=vmin, vmax=vmax, 
                             cbar_kwargs={'orientation' : 'horizontal'})
    plot.colorbar.ax.set_label('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    #%%
    if saving == True:
        save_figure(fig, path=path, name=name)
    plt.show()

def save_figure(fig, path, name=''):

    now = datetime.datetime.now().strftime('%y-%m-%d_%H:%M')
    if path == 'default':
        path = '/Users/semvijverberg/Downloads'
    else:
        path = path
    import datetime
    today = datetime.datetime.today().strftime("%d-%m-%y_%H'%M")
    if name == '':
        name = now
    if name != '':
        print('input name is: {}'.format(name))
        name = name + '.png'
        pass
    else:
        name = 'fig_' + today + '.png'
    print(('{} to path {}'.format(name, path)))
    fig.savefig(os.path.join(path,name), format='png', dpi=300, bbox_inches='tight')
    return fig

def plotting_wrapper(plotarr, ex, filename=None,  kwrgs=None, map_proj=None):
#    map_proj = ccrs.Miller(central_longitude=240) 
    try:
        folder_name = ex['path_fig']
    except:
        folder_name = '/Users/semvijverberg/Downloads'
    if os.path.isdir(folder_name) != True : 
        os.makedirs(folder_name)

    if kwrgs == None:
        kwrgs = dict( {'title' : plotarr.name, 'clevels' : 'notdefault', 'steps':17,
                        'vmin' : -3*plotarr.std().values, 'vmax' : 3*plotarr.std().values, 
                       'cmap' : plt.cm.RdBu_r, 'column' : 1, 'subtitles' : None,
                       'style_colormap' : 'pcolormesh'} )
    else:
        kwrgs = kwrgs
        if 'title' not in kwrgs.keys():
            kwrgs['title'] = plotarr.attrs['title']
        if 'style_colormap' not in kwrgs.keys():
            kwrgs['style_colormap'] = 'pcolormesh'
            
        
    if filename != None:
        file_name = os.path.join(folder_name, filename)
        kwrgs['savefig'] = True
    else:
        kwrgs['savefig'] = False
        file_name = 'Users/semvijverberg/Downloads/test.png'
    finalfigure(plotarr, file_name, kwrgs, map_proj=map_proj)
    

def finalfigure(xrdata, file_name, kwrgs, map_proj=None):
    #%%
    if map_proj is None:
        map_proj = ccrs.PlateCarree(central_longitude=220)  

    lons = xrdata.longitude.values
    lats = xrdata.latitude.values
    strvars = [' {} '.format(var) for var in list(xrdata.dims)]
    
    var = [var for var in strvars if var not in ' longitude latitude '][0] 
    var = var.replace(' ', '')
    g = xr.plot.FacetGrid(xrdata, col=var, col_wrap=kwrgs['column'], sharex=True,
                      sharey=True, subplot_kws={'projection': map_proj},
                      aspect= (xrdata.longitude.size) / xrdata.latitude.size, size=3)
    figwidth = g.fig.get_figwidth() ; figheight = g.fig.get_figheight()

    lon_tick = xrdata.longitude.values
    dg = abs(lon_tick[1] - lon_tick[0])
    periodic = (np.arange(0, 360, dg).size - lon_tick.size) < 1
    
    longitude_labels = np.linspace(np.min(lon_tick), np.max(lon_tick), 6, dtype=int)
    longitude_labels = np.array(sorted(list(set(np.round(longitude_labels, -1)))))

#    longitude_labels = np.concatenate([ longitude_labels, [longitude_labels[-1]], [180]])
#    longitude_labels = [-150,  -70,    0,   70,  140, 140]
    latitude_labels = np.linspace(xrdata.latitude.min(), xrdata.latitude.max(), 4, dtype=int)
    latitude_labels = sorted(list(set(np.round(latitude_labels, -1))))
    
    g.set_ticks(max_xticks=5, max_yticks=5, fontsize='small')
    g.set_xlabels(label=[str(el) for el in longitude_labels])

    
    if kwrgs['clevels'] == 'default':
        vmin = np.round(float(xrdata.min())-0.01,decimals=2) ; vmax = np.round(float(xrdata.max())+0.01,decimals=2)
        clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    else:
        vmin=kwrgs['vmin']
        vmax=kwrgs['vmax']
        
        clevels = np.linspace(vmin, vmax,kwrgs['steps'])

    cmap_ = kwrgs['cmap']
    
    if 'clim' in kwrgs.keys(): #adjust the range of colors shown in cbar
        cnorm = np.linspace(kwrgs['clim'][0],kwrgs['clim'][1],11)
        vmin = kwrgs['clim'][0]
    else:
        cnorm = clevels
        

    norm = mpl.colors.BoundaryNorm(boundaries=cnorm, ncolors=256)
    subplot_kws = {'projection': map_proj}
    
    n_plots = xrdata[var].size
    for n_ax in np.arange(0,n_plots):
        ax = g.axes.flatten()[n_ax]
#        print(n_ax)
        if periodic == True:
            plotdata = plot_maps.extend_longitude(xrdata[n_ax])
        else:
            plotdata = xrdata[n_ax].squeeze()
        if kwrgs['style_colormap'] == 'pcolormesh':
            im = plotdata.plot.pcolormesh(ax=ax, cmap=cmap_,
                               transform=ccrs.PlateCarree(),
                               subplot_kws=subplot_kws,
                                levels=clevels, add_colorbar=False)
        
        if kwrgs['style_colormap'] == 'contourf':
            im = plotdata.plot.contourf(ax=ax, cmap=cmap_,
                               transform=ccrs.PlateCarree(),
                               subplot_kws=subplot_kws,
                                levels=clevels, add_colorbar=False)

        if 'sign_stipling' in kwrgs.keys():
            if kwrgs['sign_stipling'][0] == 'colorplot':
                sigdata = kwrgs['sign_stipling'][1]
                if periodic == True:
                    sigdata = plot_maps.extend_longitude(sigdata[n_ax])
                else:
                    sigdata = sigdata[n_ax].squeeze()
                sigdata.plot.contourf(ax=ax, levels=[0, 0.5, 1],
                           transform=ccrs.PlateCarree(), hatches=['...', ''],
                           colors='none', add_colorbar=False,
                           subplot_kws={'projection': map_proj})
            
        ax.coastlines(color='black', alpha=0.3, facecolor='grey', linewidth=2)
        ax.add_feature(cfeature.LAND, facecolor='grey', alpha=0.3)
        
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], ccrs.PlateCarree())
        
        if 'contours' in kwrgs.keys():
            condata, con_levels = kwrgs['contours']
            if periodic == True:
                condata = plot_maps.extend_longitude(condata[n_ax])
            else:
                condata = condata[n_ax].squeeze()
            condata.plot.contour(ax=ax, add_colorbar=False,
                               transform=ccrs.PlateCarree(),
                               subplot_kws={'projection': map_proj},
                                levels=con_levels, cmap=cmap_)
            if 'sign_stipling' in kwrgs.keys():
                if kwrgs['sign_stipling'][0] == 'contour':
                    sigdata = kwrgs['sign_stipling'][1]
                    if periodic == True:
                        sigdata = plot_maps.extend_longitude(sigdata[n_ax])
                    else:
                        sigdata = sigdata[n_ax].squeeze()
                    sigdata.plot.contourf(ax=ax, levels=[0, 0.5, 1],
                               transform=ccrs.PlateCarree(), hatches=['...', ''],
                               colors='none', add_colorbar=False,
                               subplot_kws={'projection': map_proj})
                                
                  
        
        if kwrgs['subtitles'] is None:
            pass
        else:
            fontdict = dict({'fontsize'     : 18,
                             'fontweight'   : 'bold'})
            row = int(np.where(g.axes == ax)[0])
            col = int(np.where(g.axes == ax)[1])
            ax.set_title(kwrgs['subtitles'][row,col], fontdict=fontdict, loc='center')
        
        if 'drawbox' in kwrgs.keys():
            
            def get_ring(coords):
                '''tuple in format: west_lon, east_lon, south_lat, north_lat '''
                west_lon, east_lon, south_lat, north_lat = coords
                lons_sq = [west_lon, west_lon, east_lon, east_lon]
                lats_sq = [north_lat, south_lat, south_lat, north_lat]
                ring = LinearRing(list(zip(lons_sq , lats_sq )))
                return ring
            ring = get_ring(kwrgs['drawbox'][1])
#            lons_sq = [-215, -215, -130, -130] #[-215, -215, -125, -125] #[-215, -215, -130, -130] 
#            lats_sq = [50, 20, 20, 50]
            if kwrgs['drawbox'][0] == n_ax or kwrgs['drawbox'][0] == 'all':
                ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='green',
                              linewidth=3.5)
        
        if 'ax_text' in kwrgs.keys():
            ax.text(0.0, 1.01, kwrgs['ax_text'][n_ax],
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=15)
            
        if map_proj.proj4_params['proj'] in ['merc', 'eqc', 'cea']:
#            print(True)
            ax.set_xticks(longitude_labels[:-1], crs=ccrs.PlateCarree())
            ax.set_xticklabels(longitude_labels[:-1], fontsize=12)
            lon_formatter = cticker.LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            
            ax.set_yticks(latitude_labels, crs=ccrs.PlateCarree())
            ax.set_yticklabels(latitude_labels, fontsize=12)
            lat_formatter = cticker.LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            
        else:
            pass
        
    if 'title_h' in kwrgs.keys():
        title_height = kwrgs['title_h']
    else:
        title_height = 0.98
    g.fig.text(0.5, title_height, kwrgs['title'], fontsize=20,
               fontweight='heavy', transform=g.fig.transFigure,
               horizontalalignment='center',verticalalignment='top')
    
    if 'adj_fig_h' in kwrgs.keys():
        g.fig.set_figheight(figheight*kwrgs['adj_fig_h'], forward=True)
    if 'adj_fig_w' in kwrgs.keys():
        g.fig.set_figwidth(figwidth*kwrgs['adj_fig_w'], forward=True)

    if 'cbar_vert' in kwrgs.keys():
        cbar_vert = (figheight/40)/(n_plots*2) + kwrgs['cbar_vert']
    else:
        cbar_vert = (figheight/40)/(n_plots*2)
    if 'cbar_hght' in kwrgs.keys():
        cbar_hght = (figheight/40)/(n_plots*2) + kwrgs['cbar_hght']
    else:
        cbar_hght = (figheight/40)/(n_plots*2)
    if 'wspace' in kwrgs.keys():
        g.fig.subplots_adjust(wspace=kwrgs['wspace'])
    if 'hspace' in kwrgs.keys():
        g.fig.subplots_adjust(hspace=kwrgs['hspace'])
    if 'extend' in kwrgs.keys():
        if kwrgs['extend'] != None:
            extend = kwrgs['extend'][0]
        else:
            extend = 'neither'
    else:
        extend = 'neither'

    cbar_ax = g.fig.add_axes([0.25, cbar_vert, 
                                  0.5, cbar_hght], label='cbar')


#    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, orientation='horizontal', 
#                 extend=extend, ticks=cnorm, norm=norm)


    cbar = plt.colorbar(im, cbar_ax, cmap=cmap_, norm=norm,
                    orientation='horizontal', 
                    extend=extend)

    if 'cticks_center' in kwrgs.keys():
        cbar.set_ticks(clevels + 0.5)
        ticklabels = np.array(clevels+1, dtype=int)
        cbar.set_ticklabels(ticklabels, update_ticks=True)
        cbar.update_ticks()
    
    if 'extend' in kwrgs.keys():
        if kwrgs['extend'][0] == 'min':
            if type(kwrgs['extend'][1]) == type(str()):
                cbar.cmap.set_under(kwrgs['extend'][1])
            else:
                cbar.cmap.set_under(cbar.to_rgba(kwrgs['extend'][1]))
    cbar.set_label(xrdata.attrs['units'], fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    #%%
    return 

def func_dates_min_lag(dates, lag):
    dates_min_lag = pd.to_datetime(dates.values) - pd.Timedelta(int(lag), unit='d')
    ### exlude leap days from dates_train_min_lag ###


    # ensure that everything before the leap day is shifted one day back in time 
    # years with leapdays now have a day less, thus everything before
    # the leapday should be extended back in time by 1 day.
    mask_lpyrfeb = np.logical_and(dates_min_lag.month == 2, 
                                         dates_min_lag.is_leap_year
                                         )
    mask_lpyrjan = np.logical_and(dates_min_lag.month == 1, 
                                         dates_min_lag.is_leap_year
                                         )
    mask_ = np.logical_or(mask_lpyrfeb, mask_lpyrjan)
    new_dates = np.array(dates_min_lag)
    new_dates[mask_] = dates_min_lag[mask_] - pd.Timedelta(1, unit='d')
    dates_min_lag = pd.to_datetime(new_dates)   
    # to be able to select date in pandas dataframe
    dates_min_lag_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates_min_lag]                                         
    return dates_min_lag_str, dates_min_lag



#
#def store_timeseries(ds_Sem, RV_ts, Prec_reg, ex):
#    #%%
#    ts_3d    = Prec_reg
#    # mean of El nino 3.4
#    ts_3d_nino = find_region(Prec_reg, region='elnino3.4')[0]
#    # get lonlat array of area for taking spatial means 
#    nino_index = area_weighted(ts_3d_nino).mean(dim=('latitude', 'longitude'), skipna=True).values
#    dates = pd.to_datetime(ts_3d.time.values)
#    dates -= pd.Timedelta(dates.hour[0], unit='h')
#    df_nino = pd.DataFrame(data = nino_index[:,None], index=dates, columns=['nino3.4']) 
#    df_nino['nino3.4rm5'] = df_nino['nino3.4'].rolling(int((365/12)*5), min_periods=1).mean()
#    for lag in ex['lags']:
#        idx = ex['lags'].index(lag)
#
#
#        mask_regions = np.nan_to_num(ds_Sem['pat_num_CPPA_clust'].sel(lag=lag).values) >= 1
#        # Make time series for whole period
#        
#        mask_notnan = (np.product(np.isnan(ts_3d.values),axis=0)==False) # nans == False
#        mask = mask_notnan * mask_regions
#        ts_3d_mask     = ts_3d.where(mask==True)
#        # ts_3d is given more weight to robust precursor regions
#        ts_3d_w  = ts_3d_mask  * ds_Sem['weights'].sel(lag=lag)
#        # ts_3d_w is normalized w.r.t. std in RV dates min lag
#        ts_3d_nw = ts_3d_w / ds_Sem['std_train_min_lag'][idx]
#        # same is done for pattern
#        pattern_CPPA = ds_Sem['pattern_CPPA'].sel(lag=lag)
#        CPPA_w = pattern_CPPA #* ds_Sem['weights'].sel(lag=lag)
#        CPPA_nw = CPPA_w / ds_Sem['std_train_min_lag'][idx]
#        
#        
#        # regions for time series
#        Regions_lag_i = ds_Sem['pat_num_CPPA_clust'][idx].squeeze().values
#        regions_for_ts = np.unique(Regions_lag_i[~np.isnan(Regions_lag_i)])
#        # spatial mean (normalized & weighted)
#        ts_regions_lag_i, sign_ts_regions = spatial_mean_regions(Regions_lag_i, 
#                                regions_for_ts, ts_3d_w, CPPA_w.values)[:2]
#        
#        
#        check_nans = np.where(np.isnan(ts_regions_lag_i))
#        if check_nans[0].size != 0:
#            print('{} nans found in time series of region {}, dropping this region.'.format(
#                    check_nans[0].size, 
#                    np.unique(regions_for_ts[check_nans[1]])))
#            regions_for_ts = np.delete(regions_for_ts, check_nans[1])
#            ts_regions_lag_i = np.delete(ts_regions_lag_i, check_nans[1], axis=1)
#            sign_ts_regions  = np.delete(sign_ts_regions, check_nans[1], axis=0)
#        
#        
#        # spatial covariance of whole CPPA pattern
#        spatcov_CPPA = cross_correlation_patterns(ts_3d_w, pattern_CPPA)
#
#        
#        # merge data
#        columns = list(np.array(regions_for_ts, dtype=int))
#        columns.insert(0, 'spatcov_CPPA')
#
#               
#        data = np.concatenate([spatcov_CPPA.values[:,None],
#                               ts_regions_lag_i], axis=1)
#        data = np.array(data, dtype='float16')
#        dates = pd.to_datetime(ts_3d.time.values)
#        dates -= pd.Timedelta(dates.hour[0], unit='h')
#        df_CPPA = pd.DataFrame(data = data, index=dates, columns=columns) 
#        
#        
#        df = pd.concat([df_nino, df_CPPA], axis=1)
#        df.index.name = 'date'
#        
#        name_trainset = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
#        df.to_csv(os.path.join(ex['output_ts_folder'], name_trainset ), 
#                  float_format='%.5f', chunksize=int(dates.size/3))
#        #%%
#    return
#
#



def Welchs_t_test(sample, full, min_alpha=0.05, fieldsig=True):
    '''mask returned is True where values are non-significant'''
    from statsmodels.sandbox.stats import multicomp
    import scipy
#    np.warnings.filterwarnings('ignore')
    mask = (sample[0] == 0.).values
    nanmaskspace = np.isnan(sample.values[0])
#    mask = np.reshape(mask, (mask.size))
    n_space = full.latitude.size*full.longitude.size
    npfull = np.reshape(full.values, (full.time.size, n_space))
    npsample = np.reshape(sample.values, (sample.time.size, n_space))
    
    T, pval = scipy.stats.ttest_ind(npsample, npfull, axis=0, 
                                equal_var=False, nan_policy='propagate')
    
    
    if fieldsig == True:
        pval_fdr = pval.copy()
        pval_fdr = pval[~np.isnan(pval)]
        alpha_fdr = 2*min_alpha
        pval_fdr = multicomp.multipletests(pval_fdr, alpha=alpha_fdr, method='fdr_bh')[1]
        pval[~np.isnan(pval)] = pval_fdr
        
    pval = np.reshape(pval, (full.latitude.size, full.longitude.size))
    
    T = np.reshape(T, (full.latitude.size, full.longitude.size))
    mask_sig = (pval > min_alpha) 
    mask_sig[mask] = True
    mask_sig[nanmaskspace] = True
#    plt.imshow(mask_sig)
    return T, pval, mask_sig





#
#def import_array(filename, ex):
#    
#    ds = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)
#    variables = list(ds.variables.keys())
#    strvars = [' {} '.format(var) for var in variables]
#    common_fields = ' time time_bnds longitude latitude lev lon lat level mask '
#    var = [var for var in strvars if var not in common_fields][0]
#    var = var.replace(' ', '')
#    
#    ds = ds[var].squeeze()
#    if 'time' in ds.dims:
#        numtime = ds['time']
#        dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
#        dates = pd.to_datetime(dates)
#        dates -= pd.Timedelta(dates.hour[0], unit='h')
#        
#        ds['time'] = dates
#    return ds

#    
#def area_weighted(xarray):
#    # Area weighted, taking cos of latitude in radians     
#    coslat = np.cos(np.deg2rad(xarray.coords['latitude'].values)).clip(0., 1.)
#    area_weights = np.tile(coslat[..., np.newaxis],(1,xarray.longitude.size))
##    xarray.values = xarray.values * area_weights 
#
#    return xr.DataArray(xarray.values * area_weights, coords=xarray.coords, 
#                           dims=xarray.dims)
#    

#def rolling_mean_xr(xarray, win):
#    closed = int(win/2)
#    flatarray = xarray.values.flatten()
#    ext_array = np.insert(flatarray, 0, flatarray[-closed:])
#    ext_array = np.insert(ext_array, 0, flatarray[:closed])
#    
#    df = pd.DataFrame(ext_array)
##    std = xarray.where(xarray.values!=0.).std().values
##    scipy.signal.gaussian(win, std)
#    rollmean = df.rolling(win, center=True, 
#                          win_type='gaussian').mean(std=win/2.).dropna()
#    
#    # replace values with smoothened values
#    new_xarray = xarray.copy()
#    new_values = np.reshape(rollmean.squeeze().values, xarray.shape)
#    # ensure LSM mask
#    mask = np.array((xarray.values!=0.),dtype=int)
#    new_xarray.values = (new_values * mask)
#
#    return new_xarray
#
#def rolling_mean_time(xarray_or_file, ex, center=True):
#    #%%
##    xarray_or_file = Prec_reg
##    array = np.zeros(60)
##    array[-30:] = 1
##    xarray_or_file = xr.DataArray(array, coords=[np.arange(60)], dims=['time'])
#    
#    if type(xarray_or_file) == str:
#        file_path = os.path.join(ex['path_pp'], xarray_or_file)        
#        ds = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
#        ds_rollingmean = ds.rolling(time=ex['rollingmean'][1], center=center, min_periods=1).mean()
#        
#        new_fname = 'rm{}_'.format(ex['rollingmean'][1]) + xarray_or_file
#        file_path = os.path.join(ex['path_pp'], new_fname)
#        ds_rollingmean.to_netcdf(file_path, mode='w')
#        print('saved netcdf as {}'.format(new_fname))
#        print('functions returning None')
#        xr_rolling_mean = None
#    else:
#        # Taking rolling window mean with the closing on the right, 
#        # meaning that we are taking the mean over the past at the index/label 
#
#        if xarray_or_file.ndim == 1:
#            xr_rolling_mean = xarray_or_file.rolling(time=ex['rollingmean'][1], center=True, 
#                                                         min_periods=1).mean()
#        else: 
#            xr_rolling_mean = xarray_or_file.rolling(time=ex['rollingmean'][1], center=True, 
#                                                         min_periods=1).mean(dim='time', skipna=True)
#        if 'latitude' in xarray_or_file.dims:
#            lat = 35
#            lon = 360-20
#            
#            def find_nearest(array, value):
#                idx = (np.abs(array - value)).argmin()
#                return int(idx)
#            
#            lat_idx = find_nearest(xarray_or_file['latitude'], lat)
#            lon_idx = find_nearest(xarray_or_file['longitude'], lon)
#            
#            
#            singlegc = xarray_or_file.isel(latitude=lat_idx, 
#                                          longitude=lon_idx) 
#        else:
#            singlegc = xarray_or_file
#
#        if type(singlegc) == type(xr.Dataset()):
#            singlegc = singlegc.to_array().squeeze()
#        
#        year = 2012
#        singlegc_oneyr = singlegc.where(singlegc.time.dt.year == year).dropna(dim='time', how='all')
#        dates = pd.to_datetime(singlegc_oneyr.time.values)
#        plt.figure(figsize=(10,6))
#        plt.plot(dates, singlegc_oneyr.squeeze())
#        if 'latitude' in xarray_or_file.dims:
#            singlegc = xr_rolling_mean.isel(latitude=lat_idx, 
#                                          longitude=lon_idx) 
#        else:
#            singlegc = xr_rolling_mean
#        if type(singlegc) == type(xr.Dataset()):
#            singlegc = singlegc.to_array().squeeze()
#        singlegc_oneyr = singlegc.where(singlegc.time.dt.year == year).dropna(dim='time', how='all')
#        plt.plot(dates, singlegc_oneyr.squeeze())
#    #%%
#    return xr_rolling_mean
#
#



#def kornshell_with_input(args, ex):
##    stopped working for cdo commands
#    '''some kornshell with input '''
##    args = [anom]
#    import os
#    import subprocess
##    cwd = os.getcwd()
#    # Writing the bash script:
#    new_bash_script = os.path.join('/Users/semvijverberg/surfdrive/Scripts/CPPA/CPPA/', "bash_script.sh")
##    arg_5d_mean = 'cdo timselmean,5 {} {}'.format(infile, outfile)
#    #arg1 = 'ncea -d latitude,59.0,84.0 -d longitude,-95,-10 {} {}'.format(infile, outfile)
#    
#    bash_and_args = [new_bash_script]
#    [bash_and_args.append(arg) for arg in args]
#    with open(new_bash_script, "w") as file:
#        file.write("#!/bin/sh\n")
#        file.write("echo bash script output\n")
#        for cmd in range(len(args)):
#
#            print(args[cmd])
#            file.write("${}\n".format(cmd+1)) 
#    p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
#                         stderr=subprocess.STDOUT)
#                         
#    out = p.communicate()
#    print(out[0].decode())
#    return
#
#
#def grouping_regions_similar_coords(l_ds, ex, grouping = 'group_accros_tests_single_lag', eps=10):
#    '''Regions with similar coordinates are grouped together.
#    lower eps indicate higher density necessary to form a cluster. 
#    If eps too high, no high density is required and originally similar regions 
#    are assigned into different clusters. If eps is too low, such an high density is required that
#    it will also start to cluster together regions with similar coordinates.
#    '''
#    #%%
##    if ex['n_conv'] < 30:
##        grouping = 'group_across_test_and_lags'
##    grouping = 'group_accros_tests_single_lag'
##    grouping =  'group_across_test_and_lags'
#    # Precursor Regions Dimensions
#    all_lags_in_exp = l_ds[0]['pat_num_CPPA'].sel(lag=ex['lags']).lag.values
#    lags_ind        = np.reshape(np.argwhere(all_lags_in_exp == ex['lags']), -1)
#    PRECURSOR_DATA = np.array([data['pat_num_CPPA'].values for data in l_ds])
#    PRECURSOR_VALUES = np.array([data['pattern_CPPA'].values for data in l_ds])
#    PRECURSOR_DATA = PRECURSOR_DATA[:,lags_ind]
#    PRECURSOR_LONGITIUDE = l_ds[0]['pat_num_CPPA'].longitude.values
#    PRECURSOR_LATITUDE = l_ds[0]['pat_num_CPPA'].latitude.values
#    PRECURSOR_LAGS = l_ds[0]['pat_num_CPPA'].sel(lag=ex['lags']).lag.values
#    PRECURSOR_N_TEST_SETS = ex['n_conv']
#    
#    # Precursor Grid (to calculate precursor region centre coordinate)
#    PRECURSOR_GRID = np.zeros((len(PRECURSOR_LATITUDE), len(PRECURSOR_LONGITIUDE), 2))
#    PRECURSOR_GRID[..., 0], PRECURSOR_GRID[..., 1] = np.meshgrid(PRECURSOR_LONGITIUDE, PRECURSOR_LATITUDE)
#    
#
#    precursor_coordinates = []
#    
#    # Array Containing Precursor Region Indices for each YEAR 
#    precursor_indices = np.empty((PRECURSOR_N_TEST_SETS,
#                                  len(PRECURSOR_LAGS),
#                                  len(PRECURSOR_LATITUDE),
#                                  len(PRECURSOR_LONGITIUDE)),
#                                 np.float32)
#    precursor_indices[:,:,:,:] = np.nan
#    precursor_indices_new = precursor_indices.copy()
#    
#    ex['uniq_regs_lag'] = np.zeros(len(PRECURSOR_LAGS))
#    # Array Containing Precursor Region Weights for each YEAR and LAG
#    for lag_idx, lag in enumerate(PRECURSOR_LAGS):
#        
#        indices_across_yrs = np.squeeze(PRECURSOR_DATA[:,lag_idx,:,:])
#        
#        
#        if grouping == 'group_accros_tests_single_lag':
#            # regions are given same number across test set, not accross all lags
#            precursor_coordinates = []
#    
#    #    precursor_weights = np.zeros_like(precursor_indices, np.float32)
#        min_samples = []
#        for test_idx in range(PRECURSOR_N_TEST_SETS):
#            indices = indices_across_yrs[test_idx, :, :]
#            min_samples.append( np.nanmax(indices) )
#            for region_idx in np.unique(indices[~np.isnan(indices)]):
#            # evaluate regions
#    #            plt.figure()
#    #            plt.imshow(indices == region_idx)
#                region = precursor_indices[test_idx, lag_idx, indices == region_idx]
#                size_reg = region.size
#                sign_reg = np.sign(PRECURSOR_VALUES[test_idx, lag_idx, indices == region_idx])
#                precursor_indices[test_idx, lag_idx, indices == region_idx] = int(region_idx)
#                if sign_reg.mean() == 1:
#                    lon_lat = PRECURSOR_GRID[indices == region_idx].mean(0)
#                elif sign_reg.mean() == -1:
#                    lon_lat = PRECURSOR_GRID[indices == region_idx].mean(0) * -1
#                if np.isnan(lon_lat).any()==False:
#
#                    precursor_coordinates.append((
#                        [test_idx, lag_idx, int(region_idx)], lon_lat, size_reg))
#    
#        # Group Similar Precursor Regions Together across years for same lag
#        precursor_coordinates_index = np.array([index for index, coord, size in precursor_coordinates])
#        precursor_coordinates_coord = np.array([coord for index, coord, size in precursor_coordinates])
#        precursor_coordinates_weight = np.array([size for index, coord, size in precursor_coordinates])
#        
#        if grouping == 'group_accros_tests_single_lag':
#            # min_samples to form core cluster, lower eps indicate higher density necessary to form a cluster.
#            min_s = min_s = ex['n_conv'] * len(ex['lags']) / 2 #np.nanmin(min_samples)
#            precursor_coordinates_group = DBSCAN(min_samples=ex['n_conv'] * 0.4, eps=eps).fit_predict(
#                    precursor_coordinates_coord, sample_weight = precursor_coordinates_weight) + 2
#            
#            for (year_idx, lag_idx, region_idx), group in zip(precursor_coordinates_index, precursor_coordinates_group):
#                precursor_indices_new[year_idx, lag_idx, precursor_indices[year_idx, lag_idx] == region_idx] = group
#
#    
#    if grouping == 'group_across_test_and_lags':
#        # Group Similar Precursor Regions Together
#        min_s = ex['n_conv'] * len(ex['lags']) / 2#np.nanmax(PRECURSOR_DATA)
#        precursor_coordinates_index = np.array([index for index, coord, size in precursor_coordinates])
#        precursor_coordinates_coord = np.array([coord for index, coord, size in precursor_coordinates])
#        precursor_coordinates_weight = np.array([size for index, coord, size in precursor_coordinates])
#
#        precursor_coordinates_group = DBSCAN(min_samples=min_s, eps=eps).fit_predict(
#                precursor_coordinates_coord, sample_weight = precursor_coordinates_weight) + 2
#        
#        
#        precursor_indices_new = np.zeros_like(precursor_indices)
#        for (year_idx, lag_idx, region_idx), group in zip(precursor_coordinates_index, precursor_coordinates_group):
#    #        print(year_idx, lag_idx, region_idx, group)
#            precursor_indices_new[year_idx, lag_idx, precursor_indices[year_idx, lag_idx] == region_idx] = group
#        precursor_indices_new[precursor_indices_new==0.] = np.nan
#        
#    # couting groups
#    counting = {}
#    for r in precursor_coordinates_group:
#        c = list(precursor_coordinates_group).count(r)
#        counting[r] =c
#    # sort by counts:
#    order_count = dict(sorted(counting.items(), key = 
#             lambda kv:(kv[1], kv[0]), reverse=True))
#    
#    precursor_indices_new_ord = np.zeros_like(precursor_indices)
##    if grouping == 'group_across_test_and_lags':
#    for i, r in enumerate(order_count.keys()):
#        precursor_indices_new_ord[precursor_indices_new==r] = i+1
#    precursor_indices_new_ord[precursor_indices_new_ord==0.] = np.nan
#    # replace values in PRECURSOR_DATA|
#    PRECURSOR_DATA[:,:,:,:] = precursor_indices_new_ord[:,:,:,:]
##    else:
##        PRECURSOR_DATA[:,:,:,:] = precursor_indices_new
#    
#    ex['uniq_regs_lag'][lag_idx] = max(np.unique(precursor_indices_new_ord) )
#    
#    
#    l_ds_new = []
#    for test_idx in range(PRECURSOR_N_TEST_SETS):
#        single_ds = l_ds[test_idx].copy()
#        pattern   = single_ds['pat_num_CPPA'].copy()
#        
#        pattern.values = PRECURSOR_DATA[test_idx]
##        # set the rest to nan
##        pattern = pattern.where(pattern.values != 0.)
#        single_ds['pat_num_CPPA_clust'] = pattern
#    #    print(test_idx)
#    #    plt.figure()
#    #    single_ds['pat_num_CPPA'][0].plot()
#        # overwrite ds
#        l_ds_new.append( single_ds )
#    #    plt.figure()
#    #    l_ds_new[-1]['pat_num_CPPA'][0].plot()
#    
#    #plt.figure()
#    ex['max_N_regs'] = int(np.nanmax(PRECURSOR_DATA))
#    #%%
#    return l_ds_new, ex

#def get_area(ds):
#    longitude = ds.longitude
#    latitude = ds.latitude
#    
#    Erad = 6.371e6 # [m] Earth radius
##    global_surface = 510064471909788
#    # Semiconstants
#    gridcell = np.abs(longitude[1] - longitude[0]).values # [degrees] grid cell size
#    
#    # new area size calculation:
#    lat_n_bound = np.minimum(90.0 , latitude + 0.5*gridcell)
#    lat_s_bound = np.maximum(-90.0 , latitude - 0.5*gridcell)
#    
#    A_gridcell = np.zeros([len(latitude),1])
#    A_gridcell[:,0] = (np.pi/180.0)*Erad**2 * abs( np.sin(lat_s_bound*np.pi/180.0) - np.sin(lat_n_bound*np.pi/180.0) ) * gridcell
#    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
##    A_mean = np.mean(A_gridcell2D)
#    return A_gridcell2D
## =============================================================================
## =============================================================================
## Plotting functions
## =============================================================================
## =============================================================================
#    
#def extend_longitude(data):
#    import xarray as xr
#    import numpy as np
#    plottable = xr.concat([data, data.sel(longitude=data.longitude[:1])], dim='longitude').to_dataset(name="ds")
#    plottable["longitude"] = np.linspace(0,360, len(plottable.longitude))
#    plottable = plottable.to_array(dim='ds')
#    return plottable
#
#
#def plot_earth(view="EARTH", kwrgs={'cen_lon':0}):
#    #%%
#    import cartopy.crs as ccrs
#    import cartopy.feature as cfeature
#    # Create Big Figure
#    plt.rcParams['figure.figsize'] = [18, 12]
#
#    # create Projection and Map Elements
#    projection = ccrs.PlateCarree(
#            central_longitude=kwrgs['cen_lon'])
#    ax = plt.axes(projection=projection)
#    ax.add_feature(cfeature.COASTLINE)
#    ax.add_feature(cfeature.BORDERS)
#    ax.add_feature(cfeature.STATES)
#    ax.add_feature(cfeature.OCEAN, color="white")
#    ax.add_feature(cfeature.LAND, color="lightgray")
#
#    if view == "US":
#        ax.set_xlim(-130, -65)
#        ax.set_ylim(25, 50)
#    elif view == "EAST US":
#        ax.set_xlim(-105, -65)
#        ax.set_ylim(25, 50)
#    elif view == "EARTH":
#        ax.set_xlim(-180, 180)
#        ax.set_ylim(-90, 90)
#    #%%
#    return projection, ax
#




#def figure_for_schematic(iter_regions, composite_p1, chunks, lats, lons, ex):
#    #%%
#    reg_all_1 = iter_regions[:len(chunks)]
#    
#    map_proj = ccrs.PlateCarree(central_longitude=220) 
#    regions = np.reshape(reg_all_1, (reg_all_1.shape[0], lats.size, lons.size) )
#    name_chnks = [str(chnk) for chnk in chunks]
#    regions = xr.DataArray(regions, coords=[name_chnks, lats, lons], 
#                           dims=['yrs_out', 'latitude', 'longitude'])
#    folder = os.path.join(ex['figpathbase'], ex['CPPA_folder'], 'schematic_fig/')
#    
#    plots = 3
#    subset = np.linspace(0,regions.yrs_out.size-1,plots, dtype=int)
#    subset = [  1,  42, 80]
#    regions = regions.isel(yrs_out=subset)
#    regions  = regions.sel(latitude=slice(60.,0.))
#    regions = regions.sel(longitude=slice(160, 250))
#    
#    if os.path.isdir(folder) != True : os.makedirs(folder)
#    for i in range(len(subset))[::int(plots/3)]:
#        i = int(i)
#        cmap = plt.cm.YlOrBr
#        fig = plt.figure(figsize = (14,9))
#        ax = plt.subplot(111, projection=map_proj) 
#        plotdata = regions.isel(yrs_out=i).where(regions.isel(yrs_out=i) > 0.)
#        plotdata.plot.pcolormesh(ax=ax, cmap=cmap, vmin=0, vmax=plots,
#                               transform=ccrs.PlateCarree(),
#                               subplot_kws={'projection': map_proj},
#                                add_colorbar=False, alpha=0.9)
#        ax.coastlines(color='black', alpha=0.8, linewidth=2)
##        list_points = np.argwhere(regions.isel(yrs_out=i).values == 1)
##        x_co = regions.isel(yrs_out=i).longitude.values
##        y_co = regions.isel(yrs_out=i).latitude.values
##        for p in list_points:
##            ax.scatter(x_co[p[1]], y_co[p[0]], marker='$1$', color='black', 
##                      s=70, transform=ccrs.PlateCarree())
#    #    ax.set_extent([-110, 150,0,80], crs=ccrs.PlateCarree())
##        ax.set_extent([140, 266,0,60])
##        ax.outline_patch.set_visible(False)
##        ax.background_patch.set_visible(False)
#        ax.set_title('')
#        title = str(plotdata.yrs_out.values) 
#        t = ax.text(0.006, 0.008, 
#                    'excl. {}'.format(title),
#            verticalalignment='bottom', horizontalalignment='left',
#            transform=ax.transAxes,
#            color='black', fontsize=30.37, weight='bold')
#        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='red'))
#        
#        fig.savefig(folder+title+'.pdf',  bbox_inches='tight')
##%%
#    # figure robustness 1
#    import matplotlib.patches as mpatches
#    import cartopy.feature as cfeature
#    lats_fig = slice(60.,5.)
#    lons_fig = slice(165, 243)
#    mask_final = ( np.sum(reg_all_1, axis=0) < int(ex['FCP_thres'] * len(chunks)))
#    nparray_comp = np.reshape(np.nan_to_num(composite_p1.values), (composite_p1.size))
#    Corr_Coeff = np.ma.MaskedArray(nparray_comp, mask=mask_final)
#    lat_grid = composite_p1.latitude.values
#    lon_grid = composite_p1.longitude.values
#
#    # retrieve regions sorted in order of 'strength'
#    # strength is defined as an area weighted values in the composite
#    Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid, A_gs, ex)
#
#    # reshape to latlon grid
#    npmap = np.reshape(Regions_lag_i, (lats.size, lons.size))
#    mask_strongest = (npmap!=0) 
#    npmap[mask_strongest==False] = 0
#    xrnpmap_init = composite_p1.copy()
#    xrnpmap_init.values = npmap
#    xrnpmap_init  = xrnpmap_init.sel(latitude=lats_fig)
#    xrnpmap_init = xrnpmap_init.sel(longitude=lons_fig)
#    mask_final   = xrnpmap_init!= 0.
#    xrnpmap_init = xrnpmap_init.where(mask_final)    
#
#
#    # renumber
#    regions_label = np.unique(xrnpmap_init.values)[np.isnan(np.unique(xrnpmap_init.values))==False]
#    for i in range(regions_label.size):
#        r = regions_label[i]
#        xrnpmap_init.values[xrnpmap_init.values==r] = i+1
#        
#    
#    regions = np.reshape(reg_all_1, (reg_all_1.shape[0], lats.size, lons.size) )
#    name_chnks = [str(chnk) for chnk in chunks]
#    regions = xr.DataArray(regions, coords=[name_chnks, lats, lons], 
#                           dims=['yrs_out', 'latitude', 'longitude'])
#    regions  = regions.sel(latitude=lats_fig)
#    regions = regions.sel(longitude=lons_fig)
#    
#    fig = plt.figure(figsize = (20,14))
#    ax = plt.subplot(111, projection=map_proj)
#    robustness = np.sum(regions,axis=0)
#    n_max = robustness.max().values
#    freq_rawprec = regions.isel(yrs_out=i).copy()
#    freq_rawprec.values = robustness
#    plotdata = plotdata.sel(latitude=lats_fig)
#    plotdata = plotdata.sel(longitude=lons_fig)
#    npones = np.ones( (plotdata.shape) )
#    npones[mask_final.values==True] = 0
#    plotdata.values = npones
#    plotdata = plotdata.where(freq_rawprec.values > 0.)
#    cmap = colors.ListedColormap(['lemonchiffon' ])
#    plotdata.where(mask_final.values==False).plot.pcolormesh(ax=ax, cmap=cmap,
#                               transform=ccrs.PlateCarree(), vmin=0, vmax=plots,
#                               subplot_kws={'projection': map_proj},
#                                add_colorbar=False, alpha=0.3)
#    
##    freq_rawprec.plot.contour(ax=ax, 
##                               transform=ccrs.PlateCarree(), linewidths=3,
##                               colors=['black'], levels=[0., (ex['FCP_thres'] * n_max)-1, n_max],
##                               subplot_kws={'projection': map_proj},
##                               )
#    
#    n_regs = xrnpmap_init.max().values
#    xrnpmap_init.values = xrnpmap_init.values - 0.5
#    kwrgs = dict( {        'steps' : n_regs+1, 
#                           'vmin' : 0, 'vmax' : n_regs, 
#                           'cmap' : plt.cm.tab20, 
#                           'cticks_center' : True} )
#    
#    cmap = colors.ListedColormap(['cyan', 'green', 'purple' ])
#    clevels = np.linspace(kwrgs['vmin'], kwrgs['vmax'],kwrgs['steps'], dtype=int)
#    im = xrnpmap_init.plot.pcolormesh(ax=ax, cmap=cmap,
#                               transform=ccrs.PlateCarree(), levels=clevels,
#                               subplot_kws={'projection': map_proj},
#                                add_colorbar=False, alpha=0.5)
#    
#    freq_rawprec = freq_rawprec.where(freq_rawprec.values > 0)
#
#    ax.coastlines(color='black', alpha=0.8, linewidth=2)
#    ax.add_feature(cfeature.LAND, facecolor='silver')
#    list_points = np.argwhere(np.logical_and(freq_rawprec.values > 0, mask_final.values==False))
#    x_co = freq_rawprec.longitude.values
#    y_co = freq_rawprec.latitude.values
##        list_points = list_points - ex['grid_res']/2.
#    for p in list_points:
#        valueint = int((freq_rawprec.sel(latitude=y_co[p[0]], longitude=x_co[p[1]]).values))
#        value = str(  np.round( (int((valueint / n_max)*10)/10), 1)  )
#        ax.scatter(x_co[p[1]], y_co[p[0]], marker='${:}$'.format(value), color='black', 
#                   s=150, alpha=0.2, transform=ccrs.PlateCarree())
#    
#
#
#
#    ax.set_title('')
#    list_points = np.argwhere(mask_final.values==True)
#    x_co = freq_rawprec.longitude.values
#    y_co = freq_rawprec.latitude.values
#    for p in list_points:
#        valueint = int((freq_rawprec.sel(latitude=y_co[p[0]], longitude=x_co[p[1]]).values))
#        value =   np.round( (int((valueint / n_max)*10)/10), 1) 
#        if value == 1.0: value = int(value)
#        ax.scatter(x_co[p[1]], y_co[p[0]], marker='${:}$'.format(str(value)), color='black', 
#                   s=400, transform=ccrs.PlateCarree())
#    
#    cbar_ax = fig.add_axes([0.265, 0.07, 
#                                  0.5, 0.04], label='cbar')
#    norm = colors.BoundaryNorm(boundaries=clevels, ncolors=256)
#
#    cbar = plt.colorbar(im, cbar_ax, cmap=plt.cm.tab20, orientation='horizontal', 
#             extend='neither', norm=norm)
#    cbar.set_ticks([])
#    ticklabels = np.array(clevels, dtype=int)
#    cbar.set_ticklabels(ticklabels, update_ticks=True)
#    cbar.update_ticks()
#    cbar.set_label('Precursor regions', fontsize=30)
#    cbar.ax.tick_params(labelsize=30)
#    
#    yellowpatch = mpatches.Patch(color='lemonchiffon', alpha=1, 
#                                 label='Gridcells rejected')
#    ax.legend(handles=[yellowpatch], loc='lower left', fontsize=30)
#    title = 'Sum of incomplete composites'
#
##    text = ['Precursor mask', 
##            'Precursor regions']
##    text_add = ['(all gridcells where N-FRP > 0.8)\n',
##                '(seperate colors)\n']
##    max_len = max([len(t) for t in text])
##    for t in text:
##        idx = text.index(t)
##        t_len = len(t)
##        expand = max_len - t_len
###        if idx == 0: expand -= 2
##        text[idx] = t + ' ' * (expand) + '   :    ' + text_add[idx]
##    
##    text = text[0] + text[1] 
##    t = ax.text(0.004, 0.995, text,
##        verticalalignment='top', horizontalalignment='left',
##        transform=ax.transAxes,
##        color='white', fontsize=35)
##    t.set_bbox(dict(facecolor='black', alpha=1.0, edgecolor='grey'))
#    
#    fig.savefig(folder+title+'.pdf', bbox_inches='tight')
#    
#
#    
#    #%%
#    # composite mean
#
#    composite  = composite_p1.sel(latitude=lats_fig)
#    composite = composite.sel(longitude=lons_fig)
#    composite = composite * (freq_rawprec / freq_rawprec.max())
#    composite = composite.where(mask_final.values==True)
#    fig = plt.figure(figsize = (20,12))
#    ax = plt.subplot(111, projection=map_proj)
#    clevels = np.linspace(-0.5, 0.5, 11)
#    im = composite.plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
#                           transform=ccrs.PlateCarree(), levels=clevels,
#                           subplot_kws={'projection': map_proj},
#                            add_colorbar=False)
#    ax.coastlines(color='black', alpha=0.8, linewidth=2)
#    ax.add_feature(cfeature.LAND, facecolor='silver')
#    cbar_ax = fig.add_axes([0.265, 0.07, 
#                                  0.5, 0.04], label='cbar')
#    norm = colors.BoundaryNorm(boundaries=clevels, ncolors=256)
#    
#    cbar = plt.colorbar(im, cbar_ax, cmap=plt.cm.tab20, orientation='horizontal', 
#             extend='both', norm=norm)
#    cbar.set_ticks([-0.5, -0.3, 0.0, 0.3, 0.5])
#    cbar.ax.tick_params(labelsize=30)
#    cbar.set_label('Sea Surface Temperature [Kelvin]', fontsize=30)
#    ax.set_title('')
##    t = ax.text(0.006, 0.994, ('Composite mean all training years\n(Precursor mask applied and weighted by N-FRP)'),
##        verticalalignment='top', horizontalalignment='left',
##        transform=ax.transAxes,
##        color='white', fontsize=35)
##    t.set_bbox(dict(facecolor='black', alpha=1.0, edgecolor='grey'))
#    title = 'final_composite'
#    fig.savefig(folder+title+'.pdf', bbox_inches='tight')
##%%
#    #%%
#    return

##    # figure robustness 2
##    #%%
##    weights = np.sum(reg_all_1, axis=0)
##    weights[mask_final==True] = 0.
##    sum_count = np.reshape(weights, (lats.size, lons.size))
##    weights = sum_count / np.max(sum_count)
##    
##
##    
##    fig = plt.figure(figsize = (20,12))
##    ax = plt.subplot(111, projection=map_proj)
##    n_regs = xrnpmap_init.max().values
##
##    ax.coastlines(color='black', alpha=0.8, linewidth=2)
##    mask_wgths = xrdata.where(np.isnan(xrnpmap_init) == False)
##    list_points = np.argwhere(mask_wgths.values > int(ex['FCP_thres'] * len(chunks)) )
##    x_co = mask_wgths.longitude.values
##    y_co = mask_wgths.latitude.values
###        list_points = list_points - ex['grid_res']/2.
##    for p in list_points:
##        valueint = int((mask_wgths.sel(latitude=y_co[p[0]], longitude=x_co[p[1]]).values))
##        value = str(  np.round( ((valueint / n_max)), 1) )
##        ax.scatter(x_co[p[1]], y_co[p[0]], marker='${:}$'.format(value), color='black', 
##                   s=200, transform=ccrs.PlateCarree())
##    
##    xrnpmap_init.values = xrnpmap_init.values - 0.5
##    kwrgs = dict( {        'steps' : n_regs+1, 
##                           'vmin' : 0, 'vmax' : n_regs, 
##                           'cmap' : plt.cm.tab20, 
##                           'cticks_center' : True} )
##
##    clevels = np.linspace(kwrgs['vmin'], kwrgs['vmax'],kwrgs['steps'], dtype=int)
##    
##    im = xrnpmap_init.plot.pcolormesh(ax=ax, cmap=plt.cm.tab20,
##                               transform=ccrs.PlateCarree(), levels=clevels,
##                               subplot_kws={'projection': map_proj},
##                                add_colorbar=False, alpha=0.5)
##      
##    cbar_ax = fig.add_axes([0.265, 0.18, 
##                                  0.5, 0.04], label='cbar')
##    norm = colors.BoundaryNorm(boundaries=clevels, ncolors=256)
##
##    cbar = plt.colorbar(im, cbar_ax, cmap=plt.cm.tab20, orientation='horizontal', 
##             extend='neither', norm=norm)
##    cbar.set_ticks(clevels + 0.5)
##    ticklabels = np.array(clevels+1, dtype=int)
##    cbar.set_ticklabels(ticklabels, update_ticks=True)
##    cbar.update_ticks()
##    
##    cbar.set_label('Region label', fontsize=16)
##    cbar.ax.tick_params(labelsize=14)
###    ax.outline_patch.set_visible(False)
###    ax.background_patch.set_visible(False)
##    ax.set_title('')
##    title = 'Sum of incomplete composites'
##    t = ax.text(0.006, 0.994, ('"Robustness weights", (n/{:.0f})'.format(
##                                robustness.max().values)),
###                            'black contour line shows the gridcell passing the\n'
###                            '\'Composite robustness threshold\''),
##        verticalalignment='top', horizontalalignment='left',
##        transform=ax.transAxes,
##        color='black', fontsize=20)
##    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='green'))
##    title = 'robustness_weights'
##    fig.savefig(folder+title+'.pdf', bbox_inches='tight')
