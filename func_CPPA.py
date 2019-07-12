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
from sklearn.cluster import DBSCAN
import scipy 
flatten = lambda l: [item for sublist in l for item in sublist]
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


def main(RV_ts, Prec_reg, ex):
    #%%
    if (ex['method'] == 'no_train_test_split') : ex['n_conv'] = 1
    if ex['method'][:5] == 'split' : ex['n_conv'] = 1
    if ex['method'][:6] == 'random' : ex['n_conv'] = int(ex['method'][6:8])
    if ex['method'] == 'iter': ex['n_conv'] = ex['n_yrs'] 
        
    
    if ex['ROC_leave_n_out'] == True or ex['method'] == 'no_train_test_split': 
        print('leave_n_out set to False')
        ex['leave_n_out'] = False
    else:
        ex['tested_yrs'] = []


    rmwhere, window = ex['rollingmean']
    if rmwhere == 'all' and window != 1:
        Prec_reg = rolling_mean_time(Prec_reg, ex, center=False)
    
    train_test_list  = []
    l_ds_CPPA        = []    
    
    n = 0
    for n in range(ex['n_conv']):
        train_all_test_n_out = (ex['ROC_leave_n_out'] == True) & (n==0) 
        ex['n'] = n
        # do single run    
        # =============================================================================
        # Create train test set according to settings 
        # =============================================================================
        train, test, ex = train_test_wrapper(RV_ts, Prec_reg, ex)       
        # =============================================================================
        # Calculate Precursor
        # =============================================================================
        if train_all_test_n_out == True:
            # only train once on all years if ROC_leave_n_out == True
            ds_Sem = extract_precursor(Prec_reg, train, test, ex)    
            if ex['wghts_accross_lags'] == True:
                ds_Sem['pattern'] = filter_autocorrelation(ds_Sem, ex)
                           
        # Force Leave_n_out validation even though pattern is based on whole dataset
        if (ex['ROC_leave_n_out'] == True) & (ex['n']==0):
            # start selecting leave_n_out
            ex['leave_n_out'] = True
            train, test, ex = rand_traintest(RV_ts, Prec_reg, 
                                              ex)    
        
#        elif (ex['leave_n_out'] == True) & (ex['ROC_leave_n_out'] == False):
#        elif (ex['method'] == 'iter') or ex['method'][:5] == 'split' or ex['method'] == 'no_train_test_split':
        elif train_all_test_n_out == False:
            # train each time on only train years
            ds_Sem = extract_precursor(Prec_reg, train, test, ex)    
            
            if ex['wghts_accross_lags'] == True:
                ds_Sem['pattern'] = filter_autocorrelation(ds_Sem, ex)
                 
            

        l_ds_CPPA.append(ds_Sem)     
        
        # appending tuple
        train_test_list.append( (train, test) )
    if ex['method'] != 'no_train_test_split':
        if len(set(flatten(ex['tested_yrs']))) != ex['n_yrs']:
            print('train test set appears to contain duplicates')
    ex['train_test_list'] = train_test_list

    #%%
    return l_ds_CPPA, ex



# =============================================================================
# =============================================================================
# Wrapper functions
# =============================================================================
# =============================================================================

def train_test_wrapper(RV_ts, Prec_reg, ex):
    #%%
    now = datetime.datetime.now()
    rmwhere, window = ex['rollingmean']
    if 'RV_aggregation' not in ex.keys():
        ex['RV_aggregation'] = 'RVfullts95'
    
    if ex['leave_n_out'] == True and ex['method'][:6] == 'random':
        train, test, ex = rand_traintest(RV_ts, Prec_reg, 
                                          ex)
        if ex['n']==0:
            general_folder = '{}_leave_{}_out_{}_{}_tf{}_{}p_{}deg_{}nyr_{}tperc_{}tc_{}_rng{}_{}'.format(
                            ex['method'], ex['leave_n_years_out'], ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['event_percentile'], ex['grid_res'],
                          ex['n_oneyr'], 
                          ex['SCM_percentile_thres'], ex['FCP_thres'], ex['RV_aggregation'],
                          ex['seed'], now.strftime("%Y-%m-%d"))
                          

    elif ex['method']=='no_train_test_split' and ex['n']==0:
        print('Training on all years')
        ex['leave_n_years_out'] = 0
        Prec_train_idx = np.arange(0, Prec_reg.time.size) #range(len(full_years)) if full_years[i] in rand_train_years]
        train = dict( { 'Prec_train_idx' : Prec_train_idx,
                        'RV'    : RV_ts})
                        
        test = dict( { 'Prec_test_idx' : Prec_train_idx,
                        'RV'    : RV_ts})
        
        if ex['n']==0:    
            general_folder = 'hindcast_{}_{}_tf{}_{}p_{}deg_{}nyr_{}tperc_{}tc_{}_{}'.format(
                          ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['event_percentile'], ex['grid_res'],
                          ex['n_oneyr'], 
                          ex['SCM_percentile_thres'], ex['FCP_thres'], ex['RV_aggregation'], 
                          now.strftime("%Y-%m-%d"))
        
    elif ex['n']==0:
        train, test, ex = rand_traintest(RV_ts, Prec_reg, 
                                          ex)
    
        if ex['n']==0:
            general_folder = '{}_{}_{}_tf{}_{}p_{}deg_{}nyr_{}tperc_{}tc_{}_{}'.format(
                            ex['method'], ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['event_percentile'], ex['grid_res'],
                          ex['n_oneyr'], 
                          ex['SCM_percentile_thres'], ex['FCP_thres'], ex['RV_aggregation'],
                          now.strftime("%Y-%m-%d"))
        
        
        ex['test_years'] = 'all_years'

    if ex['n']==0:    
        subfolder         = 'lags{}Ev{}d{}p'.format(ex['lags'], ex['min_dur'], 
                                 ex['max_break'])
        subfolder = subfolder.replace(' ' ,'')
        ex['CPPA_folder'] = os.path.join(general_folder, subfolder)
        ex['output_dic_folder'] = os.path.join(ex['figpathbase'], ex['CPPA_folder'])
    

    #%%
    
    return train, test, ex
        

def extract_precursor(Prec_reg, train, test, ex):
    #%%
    Prec_train = Prec_reg.isel(time=train['Prec_train_idx'])
    lats = Prec_train.latitude
    lons = Prec_train.longitude
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern_CPPA = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='communities_composite',
                          attrs={'units':'Kelvin'})


    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pat_num_CPPA = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='commun_numb_init', 
                          attrs={'units':'Precursor regions'})
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    std_train_min_lag = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='std_train_min_lag', 
                          attrs={'units':'std [-]'})

    pat_num_CPPA.name = 'commun_numbered'
    
    weights     = pattern_CPPA.copy()
    weights.name = 'weights'
   

    RV_event_train = Ev_timeseries(train['RV'], ex['event_thres'], ex)[0]
    RV_event_train = pd.to_datetime(RV_event_train.time.values)

    RV_dates_train = pd.to_datetime(train['RV'].time.values)
    all_yrs_set = list(set(RV_dates_train.year.values))
    comp_years = list(RV_event_train.year.values)
    mask_chunks = get_chunks(all_yrs_set, comp_years, ex)
    #%%
    Comp_robust = np.ma.zeros( (len(lats) * len(lons), len(ex['lags'])) )
    
    for idx, lag in enumerate(ex['lags']):
        
        events_min_lag = func_dates_min_lag(RV_event_train, lag)[1]
        dates_train_min_lag = func_dates_min_lag(RV_dates_train, lag)[1]
        event_idx = [list(dates_train_min_lag.values).index(E) for E in events_min_lag.values]
        binary_events = np.zeros(RV_dates_train.size)    
        binary_events[event_idx] = 1
        
        std_train_min_lag[idx] = Prec_train.sel(time=dates_train_min_lag).std(dim='time', skipna=True)
        std_train_lag = std_train_min_lag[idx]
        
        
       
        # extract precursor regions composite approach
        Comp_robust[:,idx], weights[idx] = extract_regs_p1(Prec_train, mask_chunks, events_min_lag, 
                                             dates_train_min_lag, std_train_lag, ex)  
        progress = int((100*(idx+1)/len(ex['lags']) ))
        print(f"\rProgress train/test set {progress}%", end="") 
    print("\n")
#    plt.imshow(Comp_robust[:,0].reshape(lats.size, lons.size).mask)
#    weights.plot()
    #%%
    
    ex['input_freq'] = 'daily' ;  ex['file_type2'] = 'png'
    ex['exp_folder'] = ex['CPPA_folder']
    ex['splitlabeling'] = 30 
    lags = np.array(ex['lags'])
    if any(lags > 30) and any(lags <= ex['splitlabeling']):
        # split clustering accros lags
        setlags1 = lags <= ex['splitlabeling']
        setlags2 = lags > ex['splitlabeling']
        split_lags = [setlags1, setlags2]
        for i, split_lag in enumerate(split_lags):
            if i == 0: ex['distance_eps'] = ex['distance_eps_init']
            if i == 1: ex['distance_eps'] = 500
            ex['setlags'] = lags[split_lag]
            ex['params'] = str(ex['setlags'])
            ex['lag_min'] = min(lags[split_lag]) ; ex['lag_max'] = max(lags[split_lag])
            actor = act(ex['name'], Comp_robust[:,split_lag], Prec_train)
        
            actor, ex = cluster_DBSCAN_regions(actor, ex)
            pat_num_CPPA.values[split_lag] = actor.prec_labels.values
            Composite = Comp_robust[:,split_lag].data.reshape( (len(lats), len(lons), len(ex['setlags'])) )
            Composite = Composite.swapaxes(0,-1).swapaxes(1,2)
            Composite = Composite * weights.values[split_lag]
            pattern_CPPA.values[split_lag] = Composite
    else:
        ex['setlags'] = lags
        ex['distance_eps'] = ex['distance_eps_init']
        ex['params'] = str(lags)
        ex['lag_min'] = min(lags) ; ex['lag_max'] = max(lags)
        actor = act(ex['name'], Comp_robust[:,:], Prec_train)
        
        actor, ex = cluster_DBSCAN_regions(actor, ex)
        pat_num_CPPA.values = actor.prec_labels.values
        Composite = Comp_robust[:,:].reshape( (len(lats), len(lons), len(lags)) )
        Composite = Composite.swapaxes(0,-1).swapaxes(1,2)
        Composite = Composite * weights.values
        pattern_CPPA.values = Composite
#    xarray_plot(pattern_CPPA)
#        plt.figure()
#        composite_p1.plot() 
#        xrnpmap_p1.plot()

        


        #%%        
    ds_Sem = xr.Dataset( {'pattern_CPPA' : pattern_CPPA, 'pat_num_CPPA' : pat_num_CPPA,
                          'weights' : weights, 'std_train_min_lag' : std_train_min_lag } )
                          
                          
    
    return ds_Sem


# =============================================================================
# =============================================================================
# Core functions
# =============================================================================
# =============================================================================

class act:
    def __init__(self, name, Corr_Coeff, precur_arr):
        self.name = 'sst'
        self.Corr_Coeff = Corr_Coeff
        self.precur_arr = precur_arr
        self.lat_grid = precur_arr.latitude.values
        self.lon_grid = precur_arr.longitude.values
        self.area_grid = get_area(precur_arr)

def get_chunks(all_yrs_set, comp_years, ex):

    n_yrs = len(all_yrs_set)
    perc_yrs_out = ex['perc_yrs_out']
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


def extract_regs_p1(Prec_train, mask_chunks, events_min_lag, dates_train_min_lag, std_train_lag, ex):
    #%% 
#    T, pval, mask_sig = Welchs_t_test(sample, full, alpha=0.01)
#    threshold = np.reshape( mask_sig, (mask_sig.size) )
#    mask_threshold = threshold 
#    plt.figure()
#    plt.imshow(mask_sig)
    # divide train set into train-feature and train-weights part:
#    start = time.time()   
    
    lats = Prec_train.latitude
    lons = Prec_train.longitude    
    lsm  = np.isnan(Prec_train[0])
    days_before = ex['days_before']
    comp_train_stack = np.empty( (len(days_before), events_min_lag.size, lats.size* lons.size), dtype='int16')
    for i, d in enumerate(days_before):
        Prec_RV_train = Prec_train.sel(time=dates_train_min_lag - pd.Timedelta(d, 'd'))
        comp_train_i = Prec_RV_train.sel(time=events_min_lag - pd.Timedelta(d, 'd'))
        comp_train_n = np.array((comp_train_i/std_train_lag)*1000, dtype='int16')
        comp_train_n[:,lsm] = 0
        
        comp_train_n = np.reshape(np.nan_to_num(comp_train_n), 
                              (events_min_lag.size,lats.size*lons.size))
        comp_train_stack[i] = comp_train_n
        
    import numba # conda install -c conda-forge numba=0.43.1
    
    def make_composites(mask_chunk, comp_train_stack, iter_regions):
        
        
        for subset_i in range(comp_train_stack.shape[0]):
            comp_train_n = comp_train_stack[subset_i, :, :]
            for idx in range(mask_chunk.shape[0]):
#                comp_train_stack[subset_i, mask_chunk[idx], :]
                comp_subset = comp_train_n[mask_chunk[idx], :]
    
                sumcomp = np.zeros( comp_subset.shape[1] )
                for i in range(comp_subset.shape[0]):
                    sumcomp += comp_subset[i]
                mean = sumcomp / comp_subset.shape[0]
    
                threshold = np.nanpercentile(mean, 95)
                mean[np.isnan(mean)] = 0
                idx += subset_i * mask_chunk.shape[0]

                iter_regions[idx] = np.abs(mean) > ( threshold )

            
        return iter_regions

    jit_make_composites = numba.jit(nopython=True, parallel=True)(make_composites)
    
    iter_regions = np.zeros( (comp_train_stack.shape[0]*len(mask_chunks), comp_train_stack[0,0].size), dtype='int8')
    iter_regions = jit_make_composites(mask_chunks, comp_train_stack, iter_regions)
    
#    iter_regions = make_composites(mask_chunks, comp_train_stack, iter_regions)
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(np.sum(iter_regions, axis=0), (lats.size, lons.size))) ; plt.colorbar()

    mask_final = ( np.sum(iter_regions, axis=0) < int(ex['FCP_thres'] * iter_regions.shape[0]))
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(np.array(mask_final, dtype=int), (lats.size, lons.size))) ; plt.colorbar()
    weights = np.sum(iter_regions, axis=0)
    weights[mask_final==True] = 0.
    sum_count = np.reshape(weights, (lats.size, lons.size))
    weights = sum_count / np.max(sum_count)
    
    composite_p1 = Prec_train.sel(time=events_min_lag).mean(dim='time', skipna=True)
    nparray_comp = np.reshape(np.nan_to_num(composite_p1.values), (composite_p1.size))
#    nparray_comp = np.nan_to_num(composite_p1.values)
    Comp_robust_lag = np.ma.MaskedArray(nparray_comp, mask=mask_final)
    #%%
    return Comp_robust_lag, weights




def spatial_mean_regions(Regions_lag_i, regions_for_ts, ts_3d, npmean):
    #%%
    n_time   = ts_3d.time.size
    lat_grid = ts_3d.latitude
    lon_grid = ts_3d.longitude
    regions_for_ts = list(regions_for_ts)
    
    actbox = np.reshape(ts_3d.values, (n_time, 
                  lat_grid.size*lon_grid.size))  
    
    # get lonlat array of area for taking spatial means 
    lons_gph, lats_gph = np.meshgrid(lon_grid, lat_grid)
    cos_box = np.cos(np.deg2rad(lats_gph))
    cos_box_array = np.repeat(cos_box[None,:], actbox.shape[0], 0)
    cos_box_array = np.reshape(cos_box_array, (cos_box_array.shape[0], -1))
    

    # this array will be the time series for each feature
    ts_regions_lag_i = np.zeros((actbox.shape[0], len(regions_for_ts)))
    
    # track sign of eacht region
    sign_ts_regions = np.zeros( len(regions_for_ts) )
    
    # std regions
    std_regions     = np.zeros( (len(regions_for_ts)) )
    
    # composite needed for sign
    meanbox = np.reshape(npmean, (lat_grid.size*lon_grid.size))
    
    if Regions_lag_i.shape == actbox[0].shape:
        Regions = Regions_lag_i
    elif Regions_lag_i.shape == (lat_grid.shape[0], lon_grid.shape[0]):
        Regions = np.reshape(Regions_lag_i, (Regions_lag_i.size))
    # calculate area-weighted mean over features
    for r in regions_for_ts:
        idx = regions_for_ts.index(r)
        # start with empty lonlat array
        B = np.zeros(Regions.shape)
        # Mask everything except region of interest
        B[Regions == r] = 1	
#        # Calculates how values inside region vary over time, wgts vs anomaly
#        wgts_ano = meanbox[B==1] / meanbox[B==1].max()
#        ts_regions_lag_i[:,idx] = np.nanmean(actbox[:,B==1] * cos_box_array[:,B==1] * wgts_ano, axis =1)
        # Calculates how values inside region vary over time
        ts_regions_lag_i[:,idx] = np.nanmean(actbox[:,B==1] * cos_box_array[:,B==1], axis =1)
        # get sign of region
        sign_ts_regions[idx] = np.sign(np.mean(meanbox[B==1]))
#    print(sign_ts_regions)
        
    std_regions = np.std(ts_regions_lag_i, axis=0)
    #%%
    return ts_regions_lag_i, sign_ts_regions, std_regions


def store_ts_wrapper(l_ds_CPPA, RV_ts, Prec_reg, ex):
    #%%
    ex['output_ts_folder'] = os.path.join(ex['output_dic_folder'], 'timeseries_robwghts')
    if os.path.isdir(ex['output_ts_folder']) != True : os.makedirs(ex['output_ts_folder'])
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        test =ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        
        print('Storing timeseries using patterns retrieved '
              'without test year(s) {}'.format(ex['test_year']))
        
        ds_Sem = l_ds_CPPA[n]
        
        
        store_timeseries(ds_Sem, RV_ts, Prec_reg, ex)
    #%%
    return

def store_timeseries(ds_Sem, RV_ts, Prec_reg, ex):
    #%%
    ts_3d    = Prec_reg
    # mean of El nino 3.4
    ts_3d_nino = find_region(Prec_reg, region='elnino3.4')[0]
    # get lonlat array of area for taking spatial means 
    nino_index = area_weighted(ts_3d_nino).mean(dim=('latitude', 'longitude'), skipna=True).values
    dates = pd.to_datetime(ts_3d.time.values)
    dates -= pd.Timedelta(dates.hour[0], unit='h')
    df_nino = pd.DataFrame(data = nino_index[:,None], index=dates, columns=['nino3.4']) 
    df_nino['nino3.4rm5'] = df_nino['nino3.4'].rolling(int((365/12)*5), min_periods=1).mean()
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)


        mask_regions = np.nan_to_num(ds_Sem['pat_num_CPPA_clust'].sel(lag=lag).values) >= 1
        # Make time series for whole period
        
        mask_notnan = (np.product(np.isnan(ts_3d.values),axis=0)==False) # nans == False
        mask = mask_notnan * mask_regions
        ts_3d_mask     = ts_3d.where(mask==True)
        # ts_3d is given more weight to robust precursor regions
        ts_3d_w  = ts_3d_mask  * ds_Sem['weights'].sel(lag=lag)
        # ts_3d_w is normalized w.r.t. std in RV dates min lag
        ts_3d_nw = ts_3d_w / ds_Sem['std_train_min_lag'][idx]
        # same is done for pattern
        pattern_CPPA = ds_Sem['pattern_CPPA'].sel(lag=lag)
        CPPA_w = pattern_CPPA #* ds_Sem['weights'].sel(lag=lag)
        CPPA_nw = CPPA_w / ds_Sem['std_train_min_lag'][idx]
        
        
        # regions for time series
        Regions_lag_i = ds_Sem['pat_num_CPPA_clust'][idx].squeeze().values
        regions_for_ts = np.unique(Regions_lag_i[~np.isnan(Regions_lag_i)])
        # spatial mean (normalized & weighted)
        ts_regions_lag_i, sign_ts_regions = spatial_mean_regions(Regions_lag_i, 
                                regions_for_ts, ts_3d_w, CPPA_w.values)[:2]
        
        
        check_nans = np.where(np.isnan(ts_regions_lag_i))
        if check_nans[0].size != 0:
            print('{} nans found in time series of region {}, dropping this region.'.format(
                    check_nans[0].size, 
                    np.unique(regions_for_ts[check_nans[1]])))
            regions_for_ts = np.delete(regions_for_ts, check_nans[1])
            ts_regions_lag_i = np.delete(ts_regions_lag_i, check_nans[1], axis=1)
            sign_ts_regions  = np.delete(sign_ts_regions, check_nans[1], axis=0)
        
        
        # spatial covariance of whole CPPA pattern
        spatcov_CPPA = cross_correlation_patterns(ts_3d_w, pattern_CPPA)

        
        # merge data
        columns = list(np.array(regions_for_ts, dtype=int))
        columns.insert(0, 'spatcov_CPPA')

               
        data = np.concatenate([spatcov_CPPA.values[:,None],
                               ts_regions_lag_i], axis=1)
        data = np.array(data, dtype='float16')
        dates = pd.to_datetime(ts_3d.time.values)
        dates -= pd.Timedelta(dates.hour[0], unit='h')
        df_CPPA = pd.DataFrame(data = data, index=dates, columns=columns) 
        
        
        df = pd.concat([df_nino, df_CPPA], axis=1)
        df.index.name = 'date'
        
        name_trainset = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
        df.to_csv(os.path.join(ex['output_ts_folder'], name_trainset ), 
                  float_format='%.5f', chunksize=int(dates.size/3))
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


def rand_traintest(RV_ts, Prec_reg, ex):
    #%%
    
    
    
    if ex['n'] == 0: ex['tested_yrs'] = [] ; ex['n_events'] = []
    ex['all_yrs'] = list(np.unique(RV_ts.time.dt.year))
    
    if ex['datafolder'] == 'ERAint': ex['all_yrs'].append(2018)
    
    tol_from_exp_events = 0.35

    if ex['method'][:6] == 'random':
        if 'seed' not in ex.keys():
            ex['seed'] = 30 # control reproducibility train/test split
        else:
            ex['seed'] = ex['seed']
        if ex['n'] == 0: ex['seed_run'] = ex['seed']
        rng = np.random.RandomState(ex['seed_run'])
    
    
    # conditions failed initally assumed True
    a_conditions_failed = True
    count = 0

    while a_conditions_failed == True:
        count +=1
        a_conditions_failed = False


        if ex['method'][:6] == 'random':

            
            size_test  = int(np.round(ex['n_yrs'] / int(ex['method'][6:8])))
            size_train = int(ex['n_yrs'] - size_test)

            ex['leave_n_years_out'] = size_test
            yrs_to_draw_sample = [yr for yr in ex['all_yrs'] if yr not in flatten(ex['tested_yrs'])]
            if (len(yrs_to_draw_sample)) >= size_test:
                rand_test_years = rng.choice(yrs_to_draw_sample, ex['leave_n_years_out'], replace=False)
            # if last test sample will be too small for next iteration, add test yrs to current test yrs
            if (len(yrs_to_draw_sample)) < size_test:
                rand_test_years = yrs_to_draw_sample  
            check_double_test = [yr for yr in rand_test_years if yr in flatten( ex['tested_yrs'] )]
            if len(check_double_test) != 0 :
                a_conditions_failed = True
                print('test year drawn twice, redoing sampling')
                
            
        elif ex['method'] == 'iter':
            ex['leave_n_years_out'] = 1
            if ex['n'] >= ex['n_yrs']:
                n = ex['n'] - ex['n_yrs']
            else:
                n = ex['n']
            rand_test_years = [ex['all_yrs'][n]]
            
        elif ex['method'][:5] == 'split':
            size_train = int(np.percentile(range(len(ex['all_yrs'])), int(ex['method'][5:])))
            size_test  = len(ex['all_yrs']) - size_train
            ex['leave_n_years_out'] = size_test
            print('Using {} years to train and {} to test'.format(size_train, size_test))
            rand_test_years = ex['all_yrs'][-size_test:]
        
        # remove 2018 again
        if ex['datafolder'] == 'ERAint' and 2018 in rand_test_years:
            ex['all_yrs'] = np.unique(RV_ts.time.dt.year)
            rand_test_years = [y for y in rand_test_years if y != 2018]
        
            
        # test duplicates
        a_conditions_failed = np.logical_and((len(set(rand_test_years)) != ex['leave_n_years_out']),
                                             ex['n'] != ex['n_conv']-1)
        # Update random years to be selected as test years:
    #        initial_years = [yr for yr in initial_years if yr not in random_test_years]
        rand_train_years = [yr for yr in ex['all_yrs'] if yr not in rand_test_years]
        

        full_years  = list(Prec_reg.time.dt.year.values)
        RV_years  = list(RV_ts.time.dt.year.values)
        
        Prec_train_idx = [i for i in range(len(full_years)) if full_years[i] in rand_train_years]
        RV_train_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_train_years]
        
        Prec_test_idx = [i for i in range(len(full_years)) if full_years[i] in rand_test_years]
        RV_test_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_test_years]
        
        

        RV_train = RV_ts.isel(time=RV_train_idx)
        
        RV_test = RV_ts.isel(time=RV_test_idx)
        
        event_train = Ev_timeseries(RV_train, ex['event_thres'], ex)[0].time
        event_test = Ev_timeseries(RV_test, ex['event_thres'], ex)[0].time
        
        test_years = [yr for yr in list(set(RV_years)) if yr in rand_test_years]
        
        ave_events_pyr = (len(event_train) + len(event_test))/len(ex['all_yrs'])
        exp_events     = int(ave_events_pyr) * len(rand_test_years)
        tolerance      = tol_from_exp_events * exp_events
        diff           = abs(len(event_test) - exp_events)
        
        
        if diff > tolerance and ex['method'][:6] == 'random' and ex['n'] != ex['n_conv']-1: 
            print('not a representative sample drawn, drawing new sample')
            ex['seed_run'] += 1 # next random sample
            a_conditions_failed = True
        else:
            print('{}: test year is {}, with {} events'.format(ex['n'], test_years, len(event_test)))
        if count == 7:
            print(f"{ex['n']}: {count+1} attempts made, lowering tolence threshold from {tol_from_exp_events} "
                    "to 0.40 deviation from mean expected events" )
            tol_from_exp_events = 0.40
        if count == 10:
            print(f"kept sample after {count+1} attempts")
            print('{}: test year is {}, with {} events'.format(ex['n'], test_years, len(event_test)))
            a_conditions_failed = False
                   
    ex['tested_yrs'].append(test_years)
    ex['n_events'].append(len(event_test))
    
    train = dict( {    'RV'             : RV_train,
                       'Prec_train_idx' : Prec_train_idx})
    test = dict( {     'RV'             : RV_test,
                       'Prec_test_idx'  : Prec_test_idx})
    #%%
    return train, test, ex

def filter_autocorrelation(ds_Sem, ex):
    n_lags = len(ex['lags'])
    n_lats = ds_Sem['pattern'].latitude.size
    n_lons = ds_Sem['pattern'].longitude.size
    ex['n_steps'] = len(ex['lags'])
    weights = np.zeros( (n_lags, n_lats, n_lons) )
    xrweights = ds_Sem['pattern'].copy()
    xrweights.values = weights
    for lag in ex['lags']:
        data = np.nan_to_num(ds_Sem['pattern'].sel(lag=lag).values)
        mask = np.ma.masked_array(data, dtype=bool)
        idx = ex['lags'].index(lag)
        weights[idx] = mask
        xrweights[idx].values = mask
    weights = np.sum(weights, axis=0)
    return weights * ds_Sem['pattern']


def Welchs_t_test(sample, full, min_alpha=0.05, fieldsig=True):
    '''mask returned is True where values are non-significant'''
    from statsmodels.sandbox.stats import multicomp
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

def merge_neighbors(lsts):
  sets = [set(lst) for lst in lsts if lst]
  merged = 1
  while merged:
    merged = 0
    results = []
    while sets:
      common, rest = sets[0], sets[1:]
      sets = []
      for x in rest:
        if x.isdisjoint(common):
          sets.append(x)
        else:
          merged = 1
          common |= x
      results.append(common)
    sets = results
  return sets


def create_chunks(all_yrs_set, n_out, chunks):
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
            if n_out_part < len(yr_prior_1):
                yrs_to_list  = list(np.random.choice(yr_prior_1, n_out_part, replace=False))
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
  
    return chunks, count





def cluster_DBSCAN_regions(actor, ex):
    #%%
    """
	Calculates the time-series of the actors based on the correlation coefficients and plots the according regions. 
	Only caluclates regions with significant correlation coefficients
	"""
    from sklearn import cluster
    from sklearn import metrics
    from haversine import haversine
    import xarray as xr
    
#    var = 'sst'
#    actor = outdic_actors[var]
    Corr_Coeff  = actor.Corr_Coeff
    lats    = actor.lat_grid
    lons    = actor.lon_grid
    area_grid   = actor.area_grid/ 1E6 # in km2

    aver_area_km2 = 7939     # np.mean(actor.area_grid) with latitude 0-90 / 1E6
    wght_area = area_grid / aver_area_km2
    ex['min_area_km2'] = ex['min_area_in_degrees2'] * 111.131 * ex['min_area_in_degrees2'] * 78.85
    min_area = ex['min_area_km2'] / aver_area_km2
    ex['min_area_samples'] = min_area
    
    if Corr_Coeff.ndim == 1:
        lag_steps = 1
    else:
        lag_steps = Corr_Coeff.shape[1]
	

    Number_regions_per_lag = np.zeros(lag_steps)
    ex['n_tot_regs'] = 0
    
    
    def mask_sig_to_cluster(mask_and_data, lons, lats, wght_area, ex):
        mask_sig_1d = mask_and_data.mask[:,:]==False
        data = mask_and_data.data
    
        np_dbregs   = np.zeros( (lons.size*lats.size, lag_steps), dtype=int )
        labels_sign_lag = []
        label_start = 0
        
        for sign in [-1, 1]:
            mask = mask_sig_1d.copy()
            mask[np.sign(data) != sign] = False
            n_gc_sig_sign = mask[mask==True].size
            labels_for_lag = np.zeros( (lag_steps, n_gc_sig_sign), dtype=bool)
            meshgrid = np.meshgrid(lons.data, lats.data)
            mask_sig = np.reshape(mask, (lats.size, lons.size, lag_steps))
            sign_coords = [] ; count=0
            weights_core_samples = []
            for l in range(lag_steps):
                sign_c = meshgrid[0][ mask_sig[:,:,l] ], meshgrid[1][ mask_sig[:,:,l] ]
                n_sign_c_lag = len(sign_c[0])
                labels_for_lag[l][count:count+n_sign_c_lag] = True
                count += n_sign_c_lag
                sign_coords.append( [(sign_c[1][i], sign_c[0][i]-180) for i in range(sign_c[0].size)] )
                
                weights_core_samples.append(wght_area[mask_sig[:,:,l]].reshape(-1))
                
            sign_coords = flatten(sign_coords)
            weights_core_samples = flatten(weights_core_samples)
            # calculate distance between sign coords accross all lags to keep labels 
            # more consistent when clustering
            distance = metrics.pairwise_distances(sign_coords, metric=haversine)
            dbresult = cluster.DBSCAN(eps=ex['distance_eps'], min_samples=ex['min_area_samples'], 
                                      metric='precomputed').fit(distance, 
                                      sample_weight=weights_core_samples)
            labels = dbresult.labels_ + 1
            
            # all labels == -1 (now 0) are seen as noise:
            labels[labels==0] = -label_start
            individual_labels = labels + label_start
            [labels_sign_lag.append((l, sign)) for l in np.unique(individual_labels) if l != 0]

            for l in range(lag_steps):
                mask_sig_lag = mask[:,l]==True
                np_dbregs[:,l][mask_sig_lag] = individual_labels[labels_for_lag[l]]
            label_start = int(np_dbregs[mask].max())
            np_regs = np.reshape(np_dbregs, (lats.size, lons.size, lag_steps))
            np_regs = np_regs.swapaxes(0,-1).swapaxes(1,2)
        return np_regs, labels_sign_lag


    prec_labels_np = np.zeros( (lag_steps, lats.size, lons.size) )
    labels_sign = np.zeros( (lag_steps), dtype=list )
    mask_and_data = Corr_Coeff.copy()
    prec_labels_np, labels_sign_lag = mask_sig_to_cluster(mask_and_data, lons, lats, wght_area, ex)
    
    

    corr_strength = {}
    for lag in range(lag_steps):
        # check if region is higher lag is actually too small to be a cluster:
        prec_field = prec_labels_np[lag,:,:]
        
        for i, reg in enumerate(np.unique(prec_field)[1:]):
            are = area_grid.copy()
            are[prec_field!=reg]=0
            area_prec_reg = are.sum()/1E5
            if area_prec_reg < ex['min_area_km2']/1E5:
#                print(reg, area_prec_reg, ex['min_area_km2']/1E5, 'not exceeding min area size in m2')
                prec_field[prec_field==reg] = 0
            if area_prec_reg >= ex['min_area_km2']/1E5:
                Corr_value = mask_and_data.data[prec_field.reshape(-1)==reg, lag]
                Corr_sign  = np.sign(Corr_value.mean())
                Corr_strength = np.round(np.percentile(Corr_sign*Corr_value, 90), 10)
                corr_strength[Corr_strength + lag*1E-5] = '{}_{}'.format(lag,reg)
        Number_regions_per_lag[lag] = np.unique(prec_field)[1:].size
        prec_labels_np[lag,:,:] = prec_field
        
    # Reorder - strongest correlation region is number 1, etc... ,
    strongest = sorted(corr_strength.keys())[::-1]
    reassign = {} ; key_dupl = [] ; new_reg = 0 
    order_str_all = {}
    for i, key in enumerate(strongest):
        old_lag_reg = corr_strength[key]
        old_reg = int(old_lag_reg.split('_')[-1])
        if old_reg not in key_dupl:
            new_reg += 1
            reassign[old_reg] = new_reg
        key_dupl.append( old_reg )
        new_lag_reg = old_lag_reg.split('_')[0] +'_'+ str(reassign[old_reg])
        order_str_all[new_lag_reg] = i+1
    
    actor.order_str_all = order_str_all
    prec_labels_ord = np.zeros(prec_labels_np.shape, dtype=int)
    for i, reg in enumerate(reassign.keys()):   
        prec_labels_ord[prec_labels_np == reg] = reassign[reg]
#        print('reg {}, to {}'.format(reg, reassign[reg]))
#                

    
    actor.labels_sign   = labels_sign
    actor.n_regions_lag = Number_regions_per_lag
    ex['n_tot_regs']    += int(np.sum(Number_regions_per_lag))
    
    if 'setlags' in ex.keys():
        lags = ex['setlags']
    else:
        lags = list(range(ex['lag_min'], ex['lag_max']+1))
    lags = ['{} ({} {})'.format(l, l*ex['tfreq'], ex['input_freq'][:1]) for l in lags]
    prec_labels = xr.DataArray(data=prec_labels_ord, coords=[lags, lats, lons], 
                          dims=['lag','latitude','longitude'], 
                          name='{}_labels_init'.format(actor.name), 
                          attrs={'units':'Precursor regions [ordered for Corr strength]'})
    prec_labels = prec_labels.where(prec_labels_ord!=0.)
    prec_labels.attrs['title'] = prec_labels.name
    actor.prec_labels = prec_labels
    
    
    ex['max_N_regs'] = min(20, int(prec_labels.max() + 0.5))
    
    if lag_steps >= 2:
        adjust_vert_cbar = 0.0; adj_fig_h=1.4
    elif lag_steps < 2:
        adjust_vert_cbar = 0.1 ; adj_fig_h = 1.4
    
        
    cmap = plt.cm.tab20
    for_plt = prec_labels.copy()
    for_plt.values = for_plt.values-0.5
    kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                   'steps' : ex['max_N_regs']+1, 'subtitles': None,
                   'vmin' : 0, 'vmax' : ex['max_N_regs'], 
                   'cmap' : cmap, 'column' : 1,
                   'cbar_vert' : adjust_vert_cbar, 'cbar_hght' : 0.0,
                   'adj_fig_h' : adj_fig_h, 'adj_fig_w' : 1., 
                   'hspace' : 0.0, 'wspace' : 0.08, 
                   'cticks_center' : False, 'title_h' : 0.95} )
    filename = '{}_labels_init_{}_vs_{}'.format(ex['params'], ex['RV_name'], actor.name) + ex['file_type2']
    plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)

#    for_plt.where(for_plt.values==3)[0].plot()

#    if np.sum(Number_regions_per_lag) != 0:
#        assert np.where(np.isnan(tsCorr))[1].size < 0.5*tsCorr[:,0].size, ('more '
#                       'then 10% nans found, i.e. {} out of {} datapoints'.format(
#                               np.where(np.isnan(tsCorr))[1].size), tsCorr.size)
#        while np.where(np.isnan(tsCorr))[1].size != 0:
#            nans = np.where(np.isnan(tsCorr))
#            print('{} nans were found in timeseries of regions out of {} datapoints'.format(
#                    nans[1].size, tsCorr.size))
#            tsCorr[nans[0],nans[1]] = tsCorr[nans[0]-1,nans[1]]
#            print('taking value of previous timestep')
    #%%
    return actor, ex

def define_regions_and_rank_new(Corr_Coeff, lats, lons, A_gs, ex):
    #%%
    '''
	takes Corr Coeffs and defines regions by strength

	return A: the matrix whichs entries correspond to region. 1 = strongest, 2 = second strongest...
    '''
#    print('extracting features ...\n')

	
	# initialize arrays:
	# A final return array 
    A = np.ma.copy(Corr_Coeff)
#    A = np.ma.zeros(Corr_Coeff.shape)
	#========================================
	# STEP 1: mask nodes which were never significantly correlatated to index (= count=0)
	#========================================
	
	#========================================
	# STEP 2: define neighbors for everey node which passed Step 1
	#========================================

    indices_not_masked = np.where(A.mask==False)[0].tolist()

    lo = lons.shape[0]
    la = lats.shape[0]
	
	# create list of potential neighbors:
    N_pot=[[] for i in range(A.shape[0])]

	#=====================
	# Criteria 1: must bei geographical neighbors:
    n_between = ex['prec_reg_max_d']
	#=====================
    for i in indices_not_masked:
        neighb = []
        def find_neighboors(i, lo):
            n = []	
    
            col_i= i%lo
            row_i = i//lo
    
    		# knoten links oben
            if i==0:	
                n= n+[lo-1, i+1, lo ]
    
    		# knoten rechts oben	
            elif i== lo-1:
                n= n+[i-1, 0, i+lo]
    
    		# knoten links unten
            elif i==(la-1)*lo:
                n= n+ [i+lo-1, i+1, i-lo]
    
    		# knoten rechts unten
            elif i == la*lo-1:
                n= n+ [i-1, i-lo+1, i-lo]
    
    		# erste zeile
            elif i<lo:
                n= n+[i-1, i+1, i+lo]
    	
    		# letzte zeile:
            elif i>la*lo-1:
                n= n+[i-1, i+1, i-lo]
    	
    		# erste spalte
            elif col_i==0:
                n= n+[i+lo-1, i+1, i-lo, i+lo]
    	
    		# letzt spalte
            elif col_i ==lo-1:
                n= n+[i-1, i-lo+1, i-lo, i+lo]
    	
    		# nichts davon
            else:
                n = n+[i-1, i+1, i-lo, i+lo]
            return n
        
        for t in range(n_between+1):
            direct_n = find_neighboors(i, lo)
            if t == 0:
                neighb.append(direct_n)
            if t == 1:
                for n in direct_n:
                    ind_n = find_neighboors(n, lo)
                    neighb.append(ind_n)
        n = list(set(flatten(neighb)))
        if i in n:
            n.remove(i)
        
	
	#=====================
	# Criteria 2: must be all at least once be significanlty correlated 
	#=====================	
        m =[]
        for j in n:
            if j in indices_not_masked:
                m = m+[j]
		
		# now m contains the potential neighbors of gridpoint i

	
	#=====================	
	# Criteria 3: sign must be the same for each step 
	#=====================				
        l=[]
	
        cc_i = A.data[i]
        cc_i_sign = np.sign(cc_i)
		
	
        for k in m:
            cc_k = A.data[k]
            cc_k_sign = np.sign(cc_k)
		

            if cc_i_sign *cc_k_sign == 1:
                l = l +[k]

            else:
                l = l
			
            if len(l)==0:
                l =[]
                A.mask[i]=True	
    			
            elif i not in l: 
                l = l + [i]	
		
		
            N_pot[i]=N_pot[i] + l	



	#========================================	
	# STEP 3: merge overlapping set of neighbors
	#========================================
    Regions = merge_neighbors(N_pot)
	
	#========================================
	# STEP 4: assign a value to each region
	#========================================
	

	# 2) combine 1A+1B 
    B = np.abs(A)
	
	# 3) calculate the area size of each region	
#    Area =  [[] for i in range(len(Regions))]
#	
#    for i in range(len(Regions)):
#        indices = np.array(list(Regions[i]))
#        indices_lat_position = indices//lo
#        lat_nodes = lats[indices_lat_position[:]]
#        cos_nodes = np.cos(np.deg2rad(lat_nodes))		
#		
#        area_i = [np.sum(cos_nodes)]
#        Area[i]= Area[i]+area_i
    
    Area =  [[] for i in range(len(Regions))]
    A_gs_flat = np.reshape(A_gs, -1)
    for i in range(len(Regions)):
        indices = np.array(list(Regions[i]))
        Area[i] = np.sum(A_gs_flat[indices])
    
	
	#---------------------------------------
	# OPTIONAL: Exclude regions which only consist of less than n nodes
	# 3a)
	#---------------------------------------	
	
    # keep only regions which are larger then the mean size of the regions
    if ex['min_perc_area'] == 'mean':
        min_area = np.mean(Area) # mean area of all regions
    else:
        min_area = (ex['min_perc_area']/100.) * np.sum(A_gs) 
    
    R=[]
    Ar=[]
    for i in range(len(Regions)):
        if Area[i] > min_area: # area of regions must be larger then min_area
            Ar.append(Area[i])
            R.append(Regions[i])
            
	
    Regions = R
    Area = Ar	
	
	
	
	# 4) calcualte region value:
	
    C = np.zeros(len(Regions))
	
    Area = np.array(Area)
    for i in range(len(Regions)):
        C[i]=Area[i]*np.mean(B[list(Regions[i])])


	
	
	# mask out those nodes which didnot fullfill the neighborhood criterias
    A.mask[A==0] = True	
		
		
	#========================================
	# STEP 5: rank regions by region value
	#========================================
	
	# rank indices of Regions starting with strongest:
    sorted_region_strength = np.argsort(C)[::-1]
	
	# give ranking number
	# 1 = strongest..
	# 2 = second strongest
    
    # create clean array
    Regions_lag_i = np.zeros(A.data.shape)
    for i in range(len(Regions)):
        j = list(sorted_region_strength)[i]
        Regions_lag_i[list(Regions[j])]=i+1
    
    Regions_lag_i = np.array(Regions_lag_i, dtype=int)
    #%%
    return Regions_lag_i



def Ev_timeseries(xarray, threshold, ex, grouped=False):  
    #%%
    tfreq_RVts = pd.Timedelta((xarray.time[1]-xarray.time[0]).values)
    min_dur = ex['min_dur'] ; max_break = ex['max_break']  + 1
    min_dur = pd.Timedelta(min_dur, 'd') / tfreq_RVts
    max_break = pd.Timedelta(max_break, 'd') / tfreq_RVts
    if threshold >= xarray.mean(dim='time'):
        Ev_ts = xarray.where( xarray.values > threshold) 
    else:
        Ev_ts = xarray.where( xarray.values < threshold) 
    Ev_dates = Ev_ts.dropna(how='all', dim='time').time
    events_idx = [list(xarray.time.values).index(E) for E in Ev_dates.values]
    n_timesteps = Ev_ts.size
    
    peak_o_thresh = Ev_binary(events_idx, n_timesteps, min_dur, max_break, grouped)
    
    dur = np.zeros( (peak_o_thresh.size) )
    for i in np.arange(1, max(peak_o_thresh)+1):
        size = peak_o_thresh[peak_o_thresh==i].size
        dur[peak_o_thresh==i] = size

    if np.sum(peak_o_thresh) < 1:
        Events = Ev_ts.where(peak_o_thresh > 0 ).dropna(how='all', dim='time').time
        pass
    else:
        peak_o_thresh[peak_o_thresh == 0] = np.nan
        Ev_labels = xr.DataArray(peak_o_thresh, coords=[Ev_ts.coords['time']])
        Ev_dates = Ev_labels.dropna(how='all', dim='time').time
        
#        Ev_dates = Ev_ts.time.copy()     
#        Ev_dates['Ev_label'] = Ev_labels    
#        Ev_dates = Ev_dates.groupby('Ev_label').max().values
#        Ev_dates.sort()
        Events = xarray.sel(time=Ev_dates)
    
    #%%
    return Events, dur

def Ev_binary(events_idx, n_timesteps, min_dur, max_break, grouped=False):
    
    max_break = max_break + 1
    peak_o_thresh = np.zeros((n_timesteps))
    ev_num = 1
    # group events inter event time less than max_break
    for i in range(len(events_idx)):
        if i < len(events_idx)-1:
            curr_ev = events_idx[i]
            next_ev = events_idx[i+1]
        elif i == len(events_idx)-1:
            curr_ev = events_idx[i]
            next_ev = events_idx[i-1]
                 
        if abs(next_ev - curr_ev) <= max_break:
            peak_o_thresh[curr_ev] = ev_num
        elif abs(next_ev - curr_ev) > max_break:
            peak_o_thresh[curr_ev] = ev_num
            ev_num += 1

    # remove events which are too short
    for i in np.arange(1, max(peak_o_thresh)+1):
        No_ev_ind = np.where(peak_o_thresh==i)[0]
        # if shorter then min_dur, then not counted as event
        if No_ev_ind.size < min_dur:
            peak_o_thresh[No_ev_ind] = 0
    
    if grouped == True:
        data = np.concatenate([peak_o_thresh[:,None],
                               np.arange(len(peak_o_thresh))[:,None]],
                                axis=1)
        df = pd.DataFrame(data, index = range(len(peak_o_thresh)), 
                                  columns=['values', 'idx'], dtype=int)
        grouped = df.groupby(df['values']).mean().values.squeeze()[1:]            
        peak_o_thresh[:] = 0
        peak_o_thresh[np.array(grouped, dtype=int)] = 1
    else:
        pass
    
    return peak_o_thresh

def timeseries_tofit_bins(xarray, ex):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
    
    seldays_pp = pd.date_range(start=one_yr[0], end=one_yr[-1], 
                                freq=(datetime[1] - datetime[0]))
    end_day = one_yr.max() 
    # after time averaging over 'tfreq' number of days, you want that each year 
    # consists of the same day. For this to be true, you need to make sure that
    # the selday_pp period exactly fits in a integer multiple of 'tfreq'
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
    fit_steps_yr = (end_day - seldays_pp.min() + np.timedelta64(1, 'D'))  / temporal_freq
    # line below: The +1 = include day 1 in counting
    start_day = (end_day - (temporal_freq * np.round(fit_steps_yr, decimals=0))) + 1 
    
    def make_datestr_2(datetime, start_yr):
        breakyr = datetime.year.max()
        datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
        nyears = (datetime.year[-1] - datetime.year[0])+1
        startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
        endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
        firstyear = startday[:4]
        datesdt = start_yr
        def plusyearnoleap(curr_yr, startday, endday, incr):
            startday = startday.replace(firstyear, str(curr_yr+incr))
            endday = endday.replace(firstyear, str(curr_yr+incr))
            next_yr = pd.date_range(start=startday, end=endday, 
                            freq=(datetime[1] - datetime[0]))
            # excluding leap year again
            noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
            next_yr = next_yr[noleapdays].dropna(how='all')
            return next_yr
        
        for yr in range(0,nyears-1):
            curr_yr = yr+datetime.year[0]
            next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
            datesdt = np.append(datesdt, next_yr)
#            print(len(next_yr))
#            nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
#            datesstr = datesstr + nextstr
#            print(nextstr[0])
            
            upd_start_yr = plusyearnoleap(next_yr.year[0], startday, endday, 1)

            if next_yr.year[0] == breakyr:
                break
        datesdt = pd.to_datetime(datesdt)
        return datesdt, upd_start_yr
    
    start_yr = pd.date_range(start=start_day, end=end_day, 
                                freq=(datetime[1] - datetime[0]))
    # exluding leap year from cdo select string
    noleapdays = (((start_yr.month==2) & (start_yr.day==29))==False)
    start_yr = start_yr[noleapdays].dropna(how='all')
    datesdt, next_yr = make_datestr_2(datetime, start_yr)
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    startdatestr = '{} {}'.format(start_day.day, months[start_day.month])
    enddatestr   = '{} {}'.format(end_day.day, months[end_day.month])
    print('adjusted time series to fit bins: \nFrom {} to {}'.format(
                startdatestr, enddatestr))
    adj_array = xarray.sel(time=datesdt)
    return adj_array, datesdt
    

def time_mean_bins(xarray, ex):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
    
    if one_yr.size % ex['tfreq'] != 0:
        possible = []
        for i in np.arange(1,20):
            if 214%i == 0:
                possible.append(i)
        print('Error: stepsize {} does not fit in one year\n '
                         ' supply an integer that fits {}'.format(
                             ex['tfreq'], one_yr.size))   
        print('\n Stepsize that do fit are {}'.format(possible))
        print('\n Will shorten the \'subyear\', so that the temporal'
              ' frequency fits in one year')
        xarray, datetime = timeseries_tofit_bins(xarray, ex)
        one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
          
    else:
        pass
    fit_steps_yr = (one_yr.size)  / ex['tfreq']
    bins = list(np.repeat(np.arange(0, fit_steps_yr), ex['tfreq']))
    n_years = np.unique(datetime.year).size
    for y in np.arange(1, n_years):
        x = np.repeat(np.arange(0, fit_steps_yr), ex['tfreq'])
        x = x + fit_steps_yr * y
        [bins.append(i) for i in x]
    label_bins = xr.DataArray(bins, [xarray.coords['time'][:]], name='time')
    label_dates = xr.DataArray(xarray.time.values, [xarray.coords['time'][:]], name='time')
    xarray['bins'] = label_bins
    xarray['time_dates'] = label_dates
    xarray = xarray.set_index(time=['bins','time_dates'])
    
    half_step = ex['tfreq']/2.
    newidx = np.arange(half_step, datetime.size, ex['tfreq'], dtype=int)
    newdate = label_dates[newidx]
    

    group_bins = xarray.groupby('bins').mean(dim='time', keep_attrs=True)
    group_bins['bins'] = newdate.values
    dates = pd.to_datetime(newdate.values)
    return group_bins.rename({'bins' : 'time'}), dates


def make_datestr(dates, ex, startyr, endyr, lpyr=False):
    
    sstartdate = str(startyr) + '-' + ex['startperiod']
    senddate   = str(startyr) + '-' + ex['endperiod']
    
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

def import_array(filename, ex):
    
    ds = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)
    variables = list(ds.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    common_fields = ' time time_bnds longitude latitude lev lon lat level mask '
    var = [var for var in strvars if var not in common_fields][0]
    var = var.replace(' ', '')
    
    ds = ds[var].squeeze()
    if 'time' in ds.dims:
        numtime = ds['time']
        dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
        dates = pd.to_datetime(dates)
        dates -= pd.Timedelta(dates.hour[0], unit='h')
        
        ds['time'] = dates
    return ds

def import_ds_lazy(filename, ex, loadleap=False, seldates=None):
    ds = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)
    variables = list(ds.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    common_fields = ' time time_bnds longitude latitude lev lon lat level mask '
    var = [var for var in strvars if var not in common_fields][0]
    var = var.replace(' ', '')

    ds = ds[var].squeeze()
    if 'latitude' and 'longitude' not in ds.dims:
        ds = ds.rename({'lat':'latitude',
                   'lon':'longitude'})
    if 'la_max' in ex.keys() and 'la_min' in ex.keys():
        if ds.latitude[0] > ds.latitude[1]:
            slice_ = slice(ex['la_max'], ex['la_min'])
        else:
            slice_ = slice(ex['la_min'], ex['la_max'])
        ds = ds.sel(latitude=slice_)
    if 'lo_max' in ex.keys() and 'lo_min' in ex.keys():
        ds = ds.sel(longitude=slice(ex['lo_min'], ex['lo_max']))
        
    # get dates
    numtime = ds['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])

    if numtime.attrs['calendar'] != 'gregorian':
        dates = [d.strftime('%Y-%m-%d') for d in dates]
    if 'input_freq' in ex.keys():
        if ex['input_freq'] == 'monthly':
            dates = [d.replace(day=1,hour=0) for d in pd.to_datetime(dates)]
            ex['n_oneyr'] = np.unique(pd.to_datetime(dates).month).size
    else:
        dates = pd.to_datetime(dates)
        stepsyr = dates.where(dates.year == dates.year[0]).dropna(how='all')
        test_if_fullyr = np.logical_and(dates[stepsyr.size-1].month == 12,
                                    dates[stepsyr.size-1].day == 31)
        assert test_if_fullyr, ('full is needed as raw data since rolling'
                            ' mean is applied across timesteps')

    dates = pd.to_datetime(dates)
    # set hour to 00
    if dates.hour[0] != 0:
        dates -= pd.Timedelta(dates.hour[0], unit='h')

    ds['time'] = dates
    
    if type(seldates)==type(None):
        pass
    else:
        ds = ds.sel(time=seldates)

    if loadleap==False:
        # mask away leapdays
        dates_noleap = remove_leapdays(pd.to_datetime(ds.time.values))
        ds = ds.sel(time=dates_noleap)
    return ds

def remove_leapdays(datetime):
    mask_lpyrfeb = np.logical_and((datetime.month == 2), (datetime.day == 29))

    dates_noleap = datetime[mask_lpyrfeb==False]
    return dates_noleap
    
def area_weighted(xarray):
    # Area weighted, taking cos of latitude in radians     
    coslat = np.cos(np.deg2rad(xarray.coords['latitude'].values)).clip(0., 1.)
    area_weights = np.tile(coslat[..., np.newaxis],(1,xarray.longitude.size))
#    xarray.values = xarray.values * area_weights 

    return xr.DataArray(xarray.values * area_weights, coords=xarray.coords, 
                           dims=xarray.dims)
    
def convert_longitude(data, to_format='west_east'):
    import numpy as np
    import xarray as xr
    if to_format == 'west_east':
        lon_above = data.longitude[np.where(data.longitude > 180)[0]]
        lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
        # roll all values to the right for len(lon_above amount of steps)
        data = data.roll(longitude=len(lon_above))
        # adapt longitude values above 180 to negative values
        substract = lambda x, y: (x - y)
        lon_above = xr.apply_ufunc(substract, lon_above, 360)
        if lon_normal.size != 0:
            if lon_normal[0] == 0.:
                convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')
            
            else:
                convert_lon = xr.concat([lon_normal, lon_above], dim='longitude')
        else:
            convert_lon = lon_above

    elif to_format == 'only_east':
        lon_above = data.longitude[np.where(data.longitude >= 0)[0]]
        lon_below = data.longitude[np.where(data.longitude < 0)[0]]
        lon_below += 360
        data = data.roll(longitude=len(lon_below))
        convert_lon = xr.concat([lon_above, lon_below], dim='longitude')
    data['longitude'] = convert_lon
    return data


def rolling_mean_xr(xarray, win):
    closed = int(win/2)
    flatarray = xarray.values.flatten()
    ext_array = np.insert(flatarray, 0, flatarray[-closed:])
    ext_array = np.insert(ext_array, 0, flatarray[:closed])
    
    df = pd.DataFrame(ext_array)
#    std = xarray.where(xarray.values!=0.).std().values
#    scipy.signal.gaussian(win, std)
    rollmean = df.rolling(win, center=True, 
                          win_type='gaussian').mean(std=win/2.).dropna()
    
    # replace values with smoothened values
    new_xarray = xarray.copy()
    new_values = np.reshape(rollmean.squeeze().values, xarray.shape)
    # ensure LSM mask
    mask = np.array((xarray.values!=0.),dtype=int)
    new_xarray.values = (new_values * mask)

    return new_xarray

def rolling_mean_time(xarray_or_file, ex, center=True):
    #%%
#    xarray_or_file = Prec_reg
#    array = np.zeros(60)
#    array[-30:] = 1
#    xarray_or_file = xr.DataArray(array, coords=[np.arange(60)], dims=['time'])
    
    if type(xarray_or_file) == str:
        file_path = os.path.join(ex['path_pp'], xarray_or_file)        
        ds = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
        ds_rollingmean = ds.rolling(time=ex['rollingmean'][1], center=center, min_periods=1).mean()
        
        new_fname = 'rm{}_'.format(ex['rollingmean'][1]) + xarray_or_file
        file_path = os.path.join(ex['path_pp'], new_fname)
        ds_rollingmean.to_netcdf(file_path, mode='w')
        print('saved netcdf as {}'.format(new_fname))
        print('functions returning None')
        xr_rolling_mean = None
    else:
        # Taking rolling window mean with the closing on the right, 
        # meaning that we are taking the mean over the past at the index/label 

        if xarray_or_file.ndim == 1:
            xr_rolling_mean = xarray_or_file.rolling(time=ex['rollingmean'][1], center=True, 
                                                         min_periods=1).mean()
        else: 
            xr_rolling_mean = xarray_or_file.rolling(time=ex['rollingmean'][1], center=True, 
                                                         min_periods=1).mean(dim='time', skipna=True)
        if 'latitude' in xarray_or_file.dims:
            lat = 35
            lon = 360-20
            
            def find_nearest(array, value):
                idx = (np.abs(array - value)).argmin()
                return int(idx)
            
            lat_idx = find_nearest(xarray_or_file['latitude'], lat)
            lon_idx = find_nearest(xarray_or_file['longitude'], lon)
            
            
            singlegc = xarray_or_file.isel(latitude=lat_idx, 
                                          longitude=lon_idx) 
        else:
            singlegc = xarray_or_file

        if type(singlegc) == type(xr.Dataset()):
            singlegc = singlegc.to_array().squeeze()
        
        year = 2012
        singlegc_oneyr = singlegc.where(singlegc.time.dt.year == year).dropna(dim='time', how='all')
        dates = pd.to_datetime(singlegc_oneyr.time.values)
        plt.figure(figsize=(10,6))
        plt.plot(dates, singlegc_oneyr.squeeze())
        if 'latitude' in xarray_or_file.dims:
            singlegc = xr_rolling_mean.isel(latitude=lat_idx, 
                                          longitude=lon_idx) 
        else:
            singlegc = xr_rolling_mean
        if type(singlegc) == type(xr.Dataset()):
            singlegc = singlegc.to_array().squeeze()
        singlegc_oneyr = singlegc.where(singlegc.time.dt.year == year).dropna(dim='time', how='all')
        plt.plot(dates, singlegc_oneyr.squeeze())
    #%%
    return xr_rolling_mean


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
    elif region ==  'PDO':
        west_lon = -250; east_lon = -110; south_lat = 20; north_lat = 70
#    elif region == 'for_soil':
        

    region_coords = [west_lon, east_lon, south_lat, north_lat]
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

    return all_values, region_coords


def cross_correlation_patterns(full_timeserie, pattern):
#%%
#    full_timeserie = var_train_reg
#    pattern = ds_Sem['pattern_CPPA'].sel(lag=lag)
    mask = np.ma.make_mask(np.isnan(pattern.values)==False)
    
    n_time = full_timeserie.time.size
    n_space = pattern.size
    

#    mask_pattern = np.tile(mask_pattern, (n_time,1))
    # select only gridcells where there is not a nan
    full_ts = np.nan_to_num(np.reshape( full_timeserie.values, (n_time, n_space) ))
    pattern = np.nan_to_num(np.reshape( pattern.values, (n_space) ))

    mask_pattern = np.reshape( mask, (n_space) )
    full_ts = full_ts[:,mask_pattern]
    pattern = pattern[mask_pattern]
    
#    crosscorr = np.zeros( (n_time) )
    spatcov   = np.zeros( (n_time) )
#    covself   = np.zeros( (n_time) )
#    corrself  = np.zeros( (n_time) )
    for t in range(n_time):
        # Corr(X,Y) = cov(X,Y) / ( std(X)*std(Y) )
        # cov(X,Y) = E( (x_i - mu_x) * (y_i - mu_y) )
#        crosscorr[t] = np.correlate(full_ts[t], pattern)
        M = np.stack( (full_ts[t], pattern) )
        spatcov[t] = np.cov(M)[0,1] #/ (np.sqrt(np.cov(M)[0,0]) * np.sqrt(np.cov(M)[1,1]))
#        sqrt( Var(X) ) = sigma_x = std(X)
#        spatcov[t] = np.cov(M)[0,1] / (np.std(full_ts[t]) * np.std(pattern))        
#        covself[t] = np.mean( (full_ts[t] - np.mean(full_ts[t])) * (pattern - np.mean(pattern)) )
#        corrself[t] = covself[t] / (np.std(full_ts[t]) * np.std(pattern))
    dates_test = full_timeserie.time
#    corrself = xr.DataArray(corrself, coords=[dates_test.values], dims=['time'])
    
#    # standardize
#    corrself -= corrself.mean(dim='time', skipna=True)
    
    # cov xarray
    spatcov = xr.DataArray(spatcov, coords=[dates_test.values], dims=['time'])
#%%
    return spatcov

def kornshell_with_input(args, ex):
#    stopped working for cdo commands
    '''some kornshell with input '''
#    args = [anom]
    import os
    import subprocess
#    cwd = os.getcwd()
    # Writing the bash script:
    new_bash_script = os.path.join('/Users/semvijverberg/surfdrive/Scripts/CPPA/CPPA/', "bash_script.sh")
#    arg_5d_mean = 'cdo timselmean,5 {} {}'.format(infile, outfile)
    #arg1 = 'ncea -d latitude,59.0,84.0 -d longitude,-95,-10 {} {}'.format(infile, outfile)
    
    bash_and_args = [new_bash_script]
    [bash_and_args.append(arg) for arg in args]
    with open(new_bash_script, "w") as file:
        file.write("#!/bin/sh\n")
        file.write("echo bash script output\n")
        for cmd in range(len(args)):

            print(args[cmd])
            file.write("${}\n".format(cmd+1)) 
    p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                         stderr=subprocess.STDOUT)
                         
    out = p.communicate()
    print(out[0].decode())
    return


def grouping_regions_similar_coords(l_ds, ex, grouping = 'group_accros_tests_single_lag', eps=10):
    '''Regions with similar coordinates are grouped together.
    lower eps indicate higher density necessary to form a cluster. 
    If eps too high, no high density is required and originally similar regions 
    are assigned into different clusters. If eps is too low, such an high density is required that
    it will also start to cluster together regions with similar coordinates.
    '''
    #%%
#    if ex['n_conv'] < 30:
#        grouping = 'group_across_test_and_lags'
#    grouping = 'group_accros_tests_single_lag'
#    grouping =  'group_across_test_and_lags'
    # Precursor Regions Dimensions
    all_lags_in_exp = l_ds[0]['pat_num_CPPA'].sel(lag=ex['lags']).lag.values
    lags_ind        = np.reshape(np.argwhere(all_lags_in_exp == ex['lags']), -1)
    PRECURSOR_DATA = np.array([data['pat_num_CPPA'].values for data in l_ds])
    PRECURSOR_VALUES = np.array([data['pattern_CPPA'].values for data in l_ds])
    PRECURSOR_DATA = PRECURSOR_DATA[:,lags_ind]
    PRECURSOR_LONGITIUDE = l_ds[0]['pat_num_CPPA'].longitude.values
    PRECURSOR_LATITUDE = l_ds[0]['pat_num_CPPA'].latitude.values
    PRECURSOR_LAGS = l_ds[0]['pat_num_CPPA'].sel(lag=ex['lags']).lag.values
    PRECURSOR_N_TEST_SETS = ex['n_conv']
    
    # Precursor Grid (to calculate precursor region centre coordinate)
    PRECURSOR_GRID = np.zeros((len(PRECURSOR_LATITUDE), len(PRECURSOR_LONGITIUDE), 2))
    PRECURSOR_GRID[..., 0], PRECURSOR_GRID[..., 1] = np.meshgrid(PRECURSOR_LONGITIUDE, PRECURSOR_LATITUDE)
    

    precursor_coordinates = []
    
    # Array Containing Precursor Region Indices for each YEAR 
    precursor_indices = np.empty((PRECURSOR_N_TEST_SETS,
                                  len(PRECURSOR_LAGS),
                                  len(PRECURSOR_LATITUDE),
                                  len(PRECURSOR_LONGITIUDE)),
                                 np.float32)
    precursor_indices[:,:,:,:] = np.nan
    precursor_indices_new = precursor_indices.copy()
    
    ex['uniq_regs_lag'] = np.zeros(len(PRECURSOR_LAGS))
    # Array Containing Precursor Region Weights for each YEAR and LAG
    for lag_idx, lag in enumerate(PRECURSOR_LAGS):
        
        indices_across_yrs = np.squeeze(PRECURSOR_DATA[:,lag_idx,:,:])
        
        
        if grouping == 'group_accros_tests_single_lag':
            # regions are given same number across test set, not accross all lags
            precursor_coordinates = []
    
    #    precursor_weights = np.zeros_like(precursor_indices, np.float32)
        min_samples = []
        for test_idx in range(PRECURSOR_N_TEST_SETS):
            indices = indices_across_yrs[test_idx, :, :]
            min_samples.append( np.nanmax(indices) )
            for region_idx in np.unique(indices[~np.isnan(indices)]):
            # evaluate regions
    #            plt.figure()
    #            plt.imshow(indices == region_idx)
                region = precursor_indices[test_idx, lag_idx, indices == region_idx]
                size_reg = region.size
                sign_reg = np.sign(PRECURSOR_VALUES[test_idx, lag_idx, indices == region_idx])
                precursor_indices[test_idx, lag_idx, indices == region_idx] = int(region_idx)
                if sign_reg.mean() == 1:
                    lon_lat = PRECURSOR_GRID[indices == region_idx].mean(0)
                elif sign_reg.mean() == -1:
                    lon_lat = PRECURSOR_GRID[indices == region_idx].mean(0) * -1
                if np.isnan(lon_lat).any()==False:

                    precursor_coordinates.append((
                        [test_idx, lag_idx, int(region_idx)], lon_lat, size_reg))
    
        # Group Similar Precursor Regions Together across years for same lag
        precursor_coordinates_index = np.array([index for index, coord, size in precursor_coordinates])
        precursor_coordinates_coord = np.array([coord for index, coord, size in precursor_coordinates])
        precursor_coordinates_weight = np.array([size for index, coord, size in precursor_coordinates])
        
        if grouping == 'group_accros_tests_single_lag':
            # min_samples to form core cluster, lower eps indicate higher density necessary to form a cluster.
            min_s = min_s = ex['n_conv'] * len(ex['lags']) / 2 #np.nanmin(min_samples)
            precursor_coordinates_group = DBSCAN(min_samples=ex['n_conv'] * 0.4, eps=eps).fit_predict(
                    precursor_coordinates_coord, sample_weight = precursor_coordinates_weight) + 2
            
            for (year_idx, lag_idx, region_idx), group in zip(precursor_coordinates_index, precursor_coordinates_group):
                precursor_indices_new[year_idx, lag_idx, precursor_indices[year_idx, lag_idx] == region_idx] = group

    
    if grouping == 'group_across_test_and_lags':
        # Group Similar Precursor Regions Together
        min_s = ex['n_conv'] * len(ex['lags']) / 2#np.nanmax(PRECURSOR_DATA)
        precursor_coordinates_index = np.array([index for index, coord, size in precursor_coordinates])
        precursor_coordinates_coord = np.array([coord for index, coord, size in precursor_coordinates])
        precursor_coordinates_weight = np.array([size for index, coord, size in precursor_coordinates])

        precursor_coordinates_group = DBSCAN(min_samples=min_s, eps=eps).fit_predict(
                precursor_coordinates_coord, sample_weight = precursor_coordinates_weight) + 2
        
        
        precursor_indices_new = np.zeros_like(precursor_indices)
        for (year_idx, lag_idx, region_idx), group in zip(precursor_coordinates_index, precursor_coordinates_group):
    #        print(year_idx, lag_idx, region_idx, group)
            precursor_indices_new[year_idx, lag_idx, precursor_indices[year_idx, lag_idx] == region_idx] = group
        precursor_indices_new[precursor_indices_new==0.] = np.nan
        
    # couting groups
    counting = {}
    for r in precursor_coordinates_group:
        c = list(precursor_coordinates_group).count(r)
        counting[r] =c
    # sort by counts:
    order_count = dict(sorted(counting.items(), key = 
             lambda kv:(kv[1], kv[0]), reverse=True))
    
    precursor_indices_new_ord = np.zeros_like(precursor_indices)
#    if grouping == 'group_across_test_and_lags':
    for i, r in enumerate(order_count.keys()):
        precursor_indices_new_ord[precursor_indices_new==r] = i+1
    precursor_indices_new_ord[precursor_indices_new_ord==0.] = np.nan
    # replace values in PRECURSOR_DATA|
    PRECURSOR_DATA[:,:,:,:] = precursor_indices_new_ord[:,:,:,:]
#    else:
#        PRECURSOR_DATA[:,:,:,:] = precursor_indices_new
    
    ex['uniq_regs_lag'][lag_idx] = max(np.unique(precursor_indices_new_ord) )
    
    
    l_ds_new = []
    for test_idx in range(PRECURSOR_N_TEST_SETS):
        single_ds = l_ds[test_idx].copy()
        pattern   = single_ds['pat_num_CPPA'].copy()
        
        pattern.values = PRECURSOR_DATA[test_idx]
#        # set the rest to nan
#        pattern = pattern.where(pattern.values != 0.)
        single_ds['pat_num_CPPA_clust'] = pattern
    #    print(test_idx)
    #    plt.figure()
    #    single_ds['pat_num_CPPA'][0].plot()
        # overwrite ds
        l_ds_new.append( single_ds )
    #    plt.figure()
    #    l_ds_new[-1]['pat_num_CPPA'][0].plot()
    
    #plt.figure()
    ex['max_N_regs'] = int(np.nanmax(PRECURSOR_DATA))
    #%%
    return l_ds_new, ex



def plot_precursor_regions(l_ds, n_tests, key_pattern_num, lags, subtitles, ex):
    #%%
    import seaborn as sns
    if len(lags) >= 2:
        adjust_vert_cbar = 0.0
    elif len(lags) < 2:
        adjust_vert_cbar = -0.06

    subfolder = os.path.join('', 'intermediate_results')
    
    lats = l_ds[0].latitude
    lons = l_ds[0].longitude
    array = np.zeros( (len(l_ds), len(lags), len(lats), len(lons)) )
    pattern_num = xr.DataArray(data=array, coords=[range(len(l_ds)), lags, lats, lons], 
                      dims=['n_tests', 'lag','latitude','longitude'], 
                      name=key_pattern_num, attrs={'units':'Precursor Region labels'})
    
    reg_labels = []
    for n in np.linspace(0, ex['n_conv']-1, n_tests, dtype=int): 
        pattern_num[n] = l_ds[n][key_pattern_num].sel(lag=lags) 
        for i,l in enumerate(lags):
            labels = np.unique(pattern_num[n].sel(lag=l))
            labels = labels[~np.isnan(labels)]            
            reg_labels.append( labels )
    counting = {}
    for r in np.unique(flatten(reg_labels)):
        key = flatten(reg_labels).count(r)
        counting[key] = r
    ex['max_N_regs'] = 1
    for k in counting.keys():
        if k < 0.5 * ex['n_conv'] * len(lags):
            ex['max_N_regs'] = int(counting[k]) + 1   
            break
    
    for n in np.linspace(0, ex['n_conv']-1, n_tests, dtype=int): 
        years = ex['tested_yrs']
        yr = years[n]
        for_plt = pattern_num[n]
        file_name = '{}_{}_{}_{}'.format(key_pattern_num, lags, n, yr )
        filename = os.path.join(subfolder, file_name.replace(
                                ' ','_')+'.png')
        for_plt.attrs['title'] = ('Precursor Regions - test yr(s): {}'.format(yr ))
#        for_plt = for_plt.where(for_plt.values <= ex['max_N_regs'])
        mask_noise = np.nan_to_num(for_plt.values) >=  ex['max_N_regs']
        for_plt.values[mask_noise] = ex['max_N_regs']
#        ex['max_N_regs'] = int(pattern_num.max()) +1
#        from matplotlib.colors import ListedColormap
#        cmap = ListedColormap(sns.color_palette("Paired", ex['max_N_regs']))
        cmap = plt.cm.tab20
        for_plt.values = for_plt.values-0.5
        
#        if 'max_N_regs' not in ex.keys() or key_pattern_num == 'pat_num_CPPA':
#        ex['max_N_regs'] = int(pattern_num.max() + 0.5)

        kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                       'steps' : ex['max_N_regs']+1, 'subtitles': None,
                       'vmin' : 0, 'vmax' : ex['max_N_regs'], 
                       'cmap' : cmap, 'column' : 1,
                       'cbar_vert' : adjust_vert_cbar, 'cbar_hght' : 0.0,
                       'adj_fig_h' : 1., 'adj_fig_w' : 1., 
                       'hspace' : 0.2, 'wspace' : 0.08,
                       'cticks_center' : True} )
        
        plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
    #%%
    return

def get_area(ds):
    longitude = ds.longitude
    latitude = ds.latitude
    
    Erad = 6.371e6 # [m] Earth radius
#    global_surface = 510064471909788
    # Semiconstants
    gridcell = np.abs(longitude[1] - longitude[0]).values # [degrees] grid cell size
    
    # new area size calculation:
    lat_n_bound = np.minimum(90.0 , latitude + 0.5*gridcell)
    lat_s_bound = np.maximum(-90.0 , latitude - 0.5*gridcell)
    
    A_gridcell = np.zeros([len(latitude),1])
    A_gridcell[:,0] = (np.pi/180.0)*Erad**2 * abs( np.sin(lat_s_bound*np.pi/180.0) - np.sin(lat_n_bound*np.pi/180.0) ) * gridcell
    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
#    A_mean = np.mean(A_gridcell2D)
    return A_gridcell2D
# =============================================================================
# =============================================================================
# Plotting functions
# =============================================================================
# =============================================================================
    
def extend_longitude(data):
    import xarray as xr
    import numpy as np
    plottable = xr.concat([data, data.sel(longitude=data.longitude[:1])], dim='longitude').to_dataset(name="ds")
    plottable["longitude"] = np.linspace(0,360, len(plottable.longitude))
    plottable = plottable.to_array(dim='ds')
    return plottable


def plot_earth(view="EARTH", kwrgs={'cen_lon':0}):
    #%%
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    # Create Big Figure
    plt.rcParams['figure.figsize'] = [18, 12]

    # create Projection and Map Elements
    projection = ccrs.PlateCarree(
            central_longitude=kwrgs['cen_lon'])
    ax = plt.axes(projection=projection)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.OCEAN, color="white")
    ax.add_feature(cfeature.LAND, color="lightgray")

    if view == "US":
        ax.set_xlim(-130, -65)
        ax.set_ylim(25, 50)
    elif view == "EAST US":
        ax.set_xlim(-105, -65)
        ax.set_ylim(25, 50)
    elif view == "EARTH":
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    #%%
    return projection, ax

def xarray_plot(data, path='default', name = 'default', saving=False):
    #%%
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    if type(data) == type(xr.Dataset()):
        data = data.to_array().squeeze()

    # some lon values > 180
    if len(data.longitude[np.where(data.longitude > 180)[0]]) != 0:
        # if 0 is in lon values
        if data.longitude.where(data.longitude==0).dropna(dim='longitude', how='all').size != 0.:
            print('hoi')   
            data = convert_longitude(data)
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
        lons = data.where(data.mask==True, drop=True).longitude
        lats = data.where(data.mask==True, drop=True).latitude
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
    import datetime
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

def plot_events_validation(pred1, pred2, obs, pt1, pt2, othreshold, test_year=None):
    #%%
#    pred1 = crosscorr_Sem
#    pred2 = crosscorr_mcK
#    obs = RV_ts_test
#    pt1 = Prec_threshold_Sem
#    pt2 = Prec_threshold_mcK
#    othreshold = ex['event_thres']
#    test_year = int(crosscorr_Sem.time.dt.year[0])

    
    def predyear(pred, obs):
        if str(type(test_year)) == "<class 'numpy.int64'>" or str(type(test_year)) == "<class 'int'>":
            predyear = pred.where(pred.time.dt.year == test_year).dropna(dim='time', how='any')
            obsyear  = obs.where(obs.time.dt.year == test_year).dropna(dim='time', how='any')
            predyear['time'] = obsyear.time
        elif type(test_year) == type(['list']):
            years_in_obs = list(obs.time.dt.year.values)
            test_years = [i for i in range(len(years_in_obs)) if years_in_obs[i] in test_year]
            # Warning this is wrong #!!!
            predyear = pred.isel(time=test_years)
            obsyear = obs.isel(time=test_years)
        else:
            predyear = pred
            predyear['time'] = obs.time
            obsyear = obs
        return predyear, obsyear
        
    predyear1, obsyear = predyear(pred1, obs)
    predyear2, obsyear = predyear(pred2, obs)

    eventdays = obsyear.where( obsyear.values > othreshold ) 
    eventdays = eventdays.dropna(how='all', dim='time').time
    preddays = predyear1.where(predyear1.values > pt1)
    preddays1 = preddays.dropna(how='all', dim='time').time
    preddays = predyear2.where(predyear2.values > pt2)
    preddays2 = preddays.dropna(how='all', dim='time').time
#    # standardize obsyear
#    othreshold -= obsyear.mean(dim='time', skipna=True).values
#    obsyear    -= obsyear.mean(dim='time', skipna=True)
#    
#    # standardize predyear(s)
#    pthreshold -= predyear.mean(dim='time', skipna=True).values
#    predyear    -= predyear.mean(dim='time', skipna=True)
      
    TP1 = [day for day in preddays1.time.values if day in list(eventdays.values)]
    TP2 = [day for day in preddays2.time.values if day in list(eventdays.values)]
#    pthreshold = ((pthreshold - pred1.mean()) * obsyear.std()/predyear.std()).values
#    predyear = (predyear) * obsyear.std()/predyear.std() 
    plt.figure(figsize = (10,5))
    ax1 = plt.subplot(311)
    ax1.plot(pd.to_datetime(obsyear.time.values), obsyear, label='observed',
             color = 'blue')
    ax1.axhline(y=othreshold, color='blue')
    for days in eventdays.time.values:
        ax1.axvline(x=pd.to_datetime(days), color='blue', alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(312)
    ax2.plot(pd.to_datetime(obsyear.time.values),predyear1, label='Sem pattern ts',
             color='red')
    ax2.axhline(y=pt1, color='red')
    for days in preddays1.time.values:
        ax2.axvline(x=pd.to_datetime(days), color='red', alpha=0.3)
    for days in pd.to_datetime(TP1):
        ax2.axvline(x=pd.to_datetime(days), color='green', alpha=1.)
    ax2.legend()
    # second prediction
    ax3 = plt.subplot(313)
    ax3.plot(pd.to_datetime(obsyear.time.values),predyear2, label='mcK pattern ts',
             color='red')
    ax3.axhline(y=pt2, color='red')
    for days in preddays2.time.values:
        ax3.axvline(x=pd.to_datetime(days), color='red', alpha=0.3)
    for days in pd.to_datetime(TP2):
        ax3.axvline(x=pd.to_datetime(days), color='green', alpha=1.)
    ax3.legend()


def plot_oneyr_events(xarray, ex, test_year, folder, saving=False):
    #%%
    if ex['event_percentile'] == 'std':
        # binary time serie when T95 exceeds 1 std
        threshold = xarray.mean(dim='time', skipna=True).values + xarray.std().values
    else:
        percentile = ex['event_percentile']
        threshold = np.percentile(xarray.values, percentile)
    
    testyear = xarray.where(xarray.time.dt.year == test_year).dropna(dim='time', how='any')
    freq = pd.Timedelta(testyear.time.values[1] - testyear.time.values[0])
    plotpaper = xarray.sel(time=pd.date_range(start=testyear.time.values[0], 
                                                end=testyear.time.values[-1], 
                                                freq=freq ))

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    plotpaper.plot(ax=ax, color='blue', linewidth=3, label='T95')
    plt.axhline(y=threshold, color='blue', linewidth=2 )
    plt.fill_between(plotpaper.time.values, threshold, plotpaper, where=(plotpaper.values > threshold),
                 interpolate=True, color="crimson", label="Events")
    ax.legend(fontsize='x-large', fancybox=True, facecolor='grey',
              frameon=True, framealpha=0.3)
    ax.set_title('Timeseries and events', fontsize=18)
    ax.set_ylabel('Temperature anomalies [K]', fontsize=15)
    ax.set_xlabel('')
    #%%
    if saving == True:
        filename = os.path.join(folder, 'ts_{}'.format(test_year))
        plt.savefig(filename+'.png', dpi=300)

def plotting_wrapper(plotarr, ex, filename=None,  kwrgs=None):
#    map_proj = ccrs.Miller(central_longitude=240)  
    folder_name = os.path.join(ex['figpathbase'], ex['exp_folder'])
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
    finalfigure(plotarr, file_name, kwrgs)
    

def finalfigure(xrdata, file_name, kwrgs):
    #%%
    map_proj = ccrs.PlateCarree(central_longitude=220)  
    lons = xrdata.longitude.values
    lats = xrdata.latitude.values
    strvars = [' {} '.format(var) for var in list(xrdata.dims)]
    
    var = [var for var in strvars if var not in ' longitude latitude '][0] 
    var = var.replace(' ', '')
    g = xr.plot.FacetGrid(xrdata, col=var, col_wrap=kwrgs['column'], sharex=True,
                      sharey=True, subplot_kws={'projection': map_proj},
                      aspect= (xrdata.longitude.size) / xrdata.latitude.size, size=3.5)
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
            plotdata = extend_longitude(xrdata[n_ax]).squeeze().drop('ds')
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
                    sigdata = extend_longitude(sigdata[n_ax]).squeeze().drop('ds')
                else:
                    sigdata = sigdata[n_ax].squeeze()
                sigdata.plot.contourf(ax=ax, levels=[0, 0.5, 1],
                           transform=ccrs.PlateCarree(), hatches=['...', ''],
                           colors='none', add_colorbar=False,
                           subplot_kws={'projection': map_proj})
            
        ax.coastlines(color='black', alpha=0.3, facecolor='grey')
        ax.add_feature(cfeature.LAND, facecolor='grey', alpha=0.3)
        
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], ccrs.PlateCarree())
        
        if 'contours' in kwrgs.keys():
            condata, con_levels = kwrgs['contours']
            if periodic == True:
                condata = extend_longitude(condata[n_ax]).squeeze().drop('ds')
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
                        sigdata = extend_longitude(sigdata[n_ax]).squeeze().drop('ds')
                    else:
                        sigdata = sigdata[n_ax].squeeze()
                    sigdata.plot.contourf(ax=ax, levels=[0, 0.5, 1],
                               transform=ccrs.PlateCarree(), hatches=['...', ''],
                               colors='none', add_colorbar=False,
                               subplot_kws={'projection': map_proj})
                                
                  
        
        if kwrgs['subtitles'] == None:
            pass
        else:
            fontdict = dict({'fontsize'     : 18,
                             'fontweight'   : 'bold'})
            ax.set_title(kwrgs['subtitles'][n_ax], fontdict=fontdict, loc='center')
        
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
            
        if map_proj.proj4_params['proj'] in ['merc', 'eqc']:
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
    if kwrgs['savefig'] != False:
        g.fig.savefig(file_name ,dpi=250, frameon=True)
    
    return




def figure_for_schematic(iter_regions, composite_p1, chunks, lats, lons, ex):
    #%%
    reg_all_1 = iter_regions[:len(chunks)]
    
    map_proj = ccrs.PlateCarree(central_longitude=220) 
    regions = np.reshape(reg_all_1, (reg_all_1.shape[0], lats.size, lons.size) )
    name_chnks = [str(chnk) for chnk in chunks]
    regions = xr.DataArray(regions, coords=[name_chnks, lats, lons], 
                           dims=['yrs_out', 'latitude', 'longitude'])
    folder = os.path.join(ex['figpathbase'], ex['CPPA_folder'], 'schematic_fig/')
    
    plots = 3
    subset = np.linspace(0,regions.yrs_out.size-1,plots, dtype=int)
    subset = [  1,  42, 80]
    regions = regions.isel(yrs_out=subset)
    regions  = regions.sel(latitude=slice(60.,0.))
    regions = regions.sel(longitude=slice(160, 250))
    
    if os.path.isdir(folder) != True : os.makedirs(folder)
    for i in range(len(subset))[::int(plots/3)]:
        i = int(i)
        cmap = plt.cm.YlOrBr
        fig = plt.figure(figsize = (14,9))
        ax = plt.subplot(111, projection=map_proj) 
        plotdata = regions.isel(yrs_out=i).where(regions.isel(yrs_out=i) > 0.)
        plotdata.plot.pcolormesh(ax=ax, cmap=cmap, vmin=0, vmax=plots,
                               transform=ccrs.PlateCarree(),
                               subplot_kws={'projection': map_proj},
                                add_colorbar=False, alpha=0.9)
        ax.coastlines(color='black', alpha=0.8, linewidth=2)
#        list_points = np.argwhere(regions.isel(yrs_out=i).values == 1)
#        x_co = regions.isel(yrs_out=i).longitude.values
#        y_co = regions.isel(yrs_out=i).latitude.values
#        for p in list_points:
#            ax.scatter(x_co[p[1]], y_co[p[0]], marker='$1$', color='black', 
#                      s=70, transform=ccrs.PlateCarree())
    #    ax.set_extent([-110, 150,0,80], crs=ccrs.PlateCarree())
#        ax.set_extent([140, 266,0,60])
#        ax.outline_patch.set_visible(False)
#        ax.background_patch.set_visible(False)
        ax.set_title('')
        title = str(plotdata.yrs_out.values) 
        t = ax.text(0.006, 0.008, 
                    'excl. {}'.format(title),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=30.37, weight='bold')
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='red'))
        
        fig.savefig(folder+title+'.pdf',  bbox_inches='tight')
#%%
    # figure robustness 1
    import matplotlib.patches as mpatches
    import cartopy.feature as cfeature
    lats_fig = slice(60.,5.)
    lons_fig = slice(165, 243)
    mask_final = ( np.sum(reg_all_1, axis=0) < int(ex['FCP_thres'] * len(chunks)))
    nparray_comp = np.reshape(np.nan_to_num(composite_p1.values), (composite_p1.size))
    Corr_Coeff = np.ma.MaskedArray(nparray_comp, mask=mask_final)
    lat_grid = composite_p1.latitude.values
    lon_grid = composite_p1.longitude.values

    # retrieve regions sorted in order of 'strength'
    # strength is defined as an area weighted values in the composite
    Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid, A_gs, ex)

    # reshape to latlon grid
    npmap = np.reshape(Regions_lag_i, (lats.size, lons.size))
    mask_strongest = (npmap!=0) 
    npmap[mask_strongest==False] = 0
    xrnpmap_init = composite_p1.copy()
    xrnpmap_init.values = npmap
    xrnpmap_init  = xrnpmap_init.sel(latitude=lats_fig)
    xrnpmap_init = xrnpmap_init.sel(longitude=lons_fig)
    mask_final   = xrnpmap_init!= 0.
    xrnpmap_init = xrnpmap_init.where(mask_final)    


    # renumber
    regions_label = np.unique(xrnpmap_init.values)[np.isnan(np.unique(xrnpmap_init.values))==False]
    for i in range(regions_label.size):
        r = regions_label[i]
        xrnpmap_init.values[xrnpmap_init.values==r] = i+1
        
    
    regions = np.reshape(reg_all_1, (reg_all_1.shape[0], lats.size, lons.size) )
    name_chnks = [str(chnk) for chnk in chunks]
    regions = xr.DataArray(regions, coords=[name_chnks, lats, lons], 
                           dims=['yrs_out', 'latitude', 'longitude'])
    regions  = regions.sel(latitude=lats_fig)
    regions = regions.sel(longitude=lons_fig)
    
    fig = plt.figure(figsize = (20,14))
    ax = plt.subplot(111, projection=map_proj)
    robustness = np.sum(regions,axis=0)
    n_max = robustness.max().values
    freq_rawprec = regions.isel(yrs_out=i).copy()
    freq_rawprec.values = robustness
    plotdata = plotdata.sel(latitude=lats_fig)
    plotdata = plotdata.sel(longitude=lons_fig)
    npones = np.ones( (plotdata.shape) )
    npones[mask_final.values==True] = 0
    plotdata.values = npones
    plotdata = plotdata.where(freq_rawprec.values > 0.)
    cmap = colors.ListedColormap(['lemonchiffon' ])
    plotdata.where(mask_final.values==False).plot.pcolormesh(ax=ax, cmap=cmap,
                               transform=ccrs.PlateCarree(), vmin=0, vmax=plots,
                               subplot_kws={'projection': map_proj},
                                add_colorbar=False, alpha=0.3)
    
#    freq_rawprec.plot.contour(ax=ax, 
#                               transform=ccrs.PlateCarree(), linewidths=3,
#                               colors=['black'], levels=[0., (ex['FCP_thres'] * n_max)-1, n_max],
#                               subplot_kws={'projection': map_proj},
#                               )
    
    n_regs = xrnpmap_init.max().values
    xrnpmap_init.values = xrnpmap_init.values - 0.5
    kwrgs = dict( {        'steps' : n_regs+1, 
                           'vmin' : 0, 'vmax' : n_regs, 
                           'cmap' : plt.cm.tab20, 
                           'cticks_center' : True} )
    
    cmap = colors.ListedColormap(['cyan', 'green', 'purple' ])
    clevels = np.linspace(kwrgs['vmin'], kwrgs['vmax'],kwrgs['steps'], dtype=int)
    im = xrnpmap_init.plot.pcolormesh(ax=ax, cmap=cmap,
                               transform=ccrs.PlateCarree(), levels=clevels,
                               subplot_kws={'projection': map_proj},
                                add_colorbar=False, alpha=0.5)
    
    freq_rawprec = freq_rawprec.where(freq_rawprec.values > 0)

    ax.coastlines(color='black', alpha=0.8, linewidth=2)
    ax.add_feature(cfeature.LAND, facecolor='silver')
    list_points = np.argwhere(np.logical_and(freq_rawprec.values > 0, mask_final.values==False))
    x_co = freq_rawprec.longitude.values
    y_co = freq_rawprec.latitude.values
#        list_points = list_points - ex['grid_res']/2.
    for p in list_points:
        valueint = int((freq_rawprec.sel(latitude=y_co[p[0]], longitude=x_co[p[1]]).values))
        value = str(  np.round( (int((valueint / n_max)*10)/10), 1)  )
        ax.scatter(x_co[p[1]], y_co[p[0]], marker='${:}$'.format(value), color='black', 
                   s=150, alpha=0.2, transform=ccrs.PlateCarree())
    



    ax.set_title('')
    list_points = np.argwhere(mask_final.values==True)
    x_co = freq_rawprec.longitude.values
    y_co = freq_rawprec.latitude.values
    for p in list_points:
        valueint = int((freq_rawprec.sel(latitude=y_co[p[0]], longitude=x_co[p[1]]).values))
        value =   np.round( (int((valueint / n_max)*10)/10), 1) 
        if value == 1.0: value = int(value)
        ax.scatter(x_co[p[1]], y_co[p[0]], marker='${:}$'.format(str(value)), color='black', 
                   s=400, transform=ccrs.PlateCarree())
    
    cbar_ax = fig.add_axes([0.265, 0.07, 
                                  0.5, 0.04], label='cbar')
    norm = colors.BoundaryNorm(boundaries=clevels, ncolors=256)

    cbar = plt.colorbar(im, cbar_ax, cmap=plt.cm.tab20, orientation='horizontal', 
             extend='neither', norm=norm)
    cbar.set_ticks([])
    ticklabels = np.array(clevels, dtype=int)
    cbar.set_ticklabels(ticklabels, update_ticks=True)
    cbar.update_ticks()
    cbar.set_label('Precursor regions', fontsize=30)
    cbar.ax.tick_params(labelsize=30)
    
    yellowpatch = mpatches.Patch(color='lemonchiffon', alpha=1, 
                                 label='Gridcells rejected')
    ax.legend(handles=[yellowpatch], loc='lower left', fontsize=30)
    title = 'Sum of incomplete composites'

#    text = ['Precursor mask', 
#            'Precursor regions']
#    text_add = ['(all gridcells where N-FRP > 0.8)\n',
#                '(seperate colors)\n']
#    max_len = max([len(t) for t in text])
#    for t in text:
#        idx = text.index(t)
#        t_len = len(t)
#        expand = max_len - t_len
##        if idx == 0: expand -= 2
#        text[idx] = t + ' ' * (expand) + '   :    ' + text_add[idx]
#    
#    text = text[0] + text[1] 
#    t = ax.text(0.004, 0.995, text,
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax.transAxes,
#        color='white', fontsize=35)
#    t.set_bbox(dict(facecolor='black', alpha=1.0, edgecolor='grey'))
    
    fig.savefig(folder+title+'.pdf', bbox_inches='tight')
    

    
    #%%
    # composite mean

    composite  = composite_p1.sel(latitude=lats_fig)
    composite = composite.sel(longitude=lons_fig)
    composite = composite * (freq_rawprec / freq_rawprec.max())
    composite = composite.where(mask_final.values==True)
    fig = plt.figure(figsize = (20,12))
    ax = plt.subplot(111, projection=map_proj)
    clevels = np.linspace(-0.5, 0.5, 11)
    im = composite.plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                           transform=ccrs.PlateCarree(), levels=clevels,
                           subplot_kws={'projection': map_proj},
                            add_colorbar=False)
    ax.coastlines(color='black', alpha=0.8, linewidth=2)
    ax.add_feature(cfeature.LAND, facecolor='silver')
    cbar_ax = fig.add_axes([0.265, 0.07, 
                                  0.5, 0.04], label='cbar')
    norm = colors.BoundaryNorm(boundaries=clevels, ncolors=256)
    
    cbar = plt.colorbar(im, cbar_ax, cmap=plt.cm.tab20, orientation='horizontal', 
             extend='both', norm=norm)
    cbar.set_ticks([-0.5, -0.3, 0.0, 0.3, 0.5])
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label('Sea Surface Temperature [Kelvin]', fontsize=30)
    ax.set_title('')
#    t = ax.text(0.006, 0.994, ('Composite mean all training years\n(Precursor mask applied and weighted by N-FRP)'),
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax.transAxes,
#        color='white', fontsize=35)
#    t.set_bbox(dict(facecolor='black', alpha=1.0, edgecolor='grey'))
    title = 'final_composite'
    fig.savefig(folder+title+'.pdf', bbox_inches='tight')
#%%
    #%%
    return

def get_PDO(sst):
    #%%
    from eofs.xarray import Eof
    PDO   = find_region(sst, region='PDO')[0]
    solver = Eof(area_weighted(PDO))
    # Retrieve the leading EOF, expressed as the correlation between the leading
    # PC time series and the input SST anomalies at each grid point, and the
    # leading PC time series itself.
    eof1 = solver.eofsAsCorrelation(neofs=1).squeeze()
    init_sign = np.sign(np.mean(eof1)) # flip sign oef pattern and ts
    eof1 *= init_sign
    pc1 = solver.pcs(npcs=1, pcscaling=1).squeeze()
    pc1 *= init_sign
    return eof1, pc1

def corr_matrix_pval(df, alpha=0.05):
    from scipy import stats
    if type(df) == type(pd.DataFrame()):
        cross_corr = np.zeros( (df.columns.size, df.columns.size) )
        pval_matrix = np.zeros_like(cross_corr)
        for i1, col1 in enumerate(df.columns):
            for i2, col2 in enumerate(df.columns):
                pval = stats.pearsonr(df[col1].values, df[col2].values)
                pval_matrix[i1, i2] = pval[-1]
                cross_corr[i1, i2]  = pval[0]
        # recreate pandas cross corr
        cross_corr = pd.DataFrame(data=cross_corr, columns=df.columns, 
                                  index=df.columns)
                
    sig_mask = pval_matrix < alpha
    return cross_corr, sig_mask, pval_matrix
    #%%
#    # figure robustness 2
#    #%%
#    weights = np.sum(reg_all_1, axis=0)
#    weights[mask_final==True] = 0.
#    sum_count = np.reshape(weights, (lats.size, lons.size))
#    weights = sum_count / np.max(sum_count)
#    
#
#    
#    fig = plt.figure(figsize = (20,12))
#    ax = plt.subplot(111, projection=map_proj)
#    n_regs = xrnpmap_init.max().values
#
#    ax.coastlines(color='black', alpha=0.8, linewidth=2)
#    mask_wgths = xrdata.where(np.isnan(xrnpmap_init) == False)
#    list_points = np.argwhere(mask_wgths.values > int(ex['FCP_thres'] * len(chunks)) )
#    x_co = mask_wgths.longitude.values
#    y_co = mask_wgths.latitude.values
##        list_points = list_points - ex['grid_res']/2.
#    for p in list_points:
#        valueint = int((mask_wgths.sel(latitude=y_co[p[0]], longitude=x_co[p[1]]).values))
#        value = str(  np.round( ((valueint / n_max)), 1) )
#        ax.scatter(x_co[p[1]], y_co[p[0]], marker='${:}$'.format(value), color='black', 
#                   s=200, transform=ccrs.PlateCarree())
#    
#    xrnpmap_init.values = xrnpmap_init.values - 0.5
#    kwrgs = dict( {        'steps' : n_regs+1, 
#                           'vmin' : 0, 'vmax' : n_regs, 
#                           'cmap' : plt.cm.tab20, 
#                           'cticks_center' : True} )
#
#    clevels = np.linspace(kwrgs['vmin'], kwrgs['vmax'],kwrgs['steps'], dtype=int)
#    
#    im = xrnpmap_init.plot.pcolormesh(ax=ax, cmap=plt.cm.tab20,
#                               transform=ccrs.PlateCarree(), levels=clevels,
#                               subplot_kws={'projection': map_proj},
#                                add_colorbar=False, alpha=0.5)
#      
#    cbar_ax = fig.add_axes([0.265, 0.18, 
#                                  0.5, 0.04], label='cbar')
#    norm = colors.BoundaryNorm(boundaries=clevels, ncolors=256)
#
#    cbar = plt.colorbar(im, cbar_ax, cmap=plt.cm.tab20, orientation='horizontal', 
#             extend='neither', norm=norm)
#    cbar.set_ticks(clevels + 0.5)
#    ticklabels = np.array(clevels+1, dtype=int)
#    cbar.set_ticklabels(ticklabels, update_ticks=True)
#    cbar.update_ticks()
#    
#    cbar.set_label('Region label', fontsize=16)
#    cbar.ax.tick_params(labelsize=14)
##    ax.outline_patch.set_visible(False)
##    ax.background_patch.set_visible(False)
#    ax.set_title('')
#    title = 'Sum of incomplete composites'
#    t = ax.text(0.006, 0.994, ('"Robustness weights", (n/{:.0f})'.format(
#                                robustness.max().values)),
##                            'black contour line shows the gridcell passing the\n'
##                            '\'Composite robustness threshold\''),
#        verticalalignment='top', horizontalalignment='left',
#        transform=ax.transAxes,
#        color='black', fontsize=20)
#    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='green'))
#    title = 'robustness_weights'
#    fig.savefig(folder+title+'.pdf', bbox_inches='tight')

    
## Filter out outliers
#n_std_kickout = 7
## outliers are now nan, all values that are lower then threshold are kept (==True)
#Prec_reg = Prec_reg.where(abs(Prec_reg.values) < 10*Prec_reg.std(dim='time', skipna=True).values)
#np_arr = np.array(Prec_reg.values)
##find nans
#
#mask_allways_nan = np.product(np.isnan(np_arr), axis=0)
#np_arr[mask_allways_nan] = 0.
#for attempt in range(20):
#    mask_nan = np.isnan(np_arr)
#    prev_values = np.roll(mask_nan[:], 1, axis = 0)
#    np_arr[mask_nan[:]] = np_arr[prev_values]
#    
#    print(np.argwhere(np.isnan(np_arr)).shape)
#
#idx_nan = np.argwhere(mask_nan)
## replace the nan values by the values in the previous step
#Prec_reg.values[idx_nan] = Prec_reg.values[idx_nan - [1, 0, 0]]