#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:29:23 2019

@author: semvijverberg
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

from load_data import load_1d
import ROC_score

class SCORE_CLASS():
    '''To initialize needs information on: 
       If score should be calculated per fold:
           ex['score_per_fold'] = bool
       Number of samples for bootstrap:
           ex['n_boot'] = int
       Number of train-test splits:
           ex['n_conv'] = int
       Lags (or lead time):
           ex['lags'] = int
       Path directing to Response Variable timeseries:
           ex['RV1d_ts_path'] = str
       Filename of Response Variable timeseries:
           ex['RVts_filename'] = str
       Dates of Response Variable (what period to predict):
           ex['dates_RV'] = list of datetime objects
       Number of datapoints in single year:
           ex['n_oneyr'] = int
       Total number of years
           ex['n_yrs'] = int
      First subfolder after ex['figpathbase']:
           ex['exppathbase'] = str
       
    '''
    
    def __init__(self, ex):
#        self.method         = ex['method']
        self.score_per_fold  = ex['score_per_fold'] 
        self.notraintest    = ex['size_test'] == ex['dates_RV'].size 
        self.n_boot         = ex['n_boot']
        self._n_conv        = ex['n_conv']
        self._lags          = ex['lags']
        self._n_yrs_test    = int(ex['n_yrs'] / ex['n_conv'])
        if 'regionmcK' in ex.keys() and ex['exppathbase'].split('_')[1]=='PEP':
            self.PEPpattern = True
        else:
            self.PEPpattern = False
        if self.score_per_fold: shape = (self._n_conv, len(ex['lags']) )
        if self.score_per_fold==False: shape = (1, len(ex['lags']) )
        if 'grouped' not in ex.keys(): 
            self.grouped = False
        else:
            self.grouped = ex['grouped']
            
        filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])        
        self.RVtsfull = load_1d(filename, ex)[0]
        
        self.ROC_boot = np.zeros( (shape[0], shape[1], self.n_boot ) )
        self.dates_RV = ex['dates_RV']
        self.RV_test        = pd.DataFrame(data=np.zeros( (self.dates_RV.size ) ), 
                                           index = self.dates_RV)
        self.y_true_test        = pd.DataFrame(data=np.zeros( (self.dates_RV.size ) ), 
                                           index = self.dates_RV)
        self.y_true_train_clim = pd.DataFrame(data=np.zeros( (self.dates_RV.size ) ),      
                                           index = self.dates_RV)
        self.predmodel_1      = pd.DataFrame(data=np.zeros( (self.dates_RV.size, len(ex['lags']) ) ), 
                                           columns=ex['lags'], 
                                           index = self.dates_RV) 
        self.predmodel_2     = pd.DataFrame(data=np.zeros( (self.dates_RV.size, len(ex['lags']) ) ), 
                                           columns=ex['lags'], 
                                           index = self.dates_RV) 
        # training data is different every train test set.
        trainsize = ex['n_oneyr'] * (ex['n_yrs'] - self._n_yrs_test)
        shape_train_RV          = (self._n_conv, trainsize ) 
        self.RV_train           = np.zeros( shape_train_RV )
        self.y_true_train       = np.zeros( shape_train_RV )
        shape_train             = (self._n_conv, len(ex['lags']), trainsize ) 
        self.Prec_train         = np.zeros( shape_train )

        
        self.AUC  = pd.DataFrame(data=np.zeros( shape ), 
                                     columns=ex['lags'])
        self.KSS  = pd.DataFrame(data=np.zeros( shape ), 
                                     columns=ex['lags'])
        self.BSS  = pd.DataFrame(data=np.zeros( shape ), 
                                     columns=ex['lags'])
        self.ROC_boot = np.zeros( (shape[0], shape[1], self.n_boot ) )
        self.KSS_boot = np.zeros( (shape[0], shape[1], self.n_boot ) )
        self.FP_TP    = np.zeros( shape , dtype=list )
        
        self.RV_thresholds      = np.zeros( (self._n_conv) )
        shape_stat = (self._n_conv, len(ex['lags']) )
        self.logitmodel         = np.empty( shape_stat, dtype=list ) 
        self.Prec_train_mean    = np.zeros( shape_stat )
        self.Prec_train_std     = np.zeros( shape_stat )
        self.predmodel_1_mean     = np.zeros( shape_stat )
        self.predmodel_1_std      = np.zeros( shape_stat )
        pthresholds             = np.linspace(1, 9, 9, dtype=int)
        data = np.empty( (shape_stat[0], shape_stat[1], pthresholds.size)  )
        self.xrpercentiles = xr.DataArray(data=data, 
                                          coords=[range(shape_stat[0]), ex['lags'], pthresholds], 
                                          dims=['n_tests', 'lag','percentile'], 
                                          name='percentiles') 
        filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])
        if 'RV_aggregation' not in ex.keys():
            self.RV_aggregation = 'RVfullts95'
        else:
            self.RV_aggregation = ex['RV_aggregation']
        self.RVfullts = load_1d(filename, ex, self.RV_aggregation)[0]
        self.dates_all = pd.to_datetime(self.RVfullts.time.values)
        self.bootstrap_size = min(10, ROC_score.get_bstrap_size(self.RVfullts))
        print(f"n_block_bootstrapsize is: {self.bootstrap_size}")
    @property
    def get_pvalue_AUC(self):
        pvalue = np.zeros( (self._n_conv, len(self._lags)) )
        for n in range(self._n_conv):
            for l, lag in enumerate(self._lags):
                rand = self.ROC_boot[n, l, :]
                AUC  = self.AUC.iloc[n, l]
                pvalue[n,l] = rand[rand > AUC].size / rand.size
        return pvalue

    @property
    def get_pvalue_KSS(self):
        pvalue = np.zeros( (self._n_conv, len(self._lags)) )
        for n in range(self._n_conv):
            for l, lag in enumerate(self._lags):
                rand = self.KSS_boot[n, l, :]
                KSS  = self.KSS.iloc[n, l]
                pvalue[n,l] = rand[rand > KSS].size / rand.size
        return pvalue

    @property
    def get_mean_pvalue_KSS(self):
        pvalue = np.zeros( (len(self._lags)) )
        for l, lag in enumerate(self._lags):
            rand = np.concatenate(self.KSS_boot[:, l, :], axis=0)
            KSS  = np.median(self.KSS.iloc[:, l], axis=0)
            pvalue[l] = rand[rand > KSS].size / rand.size
        return pvalue

    @property
    def get_mean_pvalue_AUC(self):
        pvalue = np.zeros( (len(self._lags)) )
        for l, lag in enumerate(self._lags):
            rand = np.concatenate(self.ROC_boot[:, l, :], axis=0)
            AUC  = np.median(self.AUC.iloc[:, l], axis=0)
            pvalue[l] = rand[rand > AUC].size / rand.size
        return pvalue
    
    def get_autocorrelation(self):
        return ROC_score.autocorrelation(self.RVtsfull)
    


        
        

#    @property
#    def get_AUC_spatcov(self, alpha=0.05, n_boot=5):
#        self.df_auc_spatcov = pd.DataFrame(data=np.zeros( (3, len(self._lags)) ), columns=[self._lags],
#                              index=['AUC', 'con_low', 'con_high'])
#        self.Prec_test_boot = pd.DataFrame(data=np.zeros( (n_boot, len(self._lags)) ), columns=[self._lags])
#                              
#        for lag in self._lags:
#            AUC_score, conf_lower, conf_upper, sorted_scores = AUC_sklearn(
#                    self.y_true_test[lag], self.Prec_test[lag], 
#                    alpha=alpha, n_bootstraps=n_boot)
#            self.df_auc_spatcov[lag] = (AUC_score, conf_lower, conf_upper) 
#            self.Prec_test_boot[lag] = sorted_scores
#        return self