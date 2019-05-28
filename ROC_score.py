# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Mon Oct 15 17:50:16 2018

@author: semvijverberg
"""

import random
import os
import numpy as np
import pandas as pd
import func_CPPA
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn.metrics import roc_auc_score


class SCORE_CLASS():
    
    def __init__(self, ex):
        self.method         = ex['method']
        self.fold           = (ex['method'][:6]=='random'
                               and ex['method'][-4:]=='fold')
        self.all_test       = (ex['leave_n_out'] == True 
                               and ex['method'] == 'iter'
                               or ex['ROC_leave_n_out']
                               or (ex['method'][:6]=='random'
                               and ex['method'][-4:]!='fold'))
        self.notraintest    = (ex['leave_n_out'] == False
                               or ex['method'][:5] == 'split')
        self.n_boot         = ex['n_boot']
        self._n_conv        = ex['n_conv']
        self._lags          = ex['lags']
        
        if self.fold: shape = (self._n_conv, len(ex['lags']) )
        if self.fold==False or self.notraintest: shape = (1, len(ex['lags']) )
        
        
        self.ROC_boot = np.zeros( (shape[0], shape[1], self.n_boot ) )

        self.RV_test        = pd.DataFrame(data=np.zeros( (ex['dates_RV'].size, len(ex['lags']) ) ), 
                                           columns=ex['lags'], 
                                           index = ex['dates_RV'])
        self.y_true_test        = pd.DataFrame(data=np.zeros( (ex['dates_RV'].size, len(ex['lags']) ) ), 
                                               columns=ex['lags'], 
                                           index = ex['dates_RV'])
        self.Prec_test      = pd.DataFrame(data=np.zeros( (ex['dates_RV'].size, len(ex['lags']) ) ), 
                                           columns=ex['lags'], 
                                           index = ex['dates_RV']) 
        self.logit_test     = pd.DataFrame(data=np.zeros( (ex['dates_RV'].size, len(ex['lags']) ) ), 
                                           columns=ex['lags'], 
                                           index = ex['dates_RV']) 
        # training data is different every train test set.
        trainsize = ex['train_test_list'][0][0]['RV'].size
        shape_train          = (self._n_conv, len(ex['lags']), trainsize ) 
        self.RV_train        = np.zeros( shape_train )
        self.y_true_train    = np.zeros( shape_train )
        self.Prec_train      = np.zeros( shape_train )
                
        

        
        self.AUC  = pd.DataFrame(data=np.zeros( shape ), 
                                     columns=ex['lags'])
        self.KSS  = pd.DataFrame(data=np.zeros( shape ), 
                                     columns=ex['lags'])
        self.ROC_boot = np.zeros( (shape[0], shape[1], self.n_boot ) )
        self.KSS_boot = np.zeros( (shape[0], shape[1], self.n_boot ) )
        self.FP_TP    = np.zeros( shape , dtype=list )
        
        shape_stat = (self._n_conv, len(ex['lags']) )
        self.logitmodel         = np.empty( shape_stat, dtype=list ) 
        self.RV_thresholds      = np.zeros( shape_stat )
        self.Prec_train_mean    = np.zeros( shape_stat )
        self.Prec_train_std     = np.zeros( shape_stat )
        self.Prec_test_mean     = np.zeros( shape_stat )
        self.Prec_test_std      = np.zeros( shape_stat )
        pthresholds             = np.linspace(1, 9, 9, dtype=int)
        data = np.empty( (shape_stat[0], shape_stat[1], pthresholds.size)  )
        self.xrpercentiles = xr.DataArray(data=data, 
                                          coords=[range(shape_stat[0]), ex['lags'], pthresholds], 
                                          dims=['n_tests', 'lag','percentile'], 
                                          name='percentiles') 
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
    
    @property
    def get_AUC_spatcov(self, alpha=0.05, n_boot=5):
        self.df_auc_spatcov = pd.DataFrame(data=np.zeros( (3, len(self._lags)) ), columns=[self._lags],
                              index=['AUC', 'con_low', 'con_high'])
        self.Prec_test_boot = pd.DataFrame(data=np.zeros( (n_boot, len(self._lags)) ), columns=[self._lags])
                              
        for lag in self._lags:
            AUC_score, conf_lower, conf_upper, sorted_scores = AUC_sklearn(
                    self.y_true_test[lag], self.Prec_test[lag], 
                    alpha=alpha, n_bootstraps=n_boot)
            self.df_auc_spatcov[lag] = (AUC_score, conf_lower, conf_upper) 
            self.Prec_test_boot[lag] = sorted_scores
        return self
                

def only_spatcov_wrapper(l_ds_CPPA, RV_ts, Prec_reg, ex):
    #%%
    # init class
    SCORE = SCORE_CLASS(ex)
    
    
    ex['test_ts_prec'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
#        ex['test_year'] = list(set(test['RV'].time.dt.year.values))

       
        
        test  = ex['train_test_list'][n][1]
        train = ex['train_test_list'][n][0]
        RV_ts_train = train['RV']
        Prec_train_idx = train['Prec_train_idx']
        Prec_train_reg = Prec_reg.isel(time=Prec_train_idx)
        
        ds_Sem = l_ds_CPPA[n]
        
        get_statistics_train(RV_ts_train, ds_Sem, Prec_train_reg, SCORE, ex)
        
    
        
        Prec_test_reg = Prec_reg.isel(time=test['Prec_test_idx'])
        ROC_score_only_spatcov(test, ds_Sem, Prec_test_reg, SCORE, ex)
#    ex['score'] = 
    #%%
    return ex, SCORE


def create_validation_plot(outdic_folders, metric='AUC'):
#    from time_series_analysis import subplots_df
    #%%
    # each folder will become a single df to plot
    import numpy as np, scipy.stats as st
    from scipy.stats import kstest
    from scipy.stats import ks_2samp
    # ks_2samp(scorecl.AUC[lag], ROC_array[-1])
    scoreclasses  = {}
    df_series     = []
    datasets      = []
    for folder in outdic_folders:
        filename = 'output_main_dic'
        dic = np.load(os.path.join(folder, filename+'.npy'),  encoding='latin1').item()
        ex = dic['ex']
        scoreclasses[ex['datafolder']] = ex['score']
        df_series.append( ex['score'].AUC.mean(0).values )
        datasets.append( ex['datafolder'] )
    df = pd.DataFrame(np.concatenate(np.array(df_series)[None,:], axis=0),
                      index=datasets, columns=ex['lags'])

    if metric == 'AUC':
        y_lim = (0.3,1)
    elif metric == 'KSS':    
        y_lim = (-1,1)
    lags_f = np.array(ex['lags']) - 0.5
    lags_s = np.array(ex['lags']) + 0.5
    g = sns.FacetGrid(df, row=len(datasets)-1, size=7, aspect=1.4,
                      ylim=y_lim)
    for i, ax in enumerate(g.axes.flatten()):
        name = datasets[0]
        scorecl = scoreclasses[name]
        if metric == 'AUC':
            score_metric = scorecl.AUC
            boot = scorecl.ROC_boot
        elif metric == 'KSS':
            score_metric = scorecl.KSS
            boot = scorecl.KSS_boot
        
        
        
        # random shuffle
        conf_int = np.empty( (2, len(ex['lags']) ) )
        if boot.size!=0:
            ROC_array = np.zeros( (len(ex['lags']),boot.shape[0]*boot.shape[2]) )
            for l, lag in enumerate(ex['lags']):

                ROC_array[l] = np.concatenate(boot[:,l,:], axis=0)
                a = ROC_array[l]
                con = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
                conf_int[:,l] = con
            ax.fill_between(ex['lags'], conf_int[0], conf_int[1], linestyle='solid', 
                        edgecolor='black', facecolor='blue', alpha=0.5)
            ax.plot(ex['lags'], np.median(ROC_array, axis=1), color='blue', 
                    linewidth=2, label='Bootstrapping')
            ax.boxplot(ROC_array.T, positions=lags_s, widths=2)
        
        # mean AUC and error?
        conf_int = np.empty( (2, len(ex['lags']) ) )
        for l, lag in enumerate(ex['lags']):
            # kolmogorov-smirnovtoets
            a = score_metric[lag]
            Ks = kstest(a, 'norm')
            if Ks.pvalue > 0.05: print('lag {} distibution is not normal'.format(lag))
            con = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
            conf_int[:,l] = con
            
        ax.fill_between(ex['lags'], conf_int[0], conf_int[1], linestyle='solid', 
                        edgecolor='black', facecolor='red', alpha=0.5)

        median_score = score_metric.median(axis=0)

        ax.plot(ex['lags'], median_score, color='red', 
                linewidth=2, label='k-fold validation {}'.format(metric))
        
        ax.boxplot(score_metric.T, positions=lags_f, widths=2)
    
        ax.set_xlim(min(ex['lags'])-5,max(ex['lags'])+5)
        ax.set_xticks(ex['lags'])  
        ax.set_xticklabels(ex['lags'])        
        ax.legend()
    
    lags_str = str(ex['lags']).replace(' ', '')
    fname = 'validation_plot_{}_{}'.format(lags_str, metric)
    filename = os.path.join(ex['figpathbase'], ex['CPPA_folder'], fname)
    g.fig.savefig(fname ,dpi=250, frameon=True)
    #%%
    return




def ROC_score_only_spatcov(test, ds_Sem, Prec_test_reg, SCORE, ex):
    #%%
    # =============================================================================
    # calc ROC scores
    # =============================================================================      
    
    for lag_idx, lag in enumerate(ex['lags']):
        
        idx = ex['lags'].index(lag)
        dates_test = pd.to_datetime(test['RV'].time.values)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')


        var_test_reg = Prec_test_reg.sel(time=dates_min_lag)        

        if ex['use_ts_logit'] == False:
            # weight by robustness of precursors
            var_test_reg = var_test_reg * ds_Sem['weights'].sel(lag=lag)
            crosscorr_Sem = func_CPPA.cross_correlation_patterns(var_test_reg, 
                                                            ds_Sem['pattern_CPPA'].sel(lag=lag))
        elif ex['use_ts_logit'] == True:
            crosscorr_Sem = ds_Sem['ts_prediction'][lag_idx]


        
        if SCORE.all_test: 
            if ex['n'] == 0:
                ex['test_RV'][lag_idx]          = test['RV'].values
                ex['test_ts_prec'][lag_idx]     = crosscorr_Sem.values
            elif len(ex['test_ts_prec'][lag_idx]) <= len(ex['dates_RV']):
                ex['test_RV'][lag_idx]      = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )  
                ex['test_ts_prec'][lag_idx] = np.concatenate( [ex['test_ts_prec'][lag_idx], crosscorr_Sem.values] )
                
                
        # ROC over folds, emptying array every fold 
        if SCORE.fold or SCORE.notraintest:
            ex['test_RV'][idx]          = test['RV'].values
            ex['test_ts_prec'][lag_idx]  = crosscorr_Sem.values
       
            
            
        if np.logical_and(SCORE.all_test, ex['n'] == ex['n_conv']-1) or SCORE.fold==True:
            
            ts_RV    = ex['test_RV'][idx]
            ts_pred  = (ex['test_ts_prec'][lag_idx] - SCORE.Prec_train_mean[ex['n'],lag_idx]) / \
                        SCORE.Prec_train_std[ex['n'],lag_idx]
            logit_pred = SCORE.logitmodel[ex['n']][lag_idx].predict(ts_pred)
            
            events_idx = np.where(ex['test_RV'][idx] > SCORE.RV_thresholds[ex['n'],lag_idx])[0]
            y_true = func_CPPA.Ev_binary(events_idx, len(ex['test_RV'][idx]),  ex['min_dur'], 
                                     ex['max_break'], grouped=False)
            y_true[y_true!=0] = 1
            
            if SCORE.fold:
                dates_tofill = dates_test
            else:
                dates_tofill = ex['dates_RV'] 
            SCORE.Prec_test_mean[ex['n'],lag_idx]   = np.mean(ts_pred)                
            SCORE.Prec_test_std[ex['n'],lag_idx]    = np.std(ts_pred)            
            percentiles_train = SCORE.xrpercentiles[ex['n']].sel(lag=lag)
            SCORE.RV_test.loc[dates_tofill, lag]    = pd.Series(ts_RV, 
                                                   index=dates_tofill)
            SCORE.y_true_test.loc[dates_tofill, lag]= pd.Series(y_true, 
                                                   index=dates_tofill)
            SCORE.Prec_test.loc[dates_tofill, lag]  = pd.Series(ts_pred, 
                                                   index=dates_tofill)
            SCORE.logit_test.loc[dates_tofill, lag]= pd.Series(logit_pred, 
                                                   index=dates_tofill)
            
        

            if lag_idx == 0:
                SCORE.Prec_len  = ts_pred.size
                SCORE.RV_len    = len(ex['test_RV'][idx])
                SCORE.n_events  = y_true[y_true!=0].sum()
            
                print('Calculating ROC scores\nDatapoints precursor length '
                  '{}\nDatapoints RV length {}, with {:.0f} events'.format(SCORE.Prec_len,
                   len(ex['test_RV'][idx]), y_true[y_true!=0].sum()))
    
            
            AUC_score, FP, TP, ROCboot, KSSboot = ROC_score(ts_pred, y_true,
                                                    n_boot=SCORE.n_boot, win=0, 
                                                    n_blocks=ex['n_yrs'], 
                                                    thr_pred=percentiles_train)
            
            
            
            if SCORE.fold==True:
                # store results fold
                SCORE.AUC.iloc[ex['n']][lag]        = AUC_score
                SCORE.KSS.iloc[ex['n']][lag]        = get_KSS(TP, FP)
                SCORE.FP_TP[ex['n'],lag_idx]        = FP, TP 
                SCORE.ROC_boot[ex['n'],lag_idx, :]  = ROCboot
                SCORE.KSS_boot[ex['n'],lag_idx, :]  = KSSboot
                if ex['n'] != ex['n_conv']-1:
                    # empty arrays.               
                    ex['test_ts_prec'] = np.zeros( len(ex['lags']) , dtype=list)
                    ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
    


   
            
        elif SCORE.notraintest:
            if idx == 0:
                print('performing hindcast')

#            ex['test_RV'][lag_idx]          = test['RV'].values
#            ex['test_ts_prec'][lag_idx]     = crosscorr_Sem.values
            ts_pred  = ((ex['test_ts_prec'][lag_idx]-SCORE.Prec_train_mean[ex['n'],lag_idx]) / \
                                    SCORE.Prec_train_std[ex['n'],lag_idx] )              
            
            if lag > 30:
                obs_array = pd.DataFrame(ex['test_RV'][0])
                obs_array = obs_array.rolling(7, center=True, min_periods=1).mean()
                threshold = (obs_array.mean() + obs_array.std()).values
                events_idx = np.where(obs_array > threshold)[0]
            else:
                events_idx = np.where(ex['test_RV'][0] > ex['event_thres'])[0]
            y_true = func_CPPA.Ev_binary(events_idx, len(ex['test_RV'][0]),  ex['min_dur'], 
                                     ex['max_break'], grouped=False)
            y_true[y_true!=0] = 1

            AUC_score, FP, TP, ROCboot, KSSboot = ROC_score(ts_pred, y_true,
                                                    n_boot=0, win=0, 
                                                    n_blocks=ex['n_yrs'])
                                                    
            


           

        if SCORE.fold==False and ex['n'] == ex['n_conv']-1:
            SCORE.AUC.iloc[0][lag]     = AUC_score
            SCORE.FP_TP[0,lag_idx]    = FP, TP 
            SCORE.ROC_boot[0,lag_idx, :] = ROCboot 
            print('\n*** ROC score for {} lag {} ***\n\nCPPA {:.2f} '
            ' ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
              lag, SCORE.AUC.iloc[0][lag], np.percentile(SCORE.ROC_boot[0,lag_idx, :], 95)))
        if SCORE.fold and ex['n'] == ex['n_conv']-1:
            print('\n*** ROC score for {} lag {} ***\n\nCPPA {:.2f} '
            ' ±{:.2f} std over {} folds, {:.2f} 95th perc random events\n'.format(ex['region'], 
              lag, np.mean(SCORE.AUC.iloc[:,lag_idx]), np.std(SCORE.AUC.iloc[:,lag_idx]), 
              ex['n_conv'], np.percentile(ROCboot, 95) ) ) 

    #%%
    return 

def get_KSS(TP, FP):
    '''Hansen Kuiper Skill Score from True & False positive rate'''
    KSS_allthreshold = np.zeros(len(TP))
    for i in range(len(TP)):
        KSS_allthreshold[i] = TP[i] - FP[i]
    return max(KSS_allthreshold)

def get_statistics_train(RV_ts_train, ds_Sem, Prec_train_reg, SCORE, ex):
#%%
                     

    pthresholds = np.linspace(1, 9, 9, dtype=int)
   
    
    for lag_idx, lag in enumerate(ex['lags']):
        dates_train = pd.to_datetime(RV_ts_train.time.values)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_train - pd.Timedelta(int(lag), unit='d')


        var_train_reg = Prec_train_reg.sel(time=dates_min_lag)   
        
        if ex['use_ts_logit'] == False:
            # weight by robustness of precursors
            var_train_reg = var_train_reg * ds_Sem['weights'].sel(lag=lag)
            spatcov = func_CPPA.cross_correlation_patterns(var_train_reg, 
                                                            ds_Sem['pattern_CPPA'].sel(lag=lag))
        elif ex['use_ts_logit'] == True:
            spatcov = ds_Sem['ts_prediction'][lag_idx]
            
        SCORE.Prec_train_mean[ex['n'],lag_idx] = spatcov.mean().values
        SCORE.Prec_train_std[ex['n'],lag_idx]  = spatcov.std().values
        
        spatcov_norm = (spatcov - SCORE.Prec_train_mean[ex['n'],lag_idx]) / \
                        SCORE.Prec_train_std[ex['n'],lag_idx]

        SCORE.RV_train[ex['n']][lag_idx][:RV_ts_train.size]  = RV_ts_train.values
        SCORE.Prec_train[ex['n']][lag_idx][:RV_ts_train.size] = spatcov_norm.values
        
        
        
        p_pred = []
        for p in pthresholds:	
            p_pred.append(np.percentile(spatcov_norm.values, p*10))
            
        SCORE.xrpercentiles[ex['n']][lag_idx] = p_pred
        
        obs_array = pd.DataFrame(RV_ts_train.values)
        if lag >= 30:
            obs_array = obs_array.rolling(7, center=True, min_periods=1).mean()
        if ex['event_percentile'] == 'std':
            # binary time serie when T95 exceeds 1 std
            threshold = obs_array.mean().values + obs_array.std().values
        else:
            percentile = ex['event_percentile']
            threshold = np.percentile(obs_array.values, percentile)
        SCORE.RV_thresholds[ex['n'],lag_idx] = threshold
        events_idx = np.where(RV_ts_train.values > threshold)[0]
        y_true_train = func_CPPA.Ev_binary(events_idx, len(RV_ts_train),  
                                           ex['min_dur'], ex['max_break'], grouped=False)
        y_true_train[y_true_train!=0] = 1               
        SCORE.y_true_train[ex['n']][lag_idx][:RV_ts_train.size] = y_true_train
        model = sm.Logit(y_true_train, spatcov_norm.values, disp=0)
        result = model.fit( disp=0 )
        SCORE.logitmodel[ex['n']][lag_idx] = result
        
#%%
    return

def ROC_score_wrapper(ex):
    #%%
    ex['score'] = []
    FP_TP    = np.zeros(len(ex['lags']), dtype=list)
    ROC_Sem  = np.zeros(len(ex['lags']))
    ROC_boot = np.zeros( (len(ex['lags']), ex['n_boot']) )
    KSS_boot = np.zeros( (len(ex['lags']), ex['n_boot']) )

    if 'n_boot' not in ex.keys():
        n_boot = 0
    else:
        n_boot = ex['n_boot']

    
    for lag_idx, lag in enumerate(ex['lags']):
        if lag > 30:
            obs_array = pd.DataFrame(ex['test_RV'][0])
            obs_array = obs_array.rolling(7, center=True, min_periods=1).mean()
            threshold = (obs_array.mean() + obs_array.std()).values
            events_idx = np.where(obs_array > threshold)[0]
        else:
            events_idx = np.where(ex['test_RV'][0] > ex['event_thres'])[0]
        y_true = func_CPPA.Ev_binary(events_idx, len(ex['test_RV'][0]),  ex['min_dur'], 
                                 ex['max_break'], grouped=False)
        y_true[y_true!=0] = 1
        if lag_idx == 0:
            print('Calculating ROC scores\nDatapoints precursor length '
              '{}\nDatapoints RV length {}'.format(len(ex['test_ts_prec'][0]),
               len(ex['test_RV'][0])))
            
        ts_pred  = ((ex['test_ts_prec'][lag_idx]-np.mean(ex['test_ts_prec'][lag_idx]))/ \
                                  (np.std(ex['test_ts_prec'][lag_idx]) ) )                 

#                func_CPPA.plot_events_validation(ex['test_ts_prec'][idx], ex['test_ts_mcK'][idx], test['RV'], Prec_threshold_Sem, 
#                                            Prec_threshold_mcK, ex['event_thres'], 2000)
        
        
        if 'use_ts_logit' in ex.keys() and ex['use_ts_logit'] == True:
            ROC_Sem[lag_idx], FP, TP, ROC_boot[lag_idx], KSS_boot[lag_idx] = ROC_score(
                            ts_pred, y_true, n_boot=n_boot, win=0, n_blocks=ex['n_yrs'])
        else:
            ROC_Sem[lag_idx], FP, TP, ROC_boot[lag_idx], KSS_boot[lag_idx] = ROC_score(
                            ts_pred, y_true, n_boot=n_boot, win=0, n_blocks=ex['n_yrs'])
                                           
        
        FP_TP[lag_idx] = FP, TP 
        
        print('\n*** ROC score for {} lag {} ***\n\nCPPA {:.2f} '
        ' ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
          lag, ROC_Sem[lag_idx], np.percentile(ROC_boot[lag_idx], 99)))
        
    ex['score'].append([ROC_Sem, ROC_boot, FP_TP])
    #%%
    return ex


def ROC_score(predictions, obs_binary, n_boot=0, win=0, n_blocks=39, thr_pred='default'):
    #%%
#    win = 7
#    predictions = pred
#    obs_binary = y_true
#    thr_event = ex['event_thres']
    
   # calculate ROC scores
    obs_binary = np.copy(obs_binary)
    # Standardize predictor time series
#    predictions = predictions - np.mean(predictions)
    # P_index = np.copy(AIR_rain_index)	
    # Test ROC-score			
    
    TP_rate = np.ones((11))
    FP_rate =  np.ones((11))
    TP_rate[10] = 0
    FP_rate[10] = 0
    
    
    #print(fixed_event_threshold) 
    events = np.where(obs_binary > 0.5)[0][:]  
    not_events = np.where(obs_binary == 0.0)[0][:]    
    
    for p in np.linspace(1, 9, 9, dtype=int):	
        if str(thr_pred) == 'default':
            p_pred = np.percentile(predictions, p*10)
        else:
            p_pred = thr_pred.sel(percentile=p).values
            
        positives_pred = np.where(predictions >= p_pred)[0][:]
        negatives_pred = np.where(predictions < p_pred)[0][:]
        
        pos_pred_win = positives_pred
        events_win = events
        if win != 0:
            for w in range(win):
                w += 1
                pos_pred_win = np.concatenate([pos_pred_win, pos_pred_win-w, pos_pred_win+w])
                events_win = np.concatenate([events_win, events_win-w, events_win+w])

            
#            neg_pred_win = [a for a in negatives_pred if a not in pos_pred_win]
            pos_pred_win = np.unique(pos_pred_win)
            events_win = np.unique(events_win)
            not_events_win = [a for a in range(len(obs_binary)) if a not in events_win ]
            # positive prediction are tested within a window
            # i: pos_pred_win is expanded ± 3 days, i.e. more positive pred can match events
            # ii: if a 
            True_pos = len([a for a in events if a in pos_pred_win])
            False_neg = len([a for a in events if a not in pos_pred_win])
            

            False_pos = len([a for a in positives_pred if a not in events_win])
            True_neg = len([a for a in negatives_pred if a in not_events_win])
            
            
#            # attempt 3
#            #! this leads to double counting of positive pred which are all in events win
#            True_pos = len([a for a in positives_pred if a in events_win])
#            False_pos = len([a for a in positives_pred if a not in events_win])
#            
##            # negative prediction should also not lead to a single event within window
##            True_neg = len([a for a in negatives_pred if a in not_events])
##            False_neg = len([a for a in negatives_pred if a not in not_events])
##
#            # negative prediction are not changed
#            True_neg = len([a for a in negatives_pred if a in not_events])
#            False_neg = len([a for a in negatives_pred if a in events])

        else:
            True_pos = len([a for a in positives_pred if a in events])
            False_neg = len([a for a in negatives_pred if a in events])
            
            False_pos = len([a for a in positives_pred if a  in not_events])
            True_neg = len([a for a in negatives_pred if a  in not_events])
            
            True_pos = len([a for a in events if a in positives_pred])
            False_neg = len([a for a in events if a in negatives_pred])
            
            False_pos = len([a for a in not_events if a  in positives_pred])
            True_neg = len([a for a in not_events if a  in negatives_pred])
        
        True_pos_rate = True_pos/(float(True_pos) + float(False_neg))
        False_pos_rate = False_pos/(float(False_pos) + float(True_neg))
        
        FP_rate[p] = False_pos_rate
        TP_rate[p] = True_pos_rate
        
    if n_boot != 0:
        ROC_boot, KSS_boot = ROC_bootstrap(predictions, obs_binary, n_boot, win=0, n_blocks=39, thr_pred='default')
    else:
        ROC_boot = 0
        KSS_boot  = 0
        
    AUC_score = np.abs(np.trapz(TP_rate, x=FP_rate ))
    
    return AUC_score, FP_rate, TP_rate, ROC_boot, KSS_boot

    #%%

def ROC_bootstrap(predictions, obs_binary, n_boot, win=0, n_blocks=39, thr_pred='default'):
    
    obs_binary = np.copy(obs_binary)
    AUC_new    = np.zeros((n_boot))
    KSS_bootstrap  = np.zeros((n_boot))
    
    ROC_bootstrap = 0
    for j in range(n_boot):
        
#        # shuffle observations / events
#        old_index = range(0,len(obs_binary),1)
#        sample_index = random.sample(old_index, len(old_index))
        
        # shuffle years, but keep years complete:
        old_index = range(0,len(obs_binary),1)
#        n_yr = ex['n_yrs']
        n_oneyr = int( len(obs_binary) / n_blocks )
        chunks = [old_index[n_oneyr*i:n_oneyr*(i+1)] for i in range(int(len(old_index)/n_oneyr))]
        # replace lost value because of python indexing 
#        chunks[-1] = range(chunks[-1][0], chunks[-1][-1])
        rand_chunks = random.sample(chunks, len(chunks))
        #print(sample_index)
#        new_obs_binary = np.reshape(obs_binary[sample_index], -1)  
        
        new_obs_binary = []
        for chunk in rand_chunks:
            new_obs_binary.append( obs_binary[chunk] )
        
        new_obs_binary = np.reshape( new_obs_binary, -1 )
        # _____________________________________________________________________________
        # calculate new AUC score and store it
        # _____________________________________________________________________________
        #
    
        new_obs_binary = np.copy(new_obs_binary)
        # P_index = np.copy(MT_rain_index)	
        # Test AUC-score			
        TP_rate = np.ones((11))
        FP_rate =  np.ones((11))
        TP_rate[10] = 0
        FP_rate[10] = 0

        events = np.where(new_obs_binary > 0.5)[0][:]  
        not_events = np.where(new_obs_binary == 0.0)[0][:]  
        
        for p in np.linspace(1, 9, 9, dtype=int):	
            if str(thr_pred) == 'default':
                p_pred = np.percentile(predictions, p*10)
            else:
                p_pred = thr_pred.sel(percentile=p).values[0]
            
            p_pred = np.percentile(predictions, p*10)
            positives_pred = np.where(predictions > p_pred)[0][:]
            negatives_pred = np.where(predictions <= p_pred)[0][:]
    
    						
            True_pos = [a for a in positives_pred if a in events]
            False_neg = [a for a in negatives_pred if a in events]
            
            False_pos = [a for a in positives_pred if a  in not_events]
            True_neg = [a for a in negatives_pred if a  in not_events]
            
            True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
            False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
            
            FP_rate[p] = False_pos_rate
            TP_rate[p] = True_pos_rate
            
            #check
            if len(True_pos+False_neg) != len(events) :
                print("check 136")
            elif len(True_neg+False_pos) != len(not_events) :
                print("check 138")
           
            True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
            False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
            
            FP_rate[p] = False_pos_rate
            TP_rate[p] = True_pos_rate
        
        
        KSS_bootstrap[j]  = get_KSS(TP_rate, FP_rate)
        AUC_score  = np.abs(np.trapz(TP_rate, FP_rate))
        AUC_new[j] = AUC_score
        AUC_new    = np.sort(AUC_new[:])[::-1]
#        pval       = (np.asarray(np.where(AUC_new > ROC_score)).size)/ n_boot
        ROC_bootstrap = AUC_new 
    #%%
    return ROC_bootstrap, KSS_bootstrap


# =============================================================================
# =============================================================================
# Plotting
# =============================================================================
# =============================================================================
def plotting_timeseries(test, yrs_to_plot, ex):
    for lag in ex['lags']:
        #%%
        idx = ex['lags'].index(lag)
        # normalize
     
        ts_pred_Sem  = ((ex['test_ts_prec'][idx]-np.mean(ex['test_ts_prec'][idx]))/ \
                                  (np.std(ex['test_ts_prec'][idx]) ) )
        norm_test_RV = ((ex['test_RV'][idx]-np.mean(ex['test_RV'][idx]))/ \
                                  (np.std(ex['test_RV'][idx]) ) ) 
        labels       = pd.to_datetime(ex['dates_RV'])
            
        
#        threshold = np.std(norm_test_RV)
        if ex['event_percentile'] == 'std':
            # binary time serie when T95 exceeds 1 std
            threshold = np.std(norm_test_RV) 
        else:
            percentile = ex['event_percentile']
            threshold= np.percentile(norm_test_RV, percentile)
        
        No_train_yrs = ex['n_yrs'] - int(test['RV'].size / ex['n_oneyr'])
        title = ('Prediction time series versus truth (lag={}), '
                 'with {} training years'.format(lag, No_train_yrs))
        years = labels.year
        years.values[-1] = ex['endyear']+1
#                    years = np.concatenate((labels, [labels[-1]+1]))
        df = pd.DataFrame(data={'RV':norm_test_RV, 'CPPA':ts_pred_Sem, 
                                'date':labels, 'year':years} )
        df['RVrm'] = df['RV'].rolling(20, center=True, min_periods=5, 
              win_type=None).mean()
        
        
        # check if yrs to plot are in test set:
        n_yrs_to_plot = len(yrs_to_plot)
        yrs_to_plot = [yr for yr in yrs_to_plot if yr in set(years)]
        n_miss = n_yrs_to_plot - len(yrs_to_plot)
        yrs_to_add = [yr for yr in set(years) if yr not in yrs_to_plot]
        np.random.shuffle(yrs_to_add)
        [yrs_to_plot.append(yr) for yr in yrs_to_add[:n_miss]]
        yrs_to_plot.append( ex['endyear']+1 )
        df['yrs_plot'] = [yr in yrs_to_plot for yr in df['year']]
        df = df.where(df['yrs_plot']==True).dropna()
        g = sns.FacetGrid(df, col='year', col_wrap=3, sharex=False, size=2.5,
                          aspect = 2)
        import matplotlib.dates as mdates
        n_plots = len(g.axes)
        for n_ax in np.arange(0,n_plots):
            ax = g.axes.flatten()[n_ax]
            df_sub = df[df['year'] == yrs_to_plot[n_ax]]
            df_sub = df_sub.groupby(by='date', as_index=False).mean()
            ax.set_ylim(-3,3)
#                        print(df_sub['date'])
#                        start_date = pd.to_datetime(df_sub['date'].iloc[0])
            ax.hlines(0, df_sub['date'].iloc[0],df_sub['date'].iloc[-1], alpha=.7)
            ax.grid(which='major', alpha=0.3)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
#                        ax.set_xticks(df_sub['date'][::20])
#                        ax.set_xlim(df_sub['date'].iloc[0],df_sub['date'].iloc[-1])
            # should normalize with std from training spatial covariance or logit ts
            ax.plot(df_sub['date'],df_sub['CPPA'], linewidth=2,
                    label='CPPA', color='blue', alpha=0.9)
            ax.plot(df_sub['date'], df_sub['RV'], alpha=0.4, 
                    label='Truth', color='red', linewidth=0.5) 
            ax.plot(df_sub['date'], df_sub['RVrm'], alpha=0.9, 
                    label='Truth roll. mean 20', color='black',
                    linewidth=2)
            
            ax.fill_between(df_sub['date'].values, threshold, df_sub['RV'].values, 
                             where=(df_sub['RV'].values > threshold),
                             interpolate=True, color="orange", alpha=0.7, label="Events")
            if n_ax+1 == n_plots:
                ax.axis('off')
                ax.legend(loc='lower center', prop={'size': 15})
        g.fig.text(0.5, 1.02, title, fontsize=15,
               fontweight='heavy', horizontalalignment='center')
        filename = '{} day lead time series prediction'.format(lag)
        path = os.path.join(ex['figpathbase'], ex['CPPA_folder'])
        file_name = os.path.join(path,filename+'.png')
        g.fig.savefig(file_name ,dpi=250, frameon=True)
        plt.show()
        #%%

def get_AUC(SCORE, y_true, prec_test, alpha=0.05, n_boot=5):
    df_auc = pd.DataFrame(data=np.zeros( (3, len(SCORE._lags)) ), columns=[SCORE._lags],
                          index=['AUC', 'con_low', 'con_high'])
    for lag in SCORE._lags:
        AUC_score, conf_lower, conf_upper, sorted_scores = AUC_sklearn(
                    y_true[lag], prec_test[lag], 
                    alpha=alpha, n_bootstraps=n_boot)
        df_auc[lag] = (AUC_score, conf_lower, conf_upper) 
    return df_auc, sorted_scores

def AUC_sklearn(y_true, y_pred, alpha=0.05, n_bootstraps=5):
    
    AUC_score = roc_auc_score(y_true, y_pred)
    print("Original ROC area: {:0.3f}".format(AUC_score))
    
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
#        print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(alpha * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1-alpha) * len(sorted_scores))]
#    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
#        confidence_lower, confidence_upper))
    return AUC_score, confidence_lower, confidence_upper, sorted_scores
