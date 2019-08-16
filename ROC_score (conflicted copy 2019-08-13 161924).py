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
from itertools import chain
from sklearn import metrics
from statsmodels.api import add_constant
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import brier_score_loss
from load_data import load_1d
flatten = lambda l: [item for sublist in l for item in sublist]

from scoringclass import SCORE_CLASS
    

                

def only_spatcov_wrapper(l_ds_CPPA, RV_ts, Prec_reg, ex):
    #%%
    if ex['method'][:6] == 'random': ex['score_per_fold'] = True
    ex['size_test'] = ex['train_test_list'][0][1]['RV'].size
    # init class
    SCORE = SCORE_CLASS(ex)
#    SCORE.n_boot = 100
    print(f"n bootstrap is: {SCORE.n_boot}")
    if SCORE.PEPpattern:
        Prec_reg = func_CPPA.find_region(Prec_reg, region=ex['regionmcK'])[0]
        
    SCORE.dates_test = np.zeros( (ex['n_conv'], ex['size_test']), dtype='datetime64[D]')
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n     
        
        test  = ex['train_test_list'][n][1]
        RV_ts_test = test['RV']
        dates_test = pd.to_datetime(RV_ts_test.time.values)
        SCORE.dates_test[ex['n']] = dates_test
        
        Prec_test_reg = Prec_reg.isel(time=test['Prec_test_idx'])
        
        train = ex['train_test_list'][n][0]
        RV_ts_train = train['RV']
        Prec_train_idx = train['Prec_train_idx']
        Prec_train_reg = Prec_reg.isel(time=Prec_train_idx)
        dates_train = pd.to_datetime(RV_ts_train.time.values)
        ds_Sem = l_ds_CPPA[n]
        
        spatcov_train = calculate_spatcov(Prec_train_reg, SCORE._lags, dates_train, ds_Sem, 
                                    PEPpattern=SCORE.PEPpattern)
        
        add_training_data_lags(RV_ts_train, spatcov_train, SCORE, ex)
        
        
        spatcov_test = calculate_spatcov(Prec_test_reg, SCORE._lags, dates_test, ds_Sem, 
                            PEPpattern=SCORE.PEPpattern)
        
        add_test_data(RV_ts_test, spatcov_test, SCORE, ex)
        
        print('Test years {}'.format( np.unique(pd.to_datetime(dates_test).year)) )
    print('Calculate scores')
    
    get_scores_per_fold(SCORE)

    SCORE.AUC_spatcov, SCORE.KSS_spatcov, SCORE.brier_spatcov, metrics_sp = get_metrics_sklearn(SCORE,
                                                                            SCORE.predmodel_1,
                                                                            n_boot=SCORE.n_boot)
    SCORE.AUC_logit, SCORE.KSS_logit, SCORE.brier_logit, metrics_lg = get_metrics_sklearn(SCORE, 
                                                                      SCORE.predmodel_2,
                                                                      n_boot=SCORE.n_boot)

#    ex['score'] = 
    #%%
    return ex, SCORE



def create_validation_plot(outdic_folders, metric='AUC', getPEP=True):
#    from time_series_analysis import subplots_df
    #%%
    # each folder will become a single df to plot
    
    def get_data_PEP(datafolder, ex):
        if 'RV_aggregation' in ex.keys():    
            RVaggr = ex['RV_aggregation']
        else:
            RVaggr = 'RVfullts95'
        if datafolder == 'era5':
            folder = ('/Users/semvijverberg/surfdrive/McKinRepl/'
                      'era5_t2mmax_sst_Northern/era5_PEP_t2mmax_sst_PEPrectangle/'
                      f'random10fold_leave_4_out_1979_2018_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_{RVaggr}_rng50_2019-07-04/'
                      'lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p')
        elif datafolder == 'ERAint':
            folder = ('/Users/semvijverberg/surfdrive/MckinRepl/'
                      'ERAint_T2mmax_sst_Northern/ERAint_PEP_t2mmax_sst_PEPrectangle/'
                      f'random10fold_leave_4_out_1979_2017_tf1_stdp_1.0deg_60nyr_95tperc_0.8tc_{RVaggr}_rng50_2019-07-08/'
                      'lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p')            
        elif datafolder == 'EC':
            folder = ('/Users/semvijverberg/surfdrive/McKinRepl/'
                      'EC_tas_tos_Northern/EC_PEP_tas_tos_PEPrectangle/'
                      'random10fold_leave_16_out_2000_2159_tf1_95p_1.125deg_60nyr_1rmRV_2019-06-17/'
                      'lags[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]Ev1d0p')

        filename = 'output_main_dic_with_score'
        dic = np.load(os.path.join(folder, filename+'.npy'),  encoding='latin1').item()
        ex = dic['ex']
        scorecl_PEP = ex['score']
        return scorecl_PEP
        
#    import numpy as np, scipy.stats as st
#    from scipy.stats import kstest
#    from scipy.stats import ks_2samp
    # ks_2samp(scorecl.AUC[lag], ROC_array[-1])
    scoreclasses  = {}
    df_series     = []
    datasets      = []
    for folder in outdic_folders:
        filename = 'output_main_dic_with_score'
        dic = np.load(os.path.join(folder, filename+'.npy'),  encoding='latin1').item()
        ex = dic['ex']
        if ex['datafolder'] == 'era5':
            lags = ex['lags']
            
        scoreclasses[ex['datafolder']] = ex['score']
        AUC_means = ex['score'].AUC.mean(0)
        try:
            AUC_values = [AUC_means[l] if l in ex['lags'] else 0 for l in lags ]
        except:
            lags = ex['lags']
            AUC_values = [AUC_means[l] if l in ex['lags'] else 0 for l in lags ]
        df_series.append( AUC_values )
        datasets.append( ex['datafolder'] )
    
    df = pd.DataFrame(np.concatenate(np.array(df_series)[None,:], axis=0),
                      index=None, columns=lags)
    df['dataset'] = pd.Series(datasets, index=df.index)


    if metric=='AUC':
        y_lim = (0,1)
    elif metric=='brier':
        y_lim = (-0.3,0.5)
    g = sns.FacetGrid(df, col='dataset', size=6, aspect=1.4, ylim=y_lim,
                      sharex=False,  sharey=False, col_wrap=3)
    g.fig.subplots_adjust(wspace=0.1)
    
    titles = {'era5' : 'ERA-5 (40 yrs)',
              'ERAint': 'ERA-Interim (39 yrs)',
              'EC':      'EC-earth (160 yrs)'}
                      
    
                     
                      
    for i, ax in enumerate(g.axes.flatten()):
        
        dataset = datasets[i]
        scorecl = scoreclasses[dataset]
        if getPEP:
            try:
                score_PEP = get_data_PEP(dataset, ex)
                AUC_spatcov_PEP     = score_PEP.AUC_spatcov
                AUC_logit_PEP     = score_PEP.AUC_logit  
                brier_spatcov_PEP = score_PEP.brier_spatcov
                brier_logit_PEP   = score_PEP.brier_logit
            except:
                print('error loading PEP')
                pass

        try:
            
            AUC_spatcov = scorecl.AUC_spatcov
            AUC_logit = scorecl.AUC_logit        
           

            brier_spatcov = scorecl.brier_spatcov
            brier_logit = scorecl.brier_logit   

        except:
            AUC_spatcov, KSS_spatcov, brier_spatcov, metrics_sp = get_metrics_sklearn(scorecl, 
                                    scorecl.y_true_test, scorecl.predmodel_1, n_boot=10000)
            AUC_logit, KSS_logit, brier_logit, metrics_lg = get_metrics_sklearn(scorecl, 
                                    scorecl.y_true_test, scorecl.predmodel_2, n_boot=10000)

        if metric=='AUC':
            score_spatcov       = AUC_spatcov
            score_logit         = AUC_logit
        if metric=='AUC' and getPEP:
            score_spatcov_PEP   = AUC_spatcov_PEP
            score_logit_PEP     = AUC_logit_PEP
        if metric=='brier':
            score_spatcov       = brier_spatcov
            score_logit         = brier_logit
        if metric=='brier' and getPEP:        
            score_spatcov_PEP   = brier_spatcov_PEP
            score_logit_PEP     = brier_logit_PEP

        
        if metric =='AUC':
            # spatcov CPPA
            score_toplot = score_spatcov
            
            color = 'red'
            style = 'solid'
            x = np.array(score_toplot.columns.levels[0])
            ax.fill_between(x, score_toplot.loc['con_low'], score_toplot.loc['con_high'], linestyle='solid', 
                            edgecolor='black', facecolor=color, alpha=0.5)
            ax.plot(x, score_toplot.iloc[0], color=color, linestyle=style,
                    linewidth=2, label='Prec. Pattern spatcov' )   
            for tt in scorecl.AUC.index:
                ax.plot(x, scorecl.AUC.iloc[tt], color=color, linestyle=style,
                    linewidth=0.5, label='_nolegend_', alpha=0.5 )   
                
            if getPEP:
                # spatcov PEP
                score_toplot = score_spatcov_PEP
                color = 'blue'
                style = 'dashed'
                ax.fill_between(x, score_toplot.loc['con_low'], score_toplot.loc['con_high'], linestyle='solid', 
                                edgecolor='black', facecolor=color, alpha=0.3)
                ax.plot(x, score_toplot.iloc[0], color=color, linestyle=style, 
                        linewidth=2, label='PEP spatcov', alpha=0.35 ) 
                
                ax.hlines(y=0.5, xmin=x.min(), xmax=x.max())
            
        if metric =='brier':
            
            score_toplot = score_logit
            BSS = (score_toplot.loc['Brier_clim'] - score_toplot.loc['Brier']) / score_toplot.loc['Brier_clim']
            BSS_con_low = (score_toplot.loc['Brier_clim'] - score_toplot.loc['con_low']) / score_toplot.loc['Brier_clim']
            BSS_con_high = (score_toplot.loc['Brier_clim'] - score_toplot.loc['con_high']) / score_toplot.loc['Brier_clim']
            # logit CPPA
#            fig, ax = plt.subplots()
            x = np.array(score_toplot.columns.levels[0])
            color = 'red'
            style = 'solid'
            ax.fill_between(x, BSS_con_low, BSS_con_high, linestyle=style, 
                            edgecolor='black', facecolor=color, alpha=0.5)
            ax.plot(x, BSS, color=color, linestyle=style, 
                    linewidth=2, label='Prec. Pattern logit' )
            
            for tt in scorecl.BSS.index:
                ax.plot(x, scorecl.BSS.iloc[tt], color=color, linestyle=style,
                    linewidth=0.5, label='_nolegend_', alpha=0.5 )   
    
            if getPEP:
                # logit PEP
                score_toplot = score_logit_PEP
                BSS = (score_toplot.loc['Brier_clim'] - score_toplot.loc['Brier']) / score_toplot.loc['Brier_clim']
                BSS_con_low = (score_toplot.loc['Brier_clim'] - score_toplot.loc['con_low']) / score_toplot.loc['Brier_clim']
                BSS_con_high = (score_toplot.loc['Brier_clim'] - score_toplot.loc['con_high']) / score_toplot.loc['Brier_clim']
                
                color = 'blue'
                style = 'dashed'
                ax.fill_between(x, BSS_con_low, BSS_con_high, linestyle='solid', 
                                edgecolor='black', facecolor=color, alpha=0.3)
                ax.plot(x, BSS, color=color, linestyle=style, 
                        linewidth=2, label='PEP logit', alpha=0.35 ) 
                
                ax.hlines(y=0.0, xmin=x.min(), xmax=x.max())

#        # spatcov brier
#        ax2 = ax.twinx()
#        ax2.set_ylim(y_lim2)
#        ax2.set_ylabel('brier score')
#        score_toplot = brier_spatcov
#        ax2.fill_between(lags, score_toplot.loc['con_low'], score_toplot.loc['con_high'], linestyle='solid', 
#                        edgecolor='black', facecolor='red', alpha=0.5)
#        ax2.plot(lags, score_toplot.iloc[0], color='red', 
#                linewidth=2, label='brier loss score: spatcov Prec. Pattern' )       
#        
##        # logit brier
#        score_toplot = brier_logit
#        ax2.fill_between(lags, score_toplot.loc['con_low'], score_toplot.loc['con_high'], linestyle='solid', 
#                        edgecolor='black', facecolor='blue', alpha=0.5)
#        ax2.plot(lags, score_toplot.iloc[0], color='blue', 
#                linewidth=2, label='brier loss score: logit' )  
        
        
        ax.set_title(titles[dataset], fontdict={'fontsize':18, 'weight':'bold'})
        if metric == 'AUC': steps = 11
        if metric == 'brier': steps = 9
        yticks = np.round(np.linspace(y_lim[0], y_lim[1]+1E-9, steps),2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontdict={'fontsize':13})
        ax.set_xlim(min(x)-5,max(x)+5)
        ax.set_xticks(x)  
        ax.legend(loc=2)
        ax.set_xticklabels(x, fontdict={'fontsize':13})
        ax.set_xlabel('lead time [days]', fontdict={'fontsize':18, 'weight':'bold'})
        ax.grid(which='major', axis='y')
        if i == 0:
            if metric == 'brier':
                ax.set_ylabel('Brier Skill Score', fontdict={'fontsize':18, 'weight':'bold'})
            elif metric == 'AUC':
                ax.set_ylabel('Area under ROC', fontdict={'fontsize':18, 'weight':'bold'})
            else:
                ax.set_ylabel(metric, fontdict={'fontsize':18, 'weight':'bold'})
#    g.fig.text(0.5, 1.1, metric, fontsize=25,
#               fontweight='heavy', transform=g.fig.transFigure,
#               horizontalalignment='center',verticalalignment='top')

    
    lags_str = str(lags).replace(' ', '')
    try:
        scorecl_e5 = scoreclasses['era5']
    except:
        scorecl_e5 = scoreclasses[dataset]
    datasetstr = [''+i+'_' for i in datasets]
    fname = '{}_{}_{}_blocksize{}_{}'.format(metric, datasetstr, lags_str,
             scorecl_e5.bootstrap_size, scorecl_e5.n_boot)
    folder  = os.path.join(outdic_folders[0], 'validation')
    if os.path.isdir(folder) != True : os.makedirs(folder)
    filename = os.path.join(folder, fname)
    g.fig.savefig(filename ,dpi=600, frameon=True, bbox_inches='tight')
    #%%
    return

def add_training_data_lags(RV_ts_train, precursors_train_lags, SCORE, ex):
#%%
    
    for lag_idx, lag in enumerate(ex['lags']):
        precursors_train = precursors_train_lags[:,lag_idx]
        add_training_data(RV_ts_train, precursors_train, SCORE, lag_idx, ex)
        
#%%
    return

def add_training_data(RV_ts_train, precursors_train, SCORE, lag_idx, ex):
    #%%
    
    n = ex['n']
    l = lag_idx 
    keys_train = precursors_train.coords['variable'].values
    SCORE.Prec_train_mean[n,l] = precursors_train.mean(dim='time')
    SCORE.Prec_train_std[n,l]  = precursors_train.std(dim='time')
    
    norm = (precursors_train - SCORE.Prec_train_mean[n,l]) / \
                    SCORE.Prec_train_std[n,l]
    
    SCORE.RV_train[n][:RV_ts_train.size]  = RV_ts_train.values
#    SCORE.Prec_train[ex['n']][lag_idx][:RV_ts_train.size] = norm
    
    
    ts_RV = pd.DataFrame(RV_ts_train.values)
    #        if lag >= 30:
    #            ts_RV = ts_RV.rolling(1, center=True, min_periods=1).mean()
    
    if ex['event_percentile'] == 'std':
        # binary time serie when T95 exceeds 1 std
        threshold = ts_RV.mean().values + ts_RV.std().values
    else:
        percentile = ex['event_percentile']
        
        threshold = np.percentile(ts_RV.values, percentile)
    SCORE.RV_thresholds[ex['n']] = threshold
    events_idx = np.where(ts_RV.values > threshold)[0]
    y_true_train = func_CPPA.Ev_binary(events_idx, len(RV_ts_train),  
                                       ex['min_dur'], ex['max_break'], 
                                       grouped=SCORE.grouped)
    y_true_train[y_true_train!=0] = 1    
    
    SCORE.y_true_train[n][:RV_ts_train.size] = y_true_train
    
    def add_pred_model_1(norm, ex):
        if norm.shape[1] > 1:
            model, p_pred = add_pred_model_2(norm, ex)

        else:
            model = 'spatcov'
            pthresholds = np.linspace(1, 9, 9, dtype=int)
            p_pred = []
            for p in pthresholds:	
                p_pred.append(np.percentile(norm, p*10, axis=0))
            
        return model, p_pred
    
    
    # Prediction model 2 - logitstic model 
    def add_pred_model_2(norm, ex):
        X = pd.DataFrame(norm.values, columns=keys_train )
        X = add_constant(X)
        model_set = sm.Logit(y_true_train, X, disp=0)
        model = model_set.fit( disp=0 )
        
        prediction_train = model.predict(X)
        
        pthresholds = np.linspace(1, 9, 9, dtype=int)
        p_pred = []
        for p in pthresholds:	
            p_pred.append(np.percentile(prediction_train, p*10, axis=0))
            
        
        return model, p_pred

    SCORE.model_1[n][l], SCORE.xrperc_m1[n][l] = add_pred_model_1(norm, ex)
    SCORE.model_2[n][l], SCORE.xrperc_m2[n][l] = add_pred_model_2(norm, ex)
    #%%
    return

def add_test_data(RV_ts_test, precursor_test, SCORE, ex):
    #%%
    # =============================================================================
    # calc ROC scores
    # =============================================================================   
    dates_test = pd.to_datetime(RV_ts_test.time.values)
    ts_RV    = pd.DataFrame(RV_ts_test.values)
    n = ex['n'] 
    
    
    events_idx = np.where(RV_ts_test.values > SCORE.RV_thresholds[ex['n']])[0]
    y_true = func_CPPA.Ev_binary(events_idx, RV_ts_test.size,  ex['min_dur'], 
                             ex['max_break'], grouped=SCORE.grouped)
    y_true[y_true!=0] = 1
    events_train = SCORE.y_true_train[ex['n']][SCORE.y_true_train[ex['n']]!=0].size
    y_true_train_clim = np.repeat( events_train/SCORE.y_true_train[ex['n']].size, y_true.size )    

    SCORE.RV_test.loc[dates_test, 0]    = pd.Series(ts_RV.values.ravel(), 
                                           index=dates_test)
    SCORE.y_true_test.loc[dates_test, 0] = pd.Series(y_true, 
                                           index=dates_test)
    SCORE.y_true_train_clim.loc[dates_test, 0] = pd.Series(y_true_train_clim, 
                                           index=dates_test)
    
    def add_predict_m2(model, ts_test):       
        '''making prediction for dates_test based upon model in 
        SCORE.model_2 '''
        
        keys_test = model.params.keys()                    
        X_pred = pd.DataFrame(ts_test.values, columns=keys_test[1:])
        X_pred = add_constant(X_pred)
        prediction = SCORE.model_2[ex['n']][l].predict(X_pred).values
        return prediction
  
    for l, lag in enumerate(ex['lags']):
              
        precursors_test = precursor_test[:,l]
        norm  = (precursors_test - SCORE.Prec_train_mean[n,l]) / \
                    SCORE.Prec_train_std[n,l]
                    
        # add 'model' 1                    
        if norm.shape[1] > 1:   
            model = SCORE.model_2[ex['n']][l]
            predmodel_1 = add_predict_m2(model, norm)
        else:
            # testing raw precusor
            predmodel_1 = norm.values
            
      
        
        model = SCORE.model_2[ex['n']][l]
        predmodel_2 = add_predict_m2(model, norm)
        
        
        if SCORE.score_per_fold:
            dates_tofill = dates_test
        elif SCORE.notraintest:
            dates_tofill = ex['dates_RV'] 
      
        SCORE.predmodel_1_mean[n,l]   = np.mean(predmodel_1)                
        SCORE.predmodel_1_std[n,l]    = np.std(predmodel_1)            
        SCORE.predmodel_1.loc[dates_tofill, lag]  = pd.Series(predmodel_1, 
                                               index=dates_tofill)
        SCORE.predmodel_2.loc[dates_tofill, lag]= pd.Series(predmodel_2, 
                                               index=dates_tofill)
    #%%
    return


def get_scores_per_fold(SCORE):
    #%%
    dates_test = SCORE.RV_test.index
    for f in range(SCORE._n_conv):
        dates_test = SCORE.dates_test[f]
        y_true_train_clim_c = SCORE.y_true_train_clim.loc[dates_test]
        
        for lag_idx, lag in enumerate(SCORE._lags):
        
            
            if SCORE.score_per_fold:
                # calculating scores per fold
                ts_pred_c = SCORE.predmodel_1.loc[dates_test, lag]
                ts_logit_c = SCORE.predmodel_2.loc[dates_test, lag]
                y_true_c = SCORE.y_true_test.loc[dates_test, 0]
                

            elif SCORE.score_per_fold == False and f == SCORE._n_conv-1:
                # calculating scores over all test data
                ts_pred_c = SCORE.predmodel_1[lag]
                ts_logit_c = SCORE.predmodel_2[lag]
                y_true_c = SCORE.y_true[0]
                y_true_train_clim_c = SCORE.y_true_train_clim[0]
                f = 0
    
            if lag_idx == 0:
                SCORE.Prec_len  = ts_pred_c.size
                SCORE.RV_len    = y_true_c.size
                SCORE.n_events  = y_true_c[y_true_c!=0].sum()            
                print('Calculating ROC scores\nDatapoints precursor length '
                  '{}\nDatapoints RV length {}, with {:.0f} events'.format(ts_pred_c.size,
                   len(y_true_c), y_true_c[y_true_c!=0].sum()))
            
            if np.unique(y_true_c).size != 2:
                print('Fold with no events')
                AUC_score, FP, TP, ROCboot, KSSboot = [np.nan] * 5
                break
            
            percentiles_train = SCORE.xrperc_m1[f].sel(lag=lag)
            AUC_score, FP, TP, ROCboot, KSSboot = func_AUC(ts_pred_c, y_true_c,
                                            n_boot=0, win=0, 
                                            n_blocks=
                                            int(y_true_c.size / SCORE.bootstrap_size), 
                                            thr_pred=percentiles_train)
            SCORE.AUC.iloc[f][lag]        = AUC_score
            
            # Scores of probabilistic logit (model) 
            metrics = metrics_sklearn(y_true_c, ts_logit_c, y_true_train_clim_c, n_boot=100, blocksize=SCORE.bootstrap_size)
    
            brier_score, brier_clim, ci_low_brier, ci_high_brier, sorted_briers = metrics['brier'] 
            FP, TP, thresholds = metrics['fpr_tpr_thres']
            SCORE.BSS.iloc[f][lag]        = (brier_clim-brier_score)/brier_clim
            SCORE.KSS.iloc[f][lag]        = get_KSS(TP, FP)
            SCORE.FP_TP[f,lag_idx]        = FP, TP 
            SCORE.ROC_boot[f,lag_idx, :]  = ROCboot
            SCORE.KSS_boot[f,lag_idx, :]  = KSSboot
    #%%
    return
 

def get_KSS(TP, FP):
    '''Hansen Kuiper Skill Score from True & False positive rate'''
    KSS_allthreshold = np.zeros(len(TP))
    for i in range(len(TP)):
        KSS_allthreshold[i] = TP[i] - FP[i]
    return max(KSS_allthreshold)


def calculate_spatcov(Prec_reg, lags, dates, ds_Sem, PEPpattern=False):
    '''
    Returns spatial covariance for dates at specific lead times 
    spatcov has shape (dates, lags)
    '''
    spatcov = np.zeros( (dates.size, len(lags)) )
    
    for lag_idx, lag in enumerate(lags):
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates - pd.Timedelta(int(lag), unit='d')
        var_train_reg = Prec_reg.sel(time=dates_min_lag)   
        
        if PEPpattern == False:
            # weight by robustness of precursors
            var_train_reg = var_train_reg * ds_Sem['weights'].sel(lag=lag)
            spatcov[:,lag_idx] = func_CPPA.cross_correlation_patterns(var_train_reg, 
                                                            ds_Sem['pattern_CPPA'].sel(lag=lag))
        elif PEPpattern == True:
            var_patt_mcK = func_CPPA.find_region(ds_Sem['pattern'].sel(lag=lag), 
                                                 region='PEPrectangle')[0]
            spatcov[:,lag_idx] = func_CPPA.cross_correlation_patterns(var_train_reg, 
                                                            var_patt_mcK)     
    return spatcov





def func_AUC(predictions, obs_binary, n_boot=0, win=0, n_blocks=39, thr_pred='default'):
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
#        pval       = (np.asarray(np.where(AUC_new > func_AUC)).size)/ n_boot
        ROC_bootstrap = AUC_new 
    #%%
    return ROC_bootstrap, KSS_bootstrap


# =============================================================================
# =============================================================================
# Plotting
# =============================================================================
# =============================================================================
def plotting_timeseries(SCORE, name_pred, yrs_to_plot, ex):
    if name_pred == 'spatcov':
        prec_test = SCORE.predmodel_1
    elif name_pred == 'logit':
        prec_test = SCORE.predmodel_2
#    
    for idx, lag in enumerate(ex['lags']):
        #%%
        # normalize
     
        norm_test_RV = (( SCORE.RV_test[lag]-SCORE.RV_test[lag].mean() )/ \
                                  (SCORE.RV_test[lag].std()) ) 
        ts_pred_Sem = ((prec_test[lag]-np.mean(prec_test[lag]))/ \
                                  (np.std(prec_test[lag]) ) ) 
        labels       = pd.to_datetime(ex['dates_RV'])
            
        
#        threshold = np.std(norm_test_RV)
        if ex['event_percentile'] == 'std':
            # binary time serie when T95 exceeds 1 std
            threshold = np.std(norm_test_RV) 
        else:
            percentile = ex['event_percentile']
            threshold= np.percentile(norm_test_RV, percentile)
        
#        No_train_yrs = ex['n_yrs'] - int(SCORE.RV_test[lag].size / ex['n_oneyr'])
        title = ('Prediction time series versus truth (lag={}) '
                 ' '.format(lag))
        years = labels.year
        years.values[-1] = ex['endyear']+1
#                    years = np.concatenate((labels, [labels[-1]+1]))
        df = pd.DataFrame(data={'RV':norm_test_RV, name_pred:ts_pred_Sem, 
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
            ax.plot(df_sub['date'],df_sub[name_pred], linewidth=2,
                    label=name_pred, color='blue', alpha=0.9)
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
        filename = f'{name_pred} {lag} day lead time series prediction'
        path = ex['output_dic_folder']
        file_name = os.path.join(path,filename+'.png')
        g.fig.savefig(file_name ,dpi=250, frameon=True)
        plt.show()
        #%%

def get_metrics_sklearn(SCORE, y_pred_all, alpha=0.05, n_boot=5):
    #%%
    y_true = SCORE.y_true_test[0]
    y_true_train_clim = SCORE.y_true_train_clim[0]

    
    df_auc = pd.DataFrame(data=np.zeros( (3, len(SCORE._lags)) ), columns=[SCORE._lags],
                          index=['AUC', 'con_low', 'con_high'])
    df_KSS = pd.DataFrame(data=np.zeros( (3, len(SCORE._lags)) ), columns=[SCORE._lags],
                          index=['KSS', 'con_low', 'con_high'])
    
    df_brier = pd.DataFrame(data=np.zeros( (7, len(SCORE._lags)) ), columns=[SCORE._lags],
                          index=['BSS', 'BSS_low', 'BSS_high', 'Brier', 'con_low', 'con_high', 'Brier_clim'])
    
    
    for lag in SCORE._lags:
        y_pred = y_pred_all[lag]

        metrics = metrics_sklearn(
                    y_true, y_pred, y_true_train_clim,
                    alpha=alpha, n_boot=n_boot, blocksize=SCORE.bootstrap_size)
        # AUC
        AUC_score, conf_lower, conf_upper, sorted_AUC = metrics['AUC']
        df_auc[lag] = (AUC_score, conf_lower, conf_upper) 
        # HKSS
        KSS_score, ci_low_KSS, ci_high_KSS, sorted_KSSs = metrics['KSS']
        df_KSS[lag] = (KSS_score, ci_low_KSS, ci_high_KSS)
        # Brier score
        brier_score, brier_clim, ci_low_brier, ci_high_brier, sorted_briers = metrics['brier']
        BSS = (brier_clim - brier_score) / brier_clim
        BSS_low = (brier_clim - ci_low_brier) / brier_clim
        BSS_high = (brier_clim - ci_high_brier) / brier_clim        
        df_brier[lag] = (BSS, BSS_low, BSS_high, 
                        brier_score, ci_low_brier, ci_high_brier, brier_clim)
    print("Original ROC area: {:0.3f}".format( float(df_auc.iloc[0][0]) ))
    #%%
    return df_auc, df_KSS, df_brier, metrics


def get_KSS_clim(y_true, y_pred, threshold_clim_events):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    idx_clim_events = np.argmin(abs(thresholds[::-1] - threshold_clim_events))
    KSS_score = tpr[idx_clim_events] - fpr[idx_clim_events]
    return KSS_score 

def metrics_sklearn(y_true, y_pred, y_true_train_clim, alpha=0.05, n_boot=5, blocksize=1):
#    y_true, y_pred, y_true_train_clim = y_true_c, ts_logit_c, y_true_train_clim_c
    #%%
    metrics = {}
    AUC_score = roc_auc_score(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    threshold_clim_events = np.sort(y_pred)[::-1][y_true[y_true>0.5].size]
    
    KSS_score = get_KSS_clim(y_true, y_pred, threshold_clim_events)

    # convert y_pred to fake probabilities if spatcov is given
    if y_pred.max() > 1 or y_pred.min() < 0: 
        y_pred = (y_pred+abs(y_pred.min()))/( y_pred.max()+abs(y_pred.min()) )
    else:
        y_pred = y_pred
        
    brier_score = brier_score_loss(y_true, y_pred)

    brier_score_clim = brier_score_loss(y_true, y_true_train_clim)
    
    rng_seed = 42  # control reproducibility
    bootstrapped_AUC = []
    bootstrapped_KSS = []
    bootstrapped_brier = []   
    
    
    old_index = range(0,len(y_pred),1)
    n_bl = blocksize
    chunks = [old_index[n_bl*i:n_bl*(i+1)] for i in range(int(len(old_index)/n_bl))]

    rng = np.random.RandomState(rng_seed)
#    random.seed(rng_seed)
    for i in range(n_boot):
        # bootstrap by sampling with replacement on the prediction indices
        ran_ind = rng.randint(0, len(chunks) - 1, len(chunks))
        ran_blok = [chunks[i] for i in ran_ind]
#        indices = random.sample(chunks, len(chunks))
        indices = list(chain.from_iterable(ran_blok))
        
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score_AUC = roc_auc_score(y_true[indices], y_pred[indices])
        score_KSS = get_KSS_clim(y_true[indices], y_pred[indices], threshold_clim_events)
        score_brier = brier_score_loss(y_true[indices], y_pred[indices])
        
        bootstrapped_AUC.append(score_AUC)
        bootstrapped_KSS.append(score_KSS)
        bootstrapped_brier.append(score_brier)
#        print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    def get_ci(bootstrapped, alpha=alpha):
        sorted_scores = np.array(bootstrapped)
        sorted_scores.sort()
        ci_low = sorted_scores[int(alpha * len(sorted_scores))]
        ci_high = sorted_scores[int((1-alpha) * len(sorted_scores))]
        return ci_low, ci_high, sorted_scores
    
    if len(bootstrapped_AUC) != 0:
        ci_low_AUC, ci_high_AUC, sorted_AUCs = get_ci(bootstrapped_AUC, alpha)
        
        ci_low_KSS, ci_high_KSS, sorted_KSSs = get_ci(bootstrapped_KSS, alpha)
        
        ci_low_brier, ci_high_brier, sorted_briers = get_ci(bootstrapped_brier, alpha)
    else:
        ci_low_AUC, ci_high_AUC, sorted_AUCs = (AUC_score, AUC_score, [AUC_score])
        
        ci_low_KSS, ci_high_KSS, sorted_KSSs = (KSS_score, KSS_score, [KSS_score])
        
        ci_low_brier, ci_high_brier, sorted_briers = (brier_score, brier_score, [brier_score])
    
   
    metrics['AUC'] = (AUC_score, ci_low_AUC, ci_high_AUC, sorted_AUCs)
    metrics['KSS'] = (KSS_score, ci_low_KSS, ci_high_KSS, sorted_KSSs)
    metrics['brier'] = (brier_score, brier_score_clim, ci_low_brier, ci_high_brier, sorted_briers)
    metrics['fpr_tpr_thres'] = fpr, tpr, thresholds
#    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
#        confidence_lower, confidence_upper))
    #%%
    return metrics


def get_logit_stat(SCORE, ex):
    log_models = SCORE.logitmodel
    pval = np.zeros( log_models.shape )
    odds = np.zeros( log_models.shape )
    for n in range(ex['n_conv']):
        
        for i, l in enumerate(ex['lags']):
            model = log_models[n,i]
            pval[n,i] = model.pvalues
            odds[n,i] = np.exp(model.params)
    SCORE.df_pval = pd.DataFrame(pval, columns=[ex['lags']])
    SCORE.df_odds = pd.DataFrame(odds, columns=[ex['lags']])
    return SCORE

def autocorrelation(x):
    xp = (x - np.mean(x))/np.std(x)
    result = np.correlate(xp, xp, mode='full')
    return result[int(result.size/2):]/(len(xp))
#%%
    
def autocorr_sm(ts, max_lag=None, alpha=0.01):
    import statsmodels as sm
    if max_lag == None:
        max_lag = ts.size
    ac, con_int = sm.tsa.stattools.acf(ts.values, nlags=max_lag-1, 
                                unbiased=True, alpha=0.01, 
                                 fft=True)
    return (ac, con_int)

def get_bstrap_size(ts, max_lag=200, n=1):
    max_lag = min(max_lag, ts.size)
    ac, con_int = autocorr_sm(ts, max_lag=max_lag, alpha=0.01)
    plt.figure()
    # con high
    plt.plot(con_int[:,1][:20], color='orange')
    # ac values
    plt.plot(range(20),ac[:20])
    # con low
    plt.plot(con_int[:,0][:20], color='orange')
    where = np.where(con_int[:,0] < 0 )[0]
    # has to be below 0 for n times (not necessarily consecutive):
    n_of_times = np.array([idx+1 - where[0] for idx in where])
    cutoff = where[np.where(n_of_times == n)[0][0] ]
    return cutoff


def get_ts_matrix(Prec_reg, final_pattern, ex, lag=0):
    filename = os.path.join(ex['RV1d_ts_path'], ex['RVts_filename'])        
    RVtsfull95 = load_1d(filename, ex, 'RVfullts95')[0]
    RVtsfull95 = func_CPPA.remove_leapdays(RVtsfull95)
#    RVtsfull_mean = load_1d(filename, ex, 'RVfullts_mean')[0]  
    
    PEP_box = func_CPPA.find_region(Prec_reg, region='PEPrectangle')[0]
    PEP_pattern = PEP_box.sel(time=ex['dates_RV']).mean(dim='time')
    PEP = func_CPPA.cross_correlation_patterns(PEP_box, PEP_pattern).values
                                           
    Prec_Patt = func_CPPA.cross_correlation_patterns(Prec_reg, 
                                         final_pattern.sel(lag=lag)).values
    
    ts_3d_nino = func_CPPA.find_region(Prec_reg, region='elnino3.4')[0]
    nino_index = func_CPPA.area_weighted(ts_3d_nino).mean(
            dim=('latitude', 'longitude'), skipna=True)

##    seasonal
#    prec_pat_seasons = crosscorr_Sem.groupby('time.season').mean(dim='time')
#    nino_season = nino_index.groupby('time.season').mean(dim='time')
#    temp_seaon        = RVtsfull_mean.groupby('time.season').mean()
#    corr_seasonal_mean = np.corrcoef(prec_pat_seasons, nino_season)
#    corr_seasonal_grad = np.corrcoef(prec_pat_seasons, np.gradient(nino_season))
#    corr_seasonal_grad_t= np.corrcoef(temp_seaon, np.gradient(nino_season))
    
    cc_dir = os.path.join('/Users/semvijverberg/surfdrive/McKinRepl/', 
                         'cross_corr_matrix')            
    PDO_pattern, PDO_ts = func_CPPA.get_PDO(Prec_reg)

    func_CPPA.xarray_plot(PDO_pattern, path=cc_dir, name = 'PDO_pattern', saving=True)
    dates = pd.to_datetime(Prec_reg.time.values)
    dates -= pd.Timedelta(dates.hour[0], unit='h')

    data = np.stack([RVtsfull95, PEP, Prec_Patt, PDO_ts.values, nino_index])
    df = pd.DataFrame(data=data.swapaxes(1,0), index=pd.DatetimeIndex(dates),
                      columns=['T95 E-U.S.', 'PEP', 
                               'Prec. Pattern', 'PDO', 'Nino3.4'])
    
    return df

def build_matrix_wrapper(df_init, ex, lag=[0], wins=[1, 20]):
    
    periods = ['fullyear', 'summer60days', 'pre60days']
    for win in wins:
        for period in periods:
            build_ts_matric(df_init, ex, win=win, lag=0, period=period)

def build_ts_matric(df_init, ex, win=20, lag=0, period='fullyear'):
    #%%
    # RV ts
    df = df_init.copy()

    # shift precursor vs. tmax 
    for c in df.columns[2:]:
        df[c] = df[c].shift(periods=-lag)
    # dates that still overlap:
    dates_overlap = df[c].dropna().index
    df = df.loc[dates_overlap]
    
    dates = df.index
    df = df.set_index(pd.DatetimeIndex(dates))
    # bin means
    df = df.resample(f'{win}D').mean()
    
    # add gradient nino
#    df['Nino3.4 grad'] = np.gradient(df['Nino3.4'], edge_order=1)
    dates = df.index
    
    if period=='fullyear':
        dates_sel = dates.strftime('%Y-%m-%d')
    elif period == 'summer60days':
        dates_sel = ex['dates_RV'].strftime('%Y-%m-%d')
    elif period == 'pre60days':
        dates_sel = (ex['dates_RV'] - pd.Timedelta(60, unit='d')).strftime('%Y-%m-%d')

    # after resampling, not all dates are in their:
    dates_sel =  pd.to_datetime([d for d in dates_sel if d in dates] )

    df_period = df.loc[dates_sel].dropna()
    corr, sig_mask, pvals = func_CPPA.corr_matrix_pval(df_period, alpha=0.01)
    # Generate a mask for the upper triangle
    mask_tri = np.ones_like(corr, dtype=np.bool)
    
    mask_tri[np.triu_indices_from(mask_tri)] = False
    mask_sig = mask_tri.copy()
    mask_sig[sig_mask==False] = True
  
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, n=9, l=30, as_cmap=True)
    
    
    ax = sns.heatmap(corr, ax=ax, mask=mask_tri, cmap=cmap, vmax=1E99, center=0,
                square=True, linewidths=.5, 
                 annot=False, annot_kws={'size':30}, cbar=False)
    
    
    sig_bold_labels = sig_bold_annot(corr, mask_sig)
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(corr, ax=ax, mask=mask_tri, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8},
                 annot=sig_bold_labels, annot_kws={'size':30}, cbar=False, fmt='s')
    ax.tick_params(axis='both', labelsize=15, labeltop=True, 
                   bottom=False, top=True, left=False, right=True,
                   labelbottom=False, labelleft=False, labelright=True)
    ax.set_xticklabels(corr.columns, fontdict={'fontweight':'bold'})
    ax.set_yticklabels(corr.columns, fontdict={'fontweight':'bold'}, rotation=0)  
    
    fname = f'cross_corr_{win}dmean_{period}.png'
    cc_dir = os.path.join('/Users/semvijverberg/surfdrive/McKinRepl/', 
                         'cross_corr_matrix')
    if os.path.isdir(cc_dir)==False: os.makedirs(cc_dir) 
    plt.savefig(os.path.join(cc_dir, fname), dpi=300)
    #%%
def sig_bold_annot(corr, pvals):
    corr_str = np.zeros_like( corr, dtype=str ).tolist()
    for i1, r in enumerate(corr.values):
        for i2, c in enumerate(r):
            if pvals[i1, i2] <= 0.05 and pvals[i1, i2] > 0.01:
                corr_str[i1][i2] = '{:.2f}*'.format(c)
            if pvals[i1, i2] <= 0.01:
                corr_str[i1][i2]= '{:.2f}**'.format(c)        
            elif pvals[i1, i2] > 0.05: 
                corr_str[i1][i2]= '{:.2f}'.format(c)
 
    return np.array(corr_str)              
    
    #%%
def add_scores_to_class(SCORE, n_shuffle=1000):
    #%%
    y_true = SCORE.y_true_test
#    thresholds = [[(1-y_true[y_true.values==1.].size / y_true.size)]]
#    thresholds.append([t/10. for t in range(2, 10, 2)])
    thresholds = [t/100. for t in range(25, 100, 25)]
#    thresholds = flatten(thresholds)
    
    
    list_dfs = []
    for lag in SCORE._lags:
        metric_names = ['Precision', 'Specifity', 'Recall', 'FPR', 'F1_score', 'Accuracy']
        stats = ['fc', 'fc shuf', 'best shuf', 'impr.'] 
        stats_keys = stats * len(thresholds)
        thres_keys = np.repeat(thresholds, len(stats))
        data = np.zeros( (len(metric_names), len(stats_keys) ) )
        df_lag = pd.DataFrame(data=data, dtype=str, 
                         columns=pd.MultiIndex.from_tuples(zip(thres_keys,stats_keys)),
                          index=metric_names)
    
        for t in thresholds:
            y_pred = SCORE.predmodel_2[lag]
            y_pred = y_pred > np.percentile(y_pred.values, 100*t)
            prec_f = metrics.precision_score(y_true, y_pred)
            recall_f = metrics.recall_score(y_true, y_pred)
    #        cm = metrics.confusion_matrix(y_true,  y_pred_lags[l])
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
            FPR_f = fp / (fp + tn)
            SP_f = tn / (tn + fp)
            Acc_f = metrics.accuracy_score(y_true, y_pred)
            f1_f = metrics.f1_score(y_true, y_pred)
            # shuffle the predictions 
            prec = [] ; recall = [] ; FPR = [] ; SP = [] ; Acc = [] ; f1 = []
            for i in range(n_shuffle):
                np.random.shuffle(y_pred); 
                prec.append(metrics.precision_score(y_true, y_pred))
                recall.append(metrics.recall_score(y_true, y_pred))
                tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
                FPR.append(fp / (fp + tn))
                SP.append(tn / (tn + fp))
                Acc.append(metrics.accuracy_score(y_true, y_pred))
                f1.append(metrics.f1_score(y_true, y_pred))
            
            df_lag.loc['Precision'][t] = pd.Series([prec_f, np.mean(prec),
                                          np.percentile(prec, 97.5), prec_f/np.mean(prec)],
                                            index=stats)
                                                     
    
            df_lag.loc['Recall'][t]  = pd.Series([recall_f, np.mean(recall),
                                                  np.percentile(recall, 97.5),
                                                     recall_f/np.mean(recall)],
                                                    index=stats)
    #        cm = metrics.confusion_matrix(y_true,  y_pred_lags[l])
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
            df_lag.loc['FPR'][t] = pd.Series([FPR_f, np.mean(FPR), np.percentile(FPR, 2.5),
                                                     np.mean(FPR)/FPR_f], index=stats)
                                                     
            df_lag.loc['Specifity'][t] = pd.Series([SP_f, np.mean(SP), np.percentile(SP, 97.5), 
                                                     SP_f/np.mean(SP)], index=stats)
            df_lag.loc['Accuracy'][t] = pd.Series([Acc_f, np.mean(Acc), np.percentile(Acc, 97.5), 
                                                      Acc_f/np.mean(Acc)], index=stats)
            df_lag.loc['F1_score'][t] = pd.Series([f1_f, np.mean(f1), np.percentile(f1, 97.5),
                                                     f1_f/np.mean(f1)], index=stats)
        
        df_lag['mean_impr'] = df_lag.iloc[:, df_lag.columns.get_level_values(1)=='impr.'].mean(axis=1)

        list_dfs.append(df_lag)

    df_sum = pd.concat(list_dfs, keys= SCORE._lags)
    #%%

    return df_sum
    
def CV_wrapper(RV_ts, ex, path_ts, lag_to_load=0, keys=['spatcov_CPPA']):
    #%%
    '''
    Function input:
        ex['method'] = 'random10fold', 'iter', 'no_train_test_split'
        keys = columns of csv file to load 
        for ERA5 [2,3,4,5,6,7]
            East-Pac = 3
            CARIBEAN = 6
            MONSOON = 5
            LAKES = 2
            ICELAND = 7
            Mid-Pac = 4 
    '''
    
    if ex['method'][:6] == 'random': ex['score_per_fold'] = True
    ex['size_test'] = ex['train_test_list'][0][1]['RV'].size
    # init class
    SCORE = SCORE_CLASS(ex)
    #    SCORE.n_boot = 100
    print(f"n bootstrap is: {SCORE.n_boot}")
    
    ex['pred_names'] = keys
    all_y_RV = SCORE.RV_ts.time.dt.year.values
    size_test = int(round(SCORE._n_yrs_test / ex['n_yrs'], 1) * all_y_RV.size)
    SCORE.dates_test = np.zeros( (SCORE._n_conv, size_test), dtype='datetime64[D]')
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n     
        
        test  = ex['train_test_list'][n][1]
        RV_ts_test = test['RV']
        test_yrs = list(set(test['RV'].time.dt.year.values))
        
        test_y_i = [i for i in range(len(all_y_RV)) if all_y_RV[i] in test_yrs]
#        RVfullts_test = SCORE.RVfullts.isel(time=test_y_i)
        
        RV_ts_test = SCORE.RV_ts.isel(time=test_y_i)
        dates_test = pd.to_datetime(RV_ts_test.time.values)
        SCORE.dates_test[ex['n']] = dates_test
#        dates_test_str = [d[:10] for d in dates_test.strftime('%Y-%m-%d')]

        print('Test years {}'.format( np.unique(pd.to_datetime(dates_test).year)) )
        
        dates_train = pd.to_datetime([i for i in SCORE.dates_RV if i not in dates_test])
        RV_ts_train = SCORE.RV_ts.sel(time=dates_train)
        RV_ts_test  = SCORE.RV_ts.sel(time=dates_test)
        csv_train_test_data = 'testyr{}_{}.csv'.format(test_yrs, 0)
        path = os.path.join(path_ts, csv_train_test_data)
        data = pd.read_csv(path, index_col='date', infer_datetime_format=True)
        data.index = pd.to_datetime(data.index)
        dates_tfreq = func_CPPA.timeseries_tofit_bins(
                                data.index, ex, ex['tfreq'], seldays='all', verb=0)
        data = data.loc[dates_tfreq]
                
        def add_other_ts(filenames, ex):
            
            for path_csv in filenames:
                data = pd.read_csv(path_csv, index_col='date', infer_datetime_format=True)
                data.index = pd.to_datetime(data.index)
                dates_tfreq = func_CPPA.timeseries_tofit_bins(
                                        data.index, ex, ex['tfreq'], seldays='all', verb=0)
                data = data.loc[dates_tfreq]
            return data
            
        
        
        def get_ts(data, SCORE, dates, keys, tfreq):           
            csv_train_test_data = 'testyr{}_{}.csv'.format(test_yrs, lag_to_load)
            path = os.path.join(path_ts, csv_train_test_data)
            data = pd.read_csv(path, index_col='date')

            data.index = pd.to_datetime(data.index)
            xarray = data.to_xarray().to_array().rename({'date':'time'})
            # check if ts in csv
            keys_ = [k for k in keys if k in data.columns]
            
            if tfreq != 1:
                xarray, dates_all = func_CPPA.time_mean_bins(xarray, ex, ex['tfreq'])
            # time mean bins 
            xr_ts = xr.DataArray(np.zeros( (dates.size, len(SCORE._lags), len(keys_)) ), 
                                 dims=['time', 'lag', 'variable'],  
                                 coords=[dates, SCORE._lags, keys_])
            for l, lag in enumerate(SCORE._lags):
                dates_min_lag = pd.to_datetime(dates - pd.Timedelta(l*ex['tfreq'], unit='d'))
                if dates_min_lag.min() < SCORE.dates_all.min():
                    print('lag too large')
                xr_ts.values[:,l,:] = xarray.sel(variable=keys_).sel(time=dates_min_lag).values.swapaxes(0,1)
                    
            return xr_ts
        

        precursors_train_lags = get_ts(data, SCORE, dates_train, keys, ex['tfreq'])
        add_training_data_lags(RV_ts_train, precursors_train_lags, SCORE, ex)
        
        precursor_test = get_ts(data, SCORE, dates_test, keys, ex['tfreq'])
        add_test_data(RV_ts_test, precursor_test, SCORE, ex)
        
   
    print('Calculate scores')
    

#    # test temp aggregation after fitting
#    to_freq = 1
#
#    def change_tfreq(df_in, to_freq=int):
#        df_in.index.name = 'time'
#        xarr = df_in.to_xarray().to_array()
#        
#        xarr, dates_tfreq = func_CPPA.timeseries_tofit_bins(
#                            xarr, ex, to_freq, 
#                            seldays='all', verb=0)
#        if to_freq != 1:
#            xarr, dates_all = func_CPPA.time_mean_bins(xarr, ex, to_freq)
#        
#        df_new = xarr.to_dataframe(name='timeseries')
#        df_new = df_new.reset_index(level='variable').rename(columns={'variable':'lag'})
#        df_new = df_new.pivot_table(values='timeseries', index='time', columns='lag')
#        return df_new
#    
#    if to_freq != 1:
#        SCORE.predmodel_1 = change_tfreq(SCORE.predmodel_1, to_freq=to_freq)
#        SCORE.predmodel_2 = change_tfreq(SCORE.predmodel_2, to_freq=to_freq)
#        SCORE.y_true_test = change_tfreq(SCORE.y_true_test, to_freq=to_freq)
#        SCORE.y_true_test  = SCORE.y_true_test > 0.4
#        SCORE.y_true_train_clim = change_tfreq(SCORE.y_true_train_clim, to_freq=to_freq)

    get_scores_per_fold(SCORE)

    SCORE.AUC_spatcov, SCORE.KSS_spatcov, SCORE.brier_spatcov, metrics_sp = get_metrics_sklearn(SCORE,
                                                                            SCORE.predmodel_1,
                                                                            n_boot=SCORE.n_boot)
    SCORE.AUC_logit, SCORE.KSS_logit, SCORE.brier_logit, metrics_lg = get_metrics_sklearn(SCORE, 
                                                                      SCORE.predmodel_2,
                                                                      n_boot=SCORE.n_boot)
    
    ex['SCORE'] = SCORE
#    RVfullts = xr.DataArray(data=onlytest.values.squeeze(), coords=[dates], dims=['time'])
#    filename = '{}_only_test_ts'.format(key1)
#    try:
#        to_dict = dict( { 'mask'      : ex['mask'],
#                     'RVfullts'   : RVfullts} )
#    except:
#        to_dict = dict( {'RVfullts'   : RVfullts} )
#    np.save(os.path.join(ex['output_ts_folder'], filename+'.npy'), to_dict)  
    #%%
    return SCORE, ex

def plot_score_freq(dict_tfreq, metric, lags_to_test):
    x = list(dict_tfreq.keys())
#    x= dict_tfreq[1].lags.values
    if metric == 'BSS':
        y_lim = (-0.6, 0.6)
        y = np.array([dict_tfreq[f].brier_logit.loc['BSS'].values for f in x])
        y_min = np.array([dict_tfreq[f].brier_logit.loc['BSS_high'].values for f in x])
        y_max = np.array([dict_tfreq[f].brier_logit.loc['BSS_low'].values for f in x])
    elif metric == 'AUC':
        y_lim = (0.4,1.0)
        y = np.array([dict_tfreq[f].AUC_spatcov.loc['AUC'].values for f in x])
        y_min = np.array([dict_tfreq[f].AUC_spatcov.loc['con_low'].values for f in x])
        y_max = np.array([dict_tfreq[f].AUC_spatcov.loc['con_high'].values for f in x])
    df = pd.DataFrame(np.swapaxes(y, 1,0), 
                      index=None, columns = x)
    df['lag'] = pd.Series(lags_to_test, index=df.index)
    
    
    g = sns.FacetGrid(df, col='lag', size=3, aspect=1.4,sharey=True, 
                      col_wrap=len(lags_to_test), ylim=y_lim)

    for i, ax in enumerate(g.axes.flatten()):
        l = df['lag'][i]
        
        ax.scatter(x, y[:,i].squeeze(), s = 8)
        ax.scatter(x, y_min[:,i].squeeze(), s=4, marker="_")
        ax.scatter(x, y_max[:,i].squeeze(), s=4, marker="_")
        ax.set_ylabel(metric) ; ax.set_xlabel('Time Aggregation')
        ax.grid(b=True, which='major')
        ax.set_title('lag {}'.format(l))
        if min(x) == 1:
            xmin = 0
        else:
            xmin = min(x)
        xticks = np.arange(xmin, max(x)+1E-9, 10) ; 
        if min(x) == 1:
            xticks[0] = 1
        ax.set_xticks(xticks)
        if metric == 'BSS':
            y_major = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            ax.set_yticks(y_major, minor=False)
            ax.set_yticklabels(y_major)
            ax.set_yticks(np.arange(-0.6,0.6+1E-9, 0.1), minor=True)
            ax.hlines(y=0, xmin=min(x), xmax=max(x), linewidth=1)
        elif metric == 'AUC':
            ax.set_yticks(np.arange(0.5,1+1E-9, 0.1), minor=True)
            ax.hlines(y=0.5, xmin=min(x), xmax=max(x), linewidth=1)
        
#    str_freq = str(x).replace(' ' ,'')  
    #%%
    return

def plot_score_lags(dict_tfreq, metric, lags_to_test):
    #%%

    tfreq = list(dict_tfreq.keys())
    
    
    if metric == 'BSS':
        y_lim = (-0.6, 0.6)
        y = np.array([dict_tfreq[f].brier_logit.loc['BSS'] for f in tfreq])
        y_min = np.array([dict_tfreq[f].brier_logit.loc['BSS_high'] for f in tfreq])
        y_max = np.array([dict_tfreq[f].brier_logit.loc['BSS_low'] for f in tfreq])
        y_cv  = np.array([dict_tfreq[f].BSS.values for f in tfreq])
    elif metric == 'AUC':
        y_lim = (0.4,1.0)
        y = np.array([dict_tfreq[f].AUC_spatcov.loc['AUC'] for f in tfreq])
        y_min = np.array([dict_tfreq[f].AUC_spatcov.loc['con_low'] for f in tfreq])
        y_max = np.array([dict_tfreq[f].AUC_spatcov.loc['con_high'] for f in tfreq])
        y_cv  = np.array([dict_tfreq[f].AUC.values for f in tfreq])

    df = pd.DataFrame(np.swapaxes(y, 1,0).T, 
                      index=None, columns = lags_to_test)
    df['tfreq'] = pd.Series(tfreq, index=df.index)
    
    g = sns.FacetGrid(df, col='tfreq', size=3, aspect=1.4,sharey=True, 
                      col_wrap=len(lags_to_test), ylim=y_lim)

    for i, ax in enumerate(g.axes.flatten()):
        col = df['tfreq'][i]
        x = [l*col for l in lags_to_test]

        color = 'red'
        style = 'solid'
        ax.fill_between(x, y_min[i,:], y_max[i,:], linestyle='solid', 
                                edgecolor='black', facecolor=color, alpha=0.3)
        ax.plot(x, y[i,:], color=color, linestyle=style, 
                        linewidth=2, alpha=1 ) 
        for f in range(y_cv.shape[1]):
            style = 'dashed'
            ax.plot(x, y_cv[i,f,:], color=color, linestyle=style, 
                        linewidth=1, alpha=0.35 ) 
        ax.set_ylabel(metric) ; ax.set_xlabel('Lead time [days]')
        ax.grid(b=True, which='major')
        ax.set_title('{}-day mean'.format(col))
        if min(x) == 1:
            xmin = 0
        else:
            xmin = min(x)
        xticks = np.arange(xmin, max(x)+1E-9, 20) ; 
        if min(x) == 1:
            xticks[0] = 1
        ax.set_xticks(xticks)
        if metric == 'BSS':
            y_major = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            ax.set_yticks(y_major, minor=False)
            ax.set_yticklabels(y_major)
            ax.set_yticks(np.arange(-0.6,0.6+1E-9, 0.1), minor=True)
            ax.hlines(y=0, xmin=min(x), xmax=max(x), linewidth=1)
        elif metric == 'AUC':
            ax.set_yticks(np.arange(0.5,1+1E-9, 0.1), minor=True)
            ax.hlines(y=0.5, xmin=min(x), xmax=max(x), linewidth=1)
        
#    str_freq = str(x).replace(' ' ,'')  
    #%%
    return

