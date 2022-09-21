#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:20:12 2019

Script that estimates the AUC of the trained models (k-fold cross validation)
and compares the statistics observed in the data with the null hypothesis
(statistical test)

@author: Eduardo Castro
"""

import numpy as np
import pandas as pd
from os.path import join as opj
from sklearn.metrics import roc_auc_score
import os
import sys
sys.path.append(os.path.abspath(opj(os.path.dirname(__file__), '..', '..')))
from utility_code import resampling_tools as rt

"""
INPUT VARIABLES
"""
inter_study = False
univ_area = 'Caudate'       # 'Striatum', 'Caudate'
study_id = 'TRACK'          # 'TRACK', 'PREDICT', 'IMAGEHD'
analysis_type = 'multiv_univ_test'    # 'multiv_univ_test', 'multiv_only_test'
nperm = int(1e4)
classif_results_base_fn = 'Probabilities4Logistic_Regression.csv'

"""
MAIN CODE
"""

# Preallocate dataframe with results
pvals_auc_df = pd.DataFrame(columns=['TRACK', 'PREDICT', 'IMAGEHD', 'ALL'],
                            index=['10-fold', 'inter-site', 'inter-study'])

# Do multivariate only or multivariate vs univariate test
if analysis_type == 'multiv_only_test':
    results_df = pd.DataFrame(columns=['Prediction', 'Label'])

    # Load predictions and actual labels
    if inter_study:
        classif_dir = ('/data2/eduardo/results/HD_n_Controls/'
                       'ROI-feats/slopes_longit/ALL_STUDIES/classification/'
                       'All_Regions')
    else:
        classif_dir = opj('/data2/eduardo/results/HD_n_Controls',
                          'ROI-feats/slopes_longit/', study_id,
                          'classification/All_Regions')

    classif_results_full_fn = opj(classif_dir, classif_results_base_fn)
    temp_df = pd.read_csv(classif_results_full_fn)
    results_df['Prediction'] = temp_df.iloc[:, -2]
    results_df['Label'] = temp_df['Label']

    # Estimate prediction under the null (random labels)
    idx_all = results_df.index.to_list()
    auc_perm_distr = []
    for perm_iter in range(nperm):
        if np.mod(perm_iter, 1000) == 0:
            print('sample #{0} of {1}'.format(perm_iter, nperm))

        perm_idx_all = rt.gen_resample_indexes(idx_all, replace=False,
                                               random_state=perm_iter)
        auc_perm_val = roc_auc_score(results_df.loc[perm_idx_all, 'Label'].values,
                                     results_df.loc[:, 'Prediction'].values)
        auc_perm_distr.append(auc_perm_val)

    auc_perm_distr = np.array(auc_perm_distr)

    # Estimate p-value associated to observed AUC
    auc_obs_val = roc_auc_score(results_df.loc[:, 'Label'].values,
                                results_df.loc[:, 'Prediction'].values)
    pval_auc = rt.gen_conf_interval(auc_perm_distr, auc_obs_val)[0]

    if inter_study:
        pvals_auc_df.loc['inter-study', 'ALL'] = pval_auc
    else:
        if study_id == 'IMAGEHD':
            pvals_auc_df.loc['10-fold', study_id] = pval_auc
        else:
            pvals_auc_df.loc['inter-site', study_id] = pval_auc
else:
    results_multiv_df = pd.DataFrame(columns=['Prediction', 'Label'])
    results_univ_df = pd.DataFrame(columns=['Prediction', 'Label'])

    # Load predictions and actual labels
    classif_multiv_dir = analysis_dir = opj('/data2/eduardo/results/'
                                            'HD_n_Controls',
                                            'ROI-feats/slopes_longit/',
                                            study_id,
                                            'classification/All_Regions')

    classif_univ_dir = analysis_dir = opj('/data2/eduardo/results/'
                                          'HD_n_Controls',
                                          'ROI-feats/slopes_longit/',
                                          study_id,
                                          opj('classification', univ_area))

    classif_multiv_full_fn = opj(classif_multiv_dir, classif_results_base_fn)
    temp_df = pd.read_csv(classif_multiv_full_fn)
    results_multiv_df['Prediction'] = temp_df.iloc[:, -2]
    results_multiv_df['Label'] = temp_df['Label']

    classif_univ_full_fn = opj(classif_univ_dir, classif_results_base_fn)
    temp_df = pd.read_csv(classif_univ_full_fn)
    results_univ_df['Prediction'] = temp_df.iloc[:, -2]
    results_univ_df['Label'] = temp_df['Label']

    # Estimate multivariate and univariate prediction under the null
    idx_all = results_multiv_df.index.to_list()
    auc_perm_distr = []
    for perm_iter in range(nperm):
        if np.mod(perm_iter, 1000) == 0:
            print('sample #{0} of {1}'.format(perm_iter, nperm))

        perm_idx_all = rt.gen_resample_indexes(idx_all, replace=False,
                                               random_state=perm_iter)
        auc_perm_multiv = roc_auc_score(results_multiv_df.loc[perm_idx_all,
                                                              'Label'].values,
                                        results_multiv_df.loc[:, 'Prediction'].values)
        auc_perm_univ = roc_auc_score(results_univ_df.loc[perm_idx_all,
                                                          'Label'].values,
                                      results_univ_df.loc[:, 'Prediction'].values)

        auc_perm_distr.append(auc_perm_multiv - auc_perm_univ)

    auc_perm_distr = np.array(auc_perm_distr)

    # Estimate p-value associated to observed AUC
    auc_obs_multiv = roc_auc_score(results_multiv_df.loc[:, 'Label'].values,
                                   results_multiv_df.loc[:, 'Prediction'].values)
    auc_obs_univ = roc_auc_score(results_univ_df.loc[:, 'Label'].values,
                                 results_univ_df.loc[:, 'Prediction'].values)
    auc_obs_diff = auc_obs_multiv - auc_obs_univ
    pval_auc = rt.gen_conf_interval(auc_perm_distr, auc_obs_diff)[0]

    if study_id == 'IMAGEHD':
        pvals_auc_df.loc['10-fold', study_id] = pval_auc
    else:
        pvals_auc_df.loc['inter-site', study_id] = pval_auc
