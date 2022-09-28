#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:07:31 2020

@author: Eduardo Castro

Script to compare the performance of two models across follow-up visits
"""

from os.path import join as opj
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(opj(os.path.dirname(__file__), '..', '..')))
from utility_code import misc_utility_fncs as muf


"""
FUNCTION DEFINITIONS
"""


def compare_err_models(err_model1, err_model2, regions, stat_method='ttest',
                       alt_hyp='two-sided'):
    """
    Do a paired two-sample test between errors (in absolute value) of
    volumetric estimates at follow-up visits for two methods. The output is
    p-vals*sign(test) (to indicate sign of effects)
    """

    stat_function_dict = {'ttest': ttest_rel, 'wilcoxon': wilcoxon}
    stat_function = stat_function_dict[stat_method]
    nvis = np.max([err_model1['follup_visit'].max(),
                  err_model2['follup_visit'].max()])

    vis_rng = range(1, nvis+1)
    stat_vals_df = pd.DataFrame(columns=regions, index=vis_rng)
    fx_size_df = pd.DataFrame(columns=regions, index=vis_rng)
    pvals_df = pd.DataFrame(columns=regions, index=vis_rng)
    pcorr_df = pd.DataFrame(columns=regions, index=vis_rng)

    for region in regions:
        for vis in vis_rng:
            mdl1_vals = err_model1.loc[err_model1['follup_visit'] == vis,
                                       region].values
            mdl2_vals = err_model2.loc[err_model2['follup_visit'] == vis,
                                       region].values

            # Statistical test estimation
            if stat_method == 'wilcoxon':
                stat_vals_df.loc[vis, region], pvals_df.loc[vis, region] =\
                    stat_function(mdl1_vals, mdl2_vals,
                                  alternative=alt_hyp)
            else:
                stat_vals_df.loc[vis, region], pvals_df.loc[vis, region] =\
                    stat_function(mdl1_vals, mdl2_vals)

            # Calculation of Cohen's d (effect size)
            fx_mean = np.mean(mdl1_vals) - np.mean(mdl2_vals)
            n1, n2 = len(mdl1_vals), len(mdl2_vals)
            fx_std = ((n1 - 1) * np.var(mdl1_vals) +
                      (n2 - 1) * np.var(mdl2_vals)) / (n1 + n2 - 2)
            fx_size_df.loc[vis, region] = fx_mean / np.sqrt(fx_std)

        # FDR correction for each region
        pcorr_df[region] =\
            muf.multiple_compar_corr(pvals_df[region].values,
                                    0.05, method='FDR')[0]


    if stat_method == 'wilcoxon':
        sign_pvals_df = pvals_df
    else:
        sign_pvals_df = stat_vals_df.apply(lambda x: np.sign(x)) * pvals_df

    return sign_pvals_df, pcorr_df, fx_size_df


"""
INPUT VARIABLES
"""

analysis_dir = opj('/data2/eduardo/results/HD_n_Controls/ROI-feats/'
                   'vol_prediction/TRACK')
store_dir = opj(analysis_dir, 'final_results')
regions = ['Thalamus', 'Caudate', 'Putamen']
models = ['VAR', 'AR']      # ['AR', 'slope']
stat_method = 'wilcoxon'
alt_hyp = 'two-sided'     # 'greater'
group_analysis = 'preHD'    # 'earlyHD'
store_flag = True


"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':

    if group_analysis == 'preHD':
        base_fname = '{}_absolute_errors_preHD.csv'
    else:
        base_fname = '{}_absolute_errors_earlyHD.csv'

    err_all_models = [pd.read_csv(opj(store_dir,
                                      base_fname.format(mdl)))
                      for mdl in models]

    sign_pvals, pvals_corr, fx_sizes =\
        compare_err_models(*err_all_models, regions=regions,
                           stat_method=stat_method, alt_hyp=alt_hyp)

    if store_flag:
        sign_pvals.to_csv(opj(store_dir,
                              '{0}_vs_{1}_pvals_uncorr_{2}.csv'.format(*models
                                                                       + [group_analysis])))
        pvals_corr.to_csv(opj(store_dir,
                              '{0}_vs_{1}_pvals_FDR_{2}.csv'.format(*models
                                                                    + [group_analysis])))
        fx_sizes.to_csv(opj(store_dir,
                            '{0}_vs_{1}_effect_sizes_{2}.csv'.format(*models
                                                                     + [group_analysis])))
