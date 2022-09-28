#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:33:59 2020

@author: Eduardo Castro

Script to visually analyze the performance of the models (slope-, AR-, and
VAR-based)
"""

from os.path import join as opj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it


"""
INPUT VARIABLES
"""

analysis_dir = opj('/data2/eduardo/results/HD_n_Controls/ROI-feats/'
                   'vol_prediction/TRACK')
store_dir = opj(analysis_dir, 'final_results')
regions = ['Thalamus', 'Caudate', 'Putamen']
models = ['slope', 'AR', 'VAR']
stages = ['preHD', 'earlyHD']    # use 'observed' for preHD
store_plot = True
plot_format = 'vector'
pal_bplot = ['#B2B2B2', '#1E90FF', '#FF7F00']   # gray, deep blue, orange


"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':

    # Put together results from all models in a single dataframe
    df_list = []
    all_cols = ['subjid', 'follup_visit', 'error', 'region', 'model', 'stage']
    base_cols = ['subjid', 'follup_visit']
    for tup in it.product(models, stages):
        err_fn = '{0}_absolute_errors_{1}.csv'.format(*tup)
        err_df = pd.read_csv(opj(store_dir, err_fn))
#        cond = 'preHD' if tup[1] == 'observed' else 'earlyHD'
        for reg in regions:
            plot_df = pd.DataFrame(columns=all_cols)
            plot_df[base_cols] = err_df[base_cols]
            plot_df['error'] = err_df[reg]
            plot_df['region'] = reg
            plot_df['model'] = tup[0]
            plot_df['stage'] = tup[1]
            df_list.append(plot_df)

    err_all = pd.concat(df_list)

    # Generate plots for a given condition and a specific region
    if plot_format == 'vector':
        file_ext = 'svg'
    else:
        file_ext = 'png'

    for cond in ['preHD', 'earlyHD']:
        base_fname = 'vol_prediction_results_{0}.{1}'.format(cond, file_ext)
        cond_df = err_all[err_all.stage == cond]
        fh = sns.catplot(x='follup_visit', y='error', hue='model',
                         data=cond_df, col='region', kind='bar',
                         palette=pal_bplot, estimator=np.mean)
        if store_plot:
            fh.savefig(opj(store_dir, base_fname), dpi=300)
