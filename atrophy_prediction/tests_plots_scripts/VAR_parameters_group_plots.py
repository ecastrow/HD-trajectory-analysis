#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:21:43 2020

@author: Eduardo Castro

This script trains two VAR models, one for controls and another one for pre-HD
individuals. Once trained, their parameters are plot to compare them across
groups.
"""

import sys
import os
import pandas as pd
import numpy as np
from copy import copy
import multiprocessing as mp
from os.path import join as opj
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(opj(os.path.dirname(__file__), '..', '..')))
from utility_code import resampling_tools as rt
from atrophy_prediction.prediction_scripts import VAR_prediction as VAR_fnc


"""
FUNCTION DEFINITIONS
"""

def plot_VAR_group_distr(VAR_mcmc_control, VAR_mcmc_preHD, regions,
                         plot_store=False, store_dir=None, ci_flag=False,
                         plot_format='vector'):
    """
    Plot the distribution of the VAR parameters after MCMC sampling
    for both controls and pre-HD individuals.
    """

    # Arrange MCMC samples of VAR parameters of both groups in a single df
    dim = len(regions)
    groups = ['control', 'preHD']
    VAR_distr_df = []

    for grp in groups:
        dict_data = {}
        # Get number of samples for each group (they may be of different sizes)
        str_cmd = 'VAR_mcmc_{}.shape[-1]'.format(grp)
        nsamples = eval(str_cmd)
        dict_data['group'] = np.repeat([grp], nsamples)
        if grp == 'control':
            VAR_mcmc_grp = VAR_mcmc_control
        else:
            VAR_mcmc_grp = VAR_mcmc_preHD

        for var_idx in range(dim ** 2):
            idx0, idx1 = np.unravel_index(var_idx, (dim, dim))
            dict_data['VAR_coef_{}'.format(var_idx)] =\
                VAR_mcmc_grp[idx0, idx1, :]
        VAR_distr_df.append(pd.DataFrame(dict_data))

    VAR_distr_df = pd.concat(VAR_distr_df)

    # Generation of boxplots for both groups
    if plot_format == 'vector':
        file_ext = 'svg'
    else:
        file_ext = 'png'

    yranges = [(0.975, 1.0), (-0.035, 0.035), (-0.03, 0.03),
               (-0.03, 0.005), (0.975, 1.0), (-0.01, 0.04),
               (-0.01, 0.02), (-0.02, 0.05), (0.975, 1.0)]
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(13, 10),
                            sharex=True)
    for var_idx in range(dim ** 2):
        tup_idx = np.unravel_index(var_idx, (dim, dim))
        idx0, idx1 = tup_idx
        ax = axs[tup_idx]

        var_col_name = 'VAR_coef_{}'.format(var_idx)
        if ci_flag:
            sns.boxplot(x='group', y=var_col_name, data=VAR_distr_df,
                        sym='', whis=[2.5, 97.5], bootstrap=1000, notch=True,
                        ax=ax)
        else:
            sns.boxplot(x='group', y=var_col_name, data=VAR_distr_df,
                        sym='', ax=ax)
        ax.set_ylim(yranges[var_idx])

        if idx0 == idx1:
            ax.set_title('{} AR(1)'.format(regions[idx0]))
        else:
            ax.set_title('{1} to {0} AR(1)'.format(regions[idx0],
                                                   regions[idx1]))
            ax.axhline(y=0, color='dodgerblue', linestyle='--')

        if idx1 == 0:
            ax.set_ylabel('AR(1) estimates')
        else:
            ax.axes.yaxis.set_label_text('')
        if idx0 == dim-1:
            ax.set_xlabel('Analyzed groups')
        else:
            ax.axes.xaxis.set_label_text('')

        if plot_store:
            fig.savefig(opj(store_dir,
                            '.'.join(['VAR_distribution_control_preHD',
                                      file_ext])),
                        dpi=300)

    return


def train_VAR_adjusted(data, group_analysis, reg_flag, reg_coef_diag,
                       reg_coef_offdiag, reg_coef_bias, regions,
                       N_tpoints, bstrap_indexes, method='PML',
                       nsamples=int(1e4), thinning=10):

    """
    VAR parameter estimation function adjusted to compare controls and pre-HD
    individuals fairly, that is, using the same number of people for both
    groups. This is done by taking various bootstrap samples of the larger
    group (pre-HD) with a similar number of subjects as the control group (it
    can be proven that the percentage of original observations in a bootstrap
    sample is 63.2% on average).
    """
    indep_var = 'follup_visit'

    data_bstrap = data[data.subjid.isin(bstrap_indexes)].copy()
    thin_bstrap = copy(thinning)

    if group_analysis == 'preHD':
        thin_bstrap = 10 * thin_bstrap

    # Discard duplicate entries; interleave time series; get group AR params
    data_bstrap.drop_duplicates(inplace=True)
    N_instances = len(data_bstrap.subjid.unique())
    data_ileaved_df = VAR_fnc.interleave_data_instances(data_bstrap, N_instances,
                                                        N_tpoints, regions)
    data_ileaved_df.set_index(indep_var, inplace=True)

    try:
        np.random.seed(seed=4893)
        VAR_est, const_est = VAR_fnc.train_VAR(data_ileaved_df,
                                               reg_flag=reg_flag,
                                               reg_coef_bias=reg_coef_bias,
                                               reg_coef_diag=reg_coef_diag,
                                               reg_coef_offdiag=reg_coef_offdiag,
                                               N_instances=N_instances,
                                               regions=regions, method=method,
                                               nsamples=nsamples,
                                               thinning=thin_bstrap)

    except Exception as e:
        VAR_est = np.nan
        print('Error: {:s}'.format(e))

    return VAR_est


"""
INPUT VARIABLES
"""

analysis_dir = opj('/data2/eduardo/results/HD_n_Controls/ROI-feats/'
                   'vol_prediction/TRACK')
indep_var = 'follup_visit'
regions = ['Thalamus', 'Caudate', 'Putamen']
reg_flag = True
reg_coef_bias = 1e4
reg_coef_diag = 1e2
reg_coef_offdiag = 1e0
VAR_method = 'M-H'
MCMC_samples = int(5e3)
bstrap_nsamp = 100
thinning = 5

N_tpoints = 7
store_flag = True
plot_format = 'raster'      # 'vector'
ci_flag = True
parallel_flag = True
nproc = 20

"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':

    # Load subcortical vols (adjusted by ICV)
    vols_df = pd.read_csv(opj(analysis_dir, 'vols_longit_norm.csv'))

    # Retrieve volumes for both groups separately
    vols_control = vols_df[vols_df.group == 'control']
    vols_preHD = vols_df[vols_df.group == 'preHD']

    # Retrieve training sets for both groups
    train_vol_control = vols_control[vols_control['pred_use'] == 'train']
    train_vol_preHD = vols_preHD[vols_preHD['pred_use'] == 'train']

    # Estimate VAR parameters for controls
    N_instances = len(train_vol_control.subjid.unique())
    data_ileaved_df = VAR_fnc.interleave_data_instances(train_vol_control,
                                                        N_instances,
                                                        N_tpoints, regions)
    data_ileaved_df.set_index(indep_var, inplace=True)

    try:
        np.random.seed(seed=4893)
        VAR_control, const_est =\
            VAR_fnc.train_VAR(data_ileaved_df, reg_flag=reg_flag,
                              reg_coef_bias=reg_coef_bias,
                              reg_coef_diag=reg_coef_diag,
                              reg_coef_offdiag=reg_coef_offdiag,
                              N_instances=N_instances,
                              regions=regions, method=VAR_method,
                              nsamples=MCMC_samples, thinning=thinning)
    except Exception as e:
        VAR_control = np.nan
        const_est = np.nan
        print('Error: {:s}'.format(e))


    # Estimate VAR parameters for pre-HD
    bstrap_idx_list = []
    preHD_subjid = train_vol_preHD.subjid.unique()
    for biter in range(bstrap_nsamp):
        bstrap_indexes = rt.gen_resample_indexes(preHD_subjid,
                                                 replace=True,
                                                 random_state=biter)
        bstrap_idx_list.append(bstrap_indexes)
    
    group_analysis = 'preHD'
    train_VAR_wrap = partial(train_VAR_adjusted, data=train_vol_preHD,
                             group_analysis=group_analysis, reg_flag=reg_flag,
                             reg_coef_diag=reg_coef_diag,
                             reg_coef_offdiag=reg_coef_offdiag,
                             reg_coef_bias=reg_coef_bias,
                             regions=regions, N_tpoints=N_tpoints,
                             method=VAR_method, nsamples=MCMC_samples,
                             thinning=thinning)
    
    if parallel_flag:
        print('VAR params estimation using parallel processing'
              ' ({} processes)'.format(nproc))
        pool = mp.Pool(processes=nproc)
        params_list = pool.map(train_VAR_wrap,
                               [bs_idx for bs_idx in bstrap_idx_list])
        pool.close()
        pool.join()

        VAR_mcmc_preHD = np.concatenate(params_list, axis=2)
    else:
        print('VAR params estimation using serial processing')
        params_list = [train_VAR_wrap(bs_idx) for bs_idx in bstrap_idx_list]

        VAR_mcmc_preHD = np.concatenate(params_list, axis=2)
        
    # Plot distributions of VAR parameters for controls and patients
    plot_VAR_group_distr(VAR_control, VAR_mcmc_preHD, regions,
                         plot_store=store_flag, ci_flag=ci_flag,
                         plot_format=plot_format)
