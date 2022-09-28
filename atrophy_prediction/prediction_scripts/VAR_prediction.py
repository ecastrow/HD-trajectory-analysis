#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:59:53 2020

@author: Eduardo Castro

Script to train a VAR model using multiple instances of volumetric progression
of pre-HD individuals. It also generates prospective predictions for follow-up
visits of subjects in the test set and estimates its errors.
"""

import re
import pandas as pd
import numpy as np
import pyflux as pf
from os.path import join as opj
from sklearn import linear_model
from sklearn.dummy import DummyRegressor
from ast import literal_eval


"""
FUNCTION DEFINTIONS
"""


def retrieve_subjects_window_estimation(base_df, group_eval, CAG_base=39,
                                        nsubj_window=20, fixed_stride=5,
                                        window_idx=1):
    """
    Get a subset of subjects from the training set to train AR models with
    multiple time series. The idea is to get multiple groups of subjects that
    overlap with each other and estimate the AR parameters for each subgroup,
    a rationale similar to that of the convolution operation. The stride
    (like the displacement of the convolution filter) is defined by an initial
    value of CAG repeats. Since controls don't have recorded CAG repeats, the
    stride for that group is defined by a fixed displacement of the subjects.

    Parameters
    ----------
    base_df: pandas DataFrame
        Dataframe with basic baseline information. It includes the
        subject IDs, the number of CAG repeats, the group membership (control
        or preHD) and the validation group (train or test).
    group_eval: str
        Group to be analyzed (preHD or controls)
    CAG_base: int, optional
        Only subjects with CAG repeats equal to or higher than this baseline
        CAG are retrieved (pre-HD only)
    nsubj_window: int, optional
        Number of subjects retrieved per window estimation
    fixed_stride: int, optional
        For pre-HD, the stride is defined by CAG_base. For controls, it is
        defined by fixed_stride and window_idx (each window is defined by
        multiples of the fixed stride)
    window_idx: int, optional
        Index of the window that will be retrieved. Only valid for controls

    Returns
    -------
    subjid_list : list
        List with the IDs of the subjects to be used for the window estimation
    """

    base_df_cp = base_df.copy()

    if group_eval == 'preHD':
        base_df_cp.sort_values('CAG', inplace=True)
        CAG_valid_idx = base_df_cp['CAG'] >= CAG_base
        subjid_list = base_df_cp.loc[CAG_valid_idx].\
            head(nsubj_window).index.values
    if group_eval == 'control':
        max_idx = len(base_df_cp) - 1
        iloc_idx = np.arange(nsubj_window) + (window_idx-1)*fixed_stride
        iloc_idx = [idx for idx in iloc_idx if idx <= max_idx]
        subjid_list = base_df_cp.iloc[iloc_idx].index.values

    return subjid_list


def VAR_window_estimation(data, group_analysis, reg_flag,
                          reg_coef_diag, reg_coef_offdiag, reg_coef_bias,
                          regions, N_tpoints, n_windows=6, nsubj_window=20,
                          fixed_stride=5, method='PML', nsamples=int(1e4),
                          thinning=10):

    """
    Estimate VAR parameters for volumetric time series of different brain areas
    for various groups of subjects that overlap with each other (windows).
    These are defined by either an initial value of CAG repeats (pre-HD) or a
    fixed window-based stride (controls). Details of window-based subject
    retrieval are described in retrieve_subjects_window_estimation.
    
    Parameters
    ----------
    data: pandas DataFrame
        Dataframe with time series information of variables of interest,
        their associated CAG repeats, subject IDs and an index for the
        temporal order of the time points (follup_visit)
    group_analysis: str
        Group to be analyzed (control or pre-HD)
    reg_flag: bool
        Flag to enable (or disable) parameter regularization
    reg_coef_diag: float
        Regularization parameter of the diagonal coefficients of the VAR model
        (lag k autocorrelation terms). The higher its value, the closer they
        will be to the provided initial value (see train_VAR for more
        information)
    reg_coef_offdiag: float
        Regularization parameter of the off-diagonal coefficients of the VAR
        model (lag k cross-correlation terms). The higher its value, the closer
        to 0 they will be (see train_VAR for more information)
    reg_coef_bias: float
        Regularization parameter of the bias coefficient of the VAR model. The
        higher it is, the closer to 0 it will be
    regions: list (str)
        List with the brain regions whose values will be predicted
    N_tpoints: int
        Number of time points of the time series. The code assumes that they
        are the same across instances and that there are no missing values
    n_windows: int, optional
        Number of window-based estimates to be retrieve by this function
    nsubj_window: int, optional
        Number of subjects retrieved per window estimation
    fixed_stride: int, optional
        Defines the stride for window-based estimates. Only applicable for
        controls (for patients windows are defined by increments in CAG repeats)
    method: str, optional
        Bayesian inference options. Accepted values: 'PML' (Penalized Maximum
        Likelihood, equivalent to Maximum a Posteriori) and 'M-H'
        (Metropolis-Hastings)
    nsamples: int, optional
        If method='M-H', then nsamples random samples are retrieved from the
        VAR parameters joint posterior distribution
    thinning: int, optional
        Return every n-th sample from the estimated samples to mitigate their
        autocorrelation. Default=10
    
    Returns
    -------
    VAR_point_windows: pandas DataFrame
        Dataframe with VAR parameters for each variable of interest across
        windows. It also includes the mean CAG repeats value for the subjects
        in each subgroup and the number of subjects in them
    """
    indep_var = 'follup_visit'
    VAR_point_windows = pd.DataFrame(index=range(n_windows))
    base_df = data.groupby('subjid').first()

    if group_analysis == 'preHD':
        CAG_base = base_df.CAG.min()
    else:
        CAG_base = None

    # Iterate across CAG-based windows
    for window_idx in range(n_windows):
        print('Window index = {}'.format(window_idx))
        subjid_list =\
            retrieve_subjects_window_estimation(base_df, group_analysis,
                                                CAG_base=CAG_base,
                                                nsubj_window=nsubj_window,
                                                fixed_stride=fixed_stride,
                                                window_idx=window_idx+1)
        data_subset = data[data.subjid.isin(subjid_list)]

        # Interleave time series; compute group AR parameters
        N_instances = len(data_subset.subjid.unique())
        data_ileaved_df = interleave_data_instances(data_subset, N_instances,
                                                    N_tpoints, regions)
        data_ileaved_df.set_index(indep_var, inplace=True)

        try:
            np.random.seed(seed=4893)
            VAR_est, const_est = train_VAR(data_ileaved_df, reg_flag=reg_flag,
                                           reg_coef_bias=reg_coef_bias,
                                           reg_coef_diag=reg_coef_diag,
                                           reg_coef_offdiag=reg_coef_offdiag,
                                           N_instances=N_instances,
                                           regions=regions, method=method,
                                           nsamples=nsamples,
                                           thinning=thinning)

            VAR_median = np.median(VAR_est, 2)
            const_median = np.median(const_est, 0)

        except Exception as e:
            VAR_median = np.nan
            const_median = np.nan
            print('Error: {:s}'.format(e))

        # Estimate AR parameters; assign subgroup info
        VAR_point_windows.loc[window_idx, 'CAG_mean'] =\
            data_subset.groupby('subjid').first().CAG.mean()
        VAR_point_windows.loc[window_idx, 'N_instances'] = N_instances
        VAR_point_windows.loc[window_idx, 'VAR_est'] = str(VAR_median)
        VAR_point_windows.loc[window_idx, 'const_est'] = str(const_median)

        if group_analysis == 'preHD':
            CAG_base = CAG_base + 1

    return VAR_point_windows


def interleave_data_instances(train_vol_df, N_instances, N_tpoints, regions):
    """
    Arrange time series (region trajectories) from different subjects such
    that they are interleaved. For example, given volume trajectories of a
    given region for two subjects a and b (T time points each), the interleaved
    sequence would be a1 b1 a2 b2 ... aT bT.
    
    Parameters
    ----------
    train_vol_df: pandas DataFrame
        Dataframe with time series information of variables of interest,
        their associated CAG repeats, subject IDs and an index for the
        temporal order of the time points (follup_visit)
    N_instances: int
        Number of time series instances (in this case, subjects)
    N_tpoints: int
        Number of time points of the time series. The code assumes that they
        are the same across instances and that there are no missing values
    regions: list (str)
        List with the brain regions whose values will be predicted
    
    Returns
    -------
    data_ileaved_df: pandas DataFrame
        Dataframe with the interleaved representation of the variables of
        interest
    """
    train_vol_df = train_vol_df.sort_values(['subjid', 'follup_visit'])
    data_instances = train_vol_df[regions].values

    if N_instances == 1:
        data_ileaved = data_instances
    else:
        data_ileaved = []
        for fup_idx in range(N_tpoints):
            data_ileaved.append(data_instances[fup_idx:
                                               N_tpoints*(N_instances-1) +
                                               fup_idx+1:
                                               N_tpoints, :])
        data_ileaved = np.vstack(data_ileaved)

    data_ileaved_df = pd.DataFrame(data_ileaved, columns=regions)
    data_ileaved_df['follup_visit'] = data_ileaved_df.index

    return data_ileaved_df


def VAR_df_unpack(VAR_df, regions):
    """
    Extract VAR estimates from dataframe and transform from text to numpy
    array. Keep the rest of the fields in dataframe format.
    """
    groups_info = VAR_df.copy()

    # Retrieve optimal VAR parameters across sequences
    VAR_est = groups_info['VAR_est'].values.tolist()
    del groups_info['VAR_est']
    del groups_info['const_est']

    # Transform VAR param estimates from plain strings back to numpy arrays
    for i in range(len(VAR_est)):
        VAR_est_str = [re.sub(' +',
                              ' ',
                              val.replace('[ ', '[')).strip().replace(' ',
                                                                      ', ')
                       for val in VAR_est[i].split('\n')]
        try:
            VAR_est[i] = np.array(literal_eval(', '.join(VAR_est_str)))
        except Exception as e:
            vals = np.zeros_like(np.eye(len(regions)))
            vals[:] = np.nan
            VAR_est[i] = vals
            print('Error for subset #{0}: {1}'.format(i, e))

    # Concatenate in 3rd dimension
    VAR_est = np.stack(VAR_est, axis=2)

    return VAR_est, groups_info


def VAR_test_estimate(VAR_point_windows, test_vol_df, regions):
    """
    Estimate VAR parameters of the provided regions for each test subject. This
    is done after making a linear fit between the mean number of CAG repeats of
    the subjects used for each window and their estimated VAR parameters during
    training.
    
    Parameters
    ----------
    VAR_point_windows: pandas DataFrame
        Dataframe with VAR parameters for each variable of interest across
        windows. It also includes the mean value of the CAG repeats of
        the subjects in each subgroup and the number of subjects in them
    test_vol_df: pandas DataFrame
        Dataframe with time series information of variables of interest, along
        with their associated number of CAG repeats and their subject IDs
    regions: list (str)
        List with the brain regions whose values will be predicted
    
    Returns
    -------
    VAR_est_all: dict
        Dictionary indexed by the subject IDs in test, its elements being
        their VAR parameters
    """
    VAR_est_all = {}

    # Get rid of any NaN values in window-based VAR parameters estimation
    VAR_point_windows = VAR_point_windows[VAR_point_windows['VAR_est'] !=
                                          'nan']

    # Unpack VAR point estimates and CAG info from subgroups
    VAR_mat, groups_info = VAR_df_unpack(VAR_point_windows, regions)

    # Fit linear association between CAG repeats and VAR coefficients
    linfit_models = {}
    dim = len(regions)
    for var_idx in range(dim ** 2):
        idx0, idx1 = np.unravel_index(var_idx, (dim, dim))
        mdl = linear_model.Ridge()
        mdl.fit(groups_info['CAG_mean'].values[:, None],
                VAR_mat[idx0, idx1, :])
        linfit_models[var_idx] = mdl

    # For off-diagonal elements (R2<0.5), replace with median vals (bootstrap)
    preHD_bstrap_fn = ('/data2/eduardo/results/HD_n_Controls/ROI-feats/'
                       'vol_prediction/TRACK/''VAR_mcmc_estimates_preHD_bstrap.npy')
    VAR_bstrap = np.load(preHD_bstrap_fn)
    VAR_bstrap = np.median(VAR_bstrap, 2)
    off_diag_idx = np.reshape(range(dim**2), (dim, dim))[np.eye(3) != 1]
    for var_idx in off_diag_idx:
        idx0, idx1 = np.unravel_index(var_idx, (dim, dim))
        const = VAR_bstrap[idx0, idx1]
        mdl = DummyRegressor(strategy='constant', constant=const)
        mdl.fit(groups_info['CAG_mean'].values[:, None],
                VAR_mat[idx0, idx1, :])
        linfit_models[var_idx] = mdl

    # Estimate VAR parameters for test subjects
    CAG_df = test_vol_df[['subjid', 'CAG']]
    CAG_df = CAG_df.groupby('subjid').first()

    for subjid in CAG_df.index.values:
        VAR_est = np.zeros((dim, dim))
        for var_idx in range(dim ** 2):
            idx0, idx1 = np.unravel_index(var_idx, (dim, dim))
            VAR_est[idx0, idx1] =\
                np.squeeze(linfit_models[var_idx].
                           predict(CAG_df.loc[subjid].values[:, None]))
        VAR_est_all[subjid] = VAR_est

    return VAR_est_all


def predict_test_vols_VAR(test_real_vol_df, VAR_subj_test, regions):
    """
    Estimate future volumes based on estimated VAR params for test subjects and
    their baseline volumes.
    Eq: vol[fup_vis] = varmat^fup_vis * vol[0]
    
    Parameters
    ----------
    test_real_vol_df: pandas DataFrame
        Dataframe with actual time series information of variables of interest
        for subjects in the test set. It includes an index for the temporal
        order of the time points (follup_visit) and its associated subject IDs
    VAR_subj_test: dict
        Dictionary indexed by the subject IDs in test, its elements being
        their VAR parameters
    regions: list (str)
        List with the brain regions whose values will be predicted
    
    Returns
    -------
    test_est_vol_df: pandas DataFrame
        DataFrame with the predicted future values of the variables of interest
        (those occuring after the baseline visit) for subjects in the test set
    """
    test_real_vol_df_cp = test_real_vol_df.copy()

    # Retrieve subjid and follow-up visit info of test subjects
    test_real_vol_df_cp.sort_values(['subjid', 'follup_visit'], inplace=True)
    test_est_vol_df = pd.DataFrame(index=test_real_vol_df_cp.index,
                                   columns=regions)
    test_est_vol_df[['subjid', 'follup_visit']] =\
        test_real_vol_df_cp[['subjid', 'follup_visit']]

    # Retrieve volume values from baseline visits
    idx_base = test_real_vol_df_cp.groupby('subjid')['follup_visit'].\
        transform('idxmin')
    test_est_vol_df[regions] = test_real_vol_df_cp.loc[idx_base,
                                                       regions].values
    test_est_vol_df.set_index('subjid', inplace=True)
    test_real_vol_df_cp.set_index('subjid', inplace=True)

    # Calculate volumetric estimates for all regions (follow-up visits)
    init_vols_all = test_est_vol_df.loc[test_est_vol_df.follup_visit == 0,
                                        regions]
    pred_vols_list = []
    for subjid in test_est_vol_df.index.unique():
        all_vols_subj = test_est_vol_df.loc[subjid].copy()
        init_vols_subj = init_vols_all.loc[subjid].values[:, None]
        varmat = VAR_subj_test[subjid]

        for fup_vis in test_est_vol_df.loc[subjid].follup_visit.unique():
            if fup_vis > 0:
                pred_vols = np.dot(np.linalg.matrix_power(varmat, fup_vis),
                                   init_vols_subj)
                all_vols_subj.loc[all_vols_subj.follup_visit == fup_vis,
                                  regions] = pred_vols.squeeze()
        pred_vols_list.append(all_vols_subj)

    test_est_vol_df = pd.concat(pred_vols_list)

    # Get rid of baseline valumes after follow-up visits' estimation
    test_est_vol_df = test_est_vol_df[test_est_vol_df.follup_visit > 0]

    return test_est_vol_df


def prediction_error_estimation(test_real_trajec_df, test_est_trajec_df,
                                predict_vars_names):
    """
    Calculate absolute error of VAR model predictions for each variable of
    interest across follow-up visits
    Parameters
    ----------
    test_real_trajec_df: pandas DataFrame
        Dataframe with actual time series information of variables of interest
        for subjects in the test set. It includes an index for the temporal
        order of the time points (follup_visit) and its associated subject IDs
    test_est_trajec_df: pandas DataFrame
        DataFrame with the predicted future values of the variables of interest
        (those occuring after the baseline visit) for subjects in the test set
    predict_vars_names: list (str)
        List with the names of the variables of interest (variables to be
        predicted) as they appear in data
    Returns
    -------
    test_diff_trajec_df: pandas DataFrame
        Dataframe with AR model prediction errors (in absolute value) for
        follow-up visits
    """
    test_real_cp = test_real_trajec_df.copy()
    test_est_cp = test_est_trajec_df.copy()
    test_est_cp.reset_index(inplace=True)
    test_est_cp.set_index(['subjid', 'follup_visit'], inplace=True)
    test_real_cp.set_index(['subjid', 'follup_visit'], inplace=True)

    # Estimate difference from actual values; take absolute values
    test_diff_trajec_df = test_est_cp
    test_diff_trajec_df = test_diff_trajec_df -\
        test_real_cp[predict_vars_names]
    test_diff_trajec_df = test_diff_trajec_df.abs()

    # Get rid of baseline visit
    test_diff_trajec_df =\
        test_diff_trajec_df[~test_diff_trajec_df.index.
                            get_level_values('follup_visit').isin([0])]

    return test_diff_trajec_df


def train_VAR(data, reg_flag, reg_coef_bias, reg_coef_diag, reg_coef_offdiag,
              N_instances, regions, method='PML', nsamples=int(1e4),
              thinning=10, all_or_median='all'):
    """
    Train VAR model (optionally with regularization) and retrieve learned
    parameters. Supports the option of being trained with multiple data
    instances from the same underlying VAR model (lag 1 model only)
    
    
    Parameters
    ----------
    data: pandas DataFrame
        Dataframe with time series information of variables of interest,
        their associated risk variable (RV), subject IDs and an index for the
        temporal order of the time points (follup_visit)
    reg_flag: bool
        Flag to enable (or disable) parameter regularization
    reg_coef_diag: float
        Regularization parameter of the diagonal coefficients of the VAR model
        (lag k autocorrelation terms). The higher its value, the closer they
        will be to the provided initial value (see train_VAR for more
        information)
    reg_coef_offdiag: float
        Regularization parameter of the off-diagonal coefficients of the VAR
        model (lag k cross-correlation terms). The higher its value, the closer
        to 0 they will be (see train_VAR for more information)
    reg_coef_bias: float
        Regularization parameter of the bias coefficient of the VAR model. The
        higher it is, the closer to 0 it will be
    N_instances: int
        Number of time series instances (in this case, subjects)
    regions: list (str)
        List with the brain regions whose values will be predicted
    method: str, optional
        Bayesian inference options. Accepted values: 'PML' (Penalized Maximum
        Likelihood, equivalent to Maximum a Posteriori) and 'M-H'
        (Metropolis-Hastings)
    nsamples: int, optional
        If method='M-H', then nsamples random samples are retrieved from the
        VAR parameters joint posterior distribution
    thinning: int, optional
        Return every n-th sample from the estimated samples to mitigate their
        autocorrelation. Default=10
    all_or_median: string, optional
        Retrieve all of the samples retrieved by Bayesian estimation or their
        median values
    
    Returns
    -------
    VAR_estimates: numpy array
        2D array with the learned VAR coefficients
    const_estimates: numpy array
        2D array with the learned bias coefficients
    """
    # Additional model parameters
    max_lag = N_instances
    nz_lags = [N_instances]
    nvars = data.shape[1]

    # Retrieve indexes of constant and diagonal (self AR) parameters
    estimate_idx = range(nvars*(nvars + 1))
    estimate_idx_mat = np.reshape(estimate_idx, (nvars, nvars+1))
    var_const_idx = estimate_idx_mat[:, 0]
    var_diag_idx = np.diag(estimate_idx_mat[:, 1:])
    var_offdiag_idx = np.setdiff1d(estimate_idx,
                                   np.r_[var_const_idx, var_diag_idx])
    var_only_idx = np.ravel(estimate_idx_mat[:, 1:])

    # Set up model
    model = pf.VAR(data=data, lags=max_lag, ar_idx_list=nz_lags)

    if reg_flag:
        # Adjust regularization on constant terms
        sigma_reg_const = np.sqrt(reg_coef_bias) ** -1
        for i in var_const_idx:
            model.adjust_prior(i, pf.Normal(0, sigma_reg_const))

        # Adjust regularization on off-diagonal (cross-correlation) terms
        sigma_reg_offdiag = np.sqrt(reg_coef_offdiag) ** -1
        for i in var_offdiag_idx:
            model.adjust_prior(i, pf.Normal(0, sigma_reg_offdiag))

        # Adjust regularization parameters on diagonal (autocorrelation) terms
        sigma_reg_ar = np.sqrt(reg_coef_diag) ** -1
        sigma_univ_ar = np.sqrt(4) ** -1
        sigma_univ_const = np.sqrt(1e6) ** -1
        for dim_idx, var_idx in enumerate(var_diag_idx):
            # Initial estimate of AR params by fitting a univariate model
            mdl_ar = pf.ARIMA(data=data, ar=max_lag, ma=0,
                              target=regions[dim_idx],
                              family=pf.Normal(),
                              ar_idx_list=nz_lags)
            mdl_ar.adjust_prior(0, pf.Normal(0, sigma_univ_const))
            mdl_ar.adjust_prior(1, pf.Normal(0, sigma_univ_ar))
            res_ar = mdl_ar.fit(method='PML')

            # Make sure that no coefficient is greater than 1
            ar_param = res_ar.z_values[1]
            if ar_param > 1:
                ar_param = 1.

            # Use the univariate estimate as the mean of the prior distribution
            model.adjust_prior(var_idx, pf.Normal(ar_param, sigma_reg_ar))

    # Fit multivariate model
    if method == 'PML':
        results = model.fit(method=method)

        # Retrieve params from the model (VAR per se and constant terms)
        all_estimates = results.z_values
        VAR_estimates = np.reshape(all_estimates[var_only_idx],
                                   (nvars, nvars))
        VAR_estimates[VAR_estimates > 1] = 1.
        const_estimates = all_estimates[var_const_idx]

    elif method == 'M-H':
        results = model.fit(method='M-H', nsims=nsamples)

        # Retrieve params from the model (VAR per se and constant terms)
        if all_or_median == 'all':
            # Retrieve VAR params per se
            all_estimates = results.samples[:, ::thinning]
            thin_samples = all_estimates.shape[1]
            VAR_estimates = [np.reshape(all_estimates[:,
                                                      samp_idx][var_only_idx],
                                        (nvars, nvars))
                             for samp_idx in range(thin_samples)]
            VAR_estimates = np.stack(VAR_estimates, axis=2)

            # Retrieve constant terms too
            const_estimates = [all_estimates[:, samp_idx][var_const_idx]
                               for samp_idx in range(thin_samples)]
            const_estimates = np.stack(const_estimates, axis=1).T
        else:
            all_estimates = results.median_est
            VAR_estimates = np.reshape(all_estimates[var_only_idx],
                                       (nvars, nvars))
            VAR_estimates[VAR_estimates > 1] = 1.
            const_estimates = all_estimates[var_const_idx]
    else:
        raise ValueError('Unknown method {:s}'.format(method))

    return VAR_estimates, const_estimates


"""
INPUT VARIABLES
"""

analysis_dir = opj('/data2/eduardo/results/HD_n_Controls/ROI-feats/'
                   'vol_prediction/TRACK')
regions = ['Thalamus', 'Caudate', 'Putamen']

reg_flag = True
reg_coef_bias = 1e4
reg_coef_diag = 1e2
reg_coef_offdiag = 1e0
VAR_method = 'M-H'
MCMC_samples = int(5e3)
thinning = 5

group_analysis = 'preHD'
n_windows = 6
nsubj_window = 20
fixed_stride = 5
N_tpoints = 7
store_flag = True

"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':

    # Load subcortical vols (adjusted by ICV); retrieve train and test sets
    vols_df = pd.read_csv(opj(analysis_dir, 'vols_longit_norm.csv'))
    vols_df = vols_df[vols_df.group == 'preHD']
    train_vol_df = vols_df[vols_df['pred_use'] == 'train']
    test_real_vol_df = vols_df[vols_df['pred_use'] == 'test']

    # Estimate VAR params (training set) across subsets of subjects (windows)
    VAR_point_windows =\
        VAR_window_estimation(data=train_vol_df,
                              group_analysis=group_analysis,
                              reg_flag=reg_flag,
                              reg_coef_diag=reg_coef_diag,
                              reg_coef_offdiag=reg_coef_offdiag,
                              reg_coef_bias=reg_coef_bias,
                              regions=regions,
                              N_tpoints=N_tpoints,
                              method=VAR_method,
                              nsamples=MCMC_samples,
                              thinning=thinning)
    
    # Estimate VAR parameters for subjects in test set
    VAR_est_all = VAR_test_estimate(VAR_point_windows=VAR_point_windows,
                                    test_vol_df=test_real_vol_df,
                                    regions=regions)

    # Predict variables of interest at follow-up visits for subjects in test
    test_est_vol_df = predict_test_vols_VAR(test_real_vol_df=test_real_vol_df,
                                            VAR_subj_test=VAR_est_all,
                                            regions=regions)

    # Calculate the model's estimation error in test set (in absolute value)
    test_diff_vol_df = prediction_error_estimation(test_real_vol_df,
                                                   test_est_vol_df,
                                                   regions)
    
    if store_flag:
        test_diff_vol_df.to_csv(opj(analysis_dir, 'VAR_absolute_errors.csv'))

