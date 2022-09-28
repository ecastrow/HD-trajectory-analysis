#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:31:45 2020

@author: Eduardo Castro

Script to train an AR model using multiple instances of volumetric progression
in pre-HD individuals. It also generates prospective predictions for follow-up
visits of subjects in the test set and estimates its errors.
"""

import pandas as pd
import numpy as np
import pyflux as pf
from os.path import join as opj
from sklearn import linear_model


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


def AR_window_estimation(data, group_analysis, reg_coef_bias, regions,
                         N_tpoints, n_windows=6, nsubj_window=20,
                         fixed_stride=5):
    """
    Estimate AR parameters of different brain areas for various groups of
    subjects that overlap with each other (windows). These are defined by
    either an initial value of CAG repeats (pre-HD) or a fixed window-based
    stride (controls). Details of window-based subject retrieval are described
    in retrieve_subjects_window_estimation.
    
    Parameters
    ----------
    data: pandas DataFrame
        Dataframe with time series information of variables of interest,
        their associated CAG repeats, subject IDs and an index for the
        temporal order of the time points (follup_visit)
    group_analysis: str
        Group to be analyzed (control or pre-HD)
    reg_coef_bias: float
        Regularization parameter of the bias coefficient of the AR model. The
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
    
    Returns
    -------
    ar_df_groups: pandas DataFrame
        Dataframe with AR parameters for each variable of interest across
        windows. It also includes the mean CAG repeats value for the subjects
        in each subgroup and the number of subjects in them
    """
    indep_var = 'follup_visit'
    ar_df_groups = pd.DataFrame(columns=regions)
    base_df = data.groupby('subjid').first()
    
    if group_analysis == 'preHD':
        CAG_base = base_df.CAG.min()
    else:
        CAG_base = None

    # Iterate across CAG-based windows
    for window_idx in range(n_windows):
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
        ar_df = train_AR_multiregion(data_ileaved_df, regions, indep_var,
                                     reg_coef_bias, N_instances)

        # Estimate AR parameters; assign subgroup info
        ar_df_groups.loc[window_idx, :] = ar_df[ar_df_groups.columns].values
        ar_df_groups.loc[window_idx, 'CAG_mean'] =\
            data_subset.groupby('subjid').first().CAG.mean()
        ar_df_groups.loc[window_idx, 'N_instances'] = N_instances

        if group_analysis == 'preHD':
            CAG_base = CAG_base + 1

    ar_df_groups[regions] = ar_df_groups[regions].astype(float)

    return ar_df_groups


def AR_test_estimate(ar_df_groups, test_vol_df, regions):
    """
    Estimate AR parameters of the provided regions for each test subject. This
    is done after making a linear fit between the mean number of CAG repeats of
    the subjects used for each window and their estimated AR parameters during
    training.
    
    Parameters
    ----------
    ar_df_groups: pandas DataFrame
        Dataframe with AR parameters for each variable of interest across
        windows. It also includes the mean value of the RV for the subjects in
        each subgroup and the number of subjects in them
    test_vol_df: pandas DataFrame
        Dataframe with time series information of variables of interest, along
        with their associated CAG repeats value and their subject IDs
    regions: list (str)
        List with the brain regions whose values will be predicted

    Returns
    -------
    ar_df_subj_test: pandas DataFrame
        Dataframe with AR parameters for each variable of interest across
        subjects in the test set
    """
    # Fit linear association between CAG repeats and per-region AR coefficients
    linfit_models = {}
    for reg in regions:
        mdl = linear_model.Ridge()
        mdl.fit(ar_df_groups['CAG_mean'].values[:, None],
                ar_df_groups[reg].values)
        linfit_models[reg] = mdl

    # Estimate AR parameters for test subjects
    CAG_df = test_vol_df[['subjid', 'CAG']]
    CAG_df = CAG_df.groupby('subjid').first()
    ar_df_subj_test = pd.DataFrame(index=CAG_df.index, columns=regions)
    for reg in regions:
        ar_df_subj_test[reg] = \
            np.squeeze(linfit_models[reg].predict(CAG_df.values))

    return ar_df_subj_test


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


def train_AR(data, reg_coef_bias, N_instances, region):
    """
    Train univariate AR model and retrieve learned parameters. Supports the
    option of being trained with multiple data instances from the same
    underlying AR model (lag 1 model only).
    
    Parameters
    ----------
    data: pandas DataFrame
        Dataframe with the interleaved representation of the variables of
        interest
    reg_coef_bias: float
        Regularization parameter of the bias coefficient of the AR model. The
        higher it is, the closer to 0 it will be
    N_instances: int
        Number of time series instances (in this case, subjects)
    region: str
    	The variable of interest that will be learned by the univariate
        AR model
    
    Returns
    -------
    params: numpy array
        Array with the learned bias and AR coefficients
    """
    # Additional model parameters
    max_lag = N_instances
    nz_lags = [N_instances]

    # Set up model
    model = pf.ARIMA(data=data, ar=max_lag, ma=0, target=region,
                     family=pf.Normal(), ar_idx_list=nz_lags)

    # Set regularization for constant term
    sigma_const = np.sqrt(reg_coef_bias) ** -1
    const_idx = 0
    model.adjust_prior(const_idx, pf.Normal(0, sigma_const))

    # Fit model
    results = model.fit(method='PML')
    params = results.z_values[:-1]  # bias and AR(1) coefficient

    return params


def train_AR_multiregion(data, regions, indep_var, reg_coef_bias, N_instances):
    """
    Estimate rates of change (AR parameters) of volumetric trajectories of
    different brain areas for a group of subjects (multiple data instances).
    It calls train_AR.
    
    Parameters
    ----------
    data: pandas DataFrame
        Dataframe with the interleaved representation of the variables of
        interest
    regions: list (str)
        List with the brain regions whose values will be predicted
    reg_coef_bias: float
        Regularization parameter of the bias coefficient of the AR model. The
        higher it is, the closer to 0 it will be
    N_instances: int
        Number of time series instances (in this case, subjects)
    
    Returns
    -------
    ar_df: pandas Series
        AR coefficients learned for each variable of interest
    """

    # Estimate AR parameter values for each region for all subjects
    ar_df = pd.Series(index=regions)
    data_train = data.copy()
    data_train.set_index(indep_var, inplace=True)

    for col in regions:
        params = train_AR(data_train, reg_coef_bias, N_instances, col)
        ar_df[col] = params[1]

    return ar_df


def predict_test_vols_AR(test_real_vol_df, ar_df, regions):
    """
    Estimate future volumes based on estimated AR params for test subjects and
    their baseline volumes.
    Eq: vol[fup_vis] = ar_param^fup_vis * vol[0]
    
    Parameters
    ----------
    test_real_vol_df: pandas DataFrame
        Dataframe with actual time series information of variables of interest
        for subjects in the test set. It includes an index for the temporal
        order of the time points (follup_visit) and its associated subject IDs
    ar_df: pandas DataFrame
        AR parameters for each variable of interest across subjects in the
        test set
    predict_vars_names: list (str)
        List with the names of the variables of interest (variables to be
        predicted) as they appear in data
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
    for subjid in test_est_vol_df.index.unique():
        for reg in regions:
            ar_param = ar_df.loc[subjid, reg]
            test_est_vol_df.loc[subjid, reg] = \
                ar_param ** test_est_vol_df.loc[subjid,
                                                'follup_visit'].values *\
                test_est_vol_df.loc[subjid, reg].values

    # Get rid of baseline valumes after follow-up visits' estimation
    test_est_vol_df = test_est_vol_df[test_est_vol_df.follup_visit > 0]

    return test_est_vol_df


def prediction_error_estimation(test_real_trajec_df, test_est_trajec_df,
                                predict_vars_names):
    """
    Calculate absolute error of AR model predictions for each variable of
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


def train_test_AR_model(train_trajec_df, test_real_trajec_df, group_analysis,
                        reg_coef_bias, predict_vars_names, N_tpoints,
                        n_windows, nsubj_window):

    """
    As the name states, this function trains the AR model and generates
    prospective predictions for subjects in the test set. Then, it estimates
    prediction errors on those prediction in absolute value
    
    Parameters
    ----------
    test_trajec_df: pandas DataFrame
        Dataframe with time series information of variables of interest,
        their associated risk variable (RV), subject IDs and an index for the
        temporal order of the time points (follup_visit). It includes an index
        for the temporal order of the time series points (follup_visit)
    test_real_trajec_df: pandas DataFrame
        Dataframe with actual time series information of variables of interest
        for subjects in the test set.
    group_analysis: str
        Group to be analyzed (control or patient)
    reg_coef_bias: float
        Regularization parameter of the bias coefficient of the AR model. The
        higher it is, the closer to 0 it will be
    predict_vars_names: list (str)
        List with the names of the variables of interest (variables to be
        predicted) as they appear in data
    N_tpoints: int
        Number of time points of the time series. The code assumes that they
        are the same across instances and that there are no missing values
    n_windows: int, optional
        Number of window-based estimates to be retrieve by this function
    nsubj_window: int, optional
        Number of subjects retrieved per window estimation

    Returns
    -------
    test_diff_trajec_df: pandas DataFrame
        Dataframe with AR model prediction errors (in absolute value) for
        follow-up visits
    """
    # Estimate AR params (training set) across subsets of subjects (windows)
    ar_df_windows = AR_window_estimation(data=train_trajec_df,
                                         group_analysis=group_analysis,
                                         reg_coef_bias=reg_coef_bias,
                                         predict_vars_names=predict_vars_names,
                                         N_tpoints=N_tpoints,
                                         n_windows=n_windows,
                                         nsubj_window=nsubj_window)

    # Estimate AR parameters for subjects in test set
    ar_df_subj_test = AR_test_estimate(ar_df_groups=ar_df_windows,
                                       test_trajec_df=test_real_trajec_df,
                                       predict_vars_names=predict_vars_names)

    # Predict variables of interest at follow-up visits for subjects in test
    test_est_trajec_df = predict_test_vols_AR(test_real_trajec_df,
                                              ar_df_subj_test,
                                              predict_vars_names)

    # Calculate the model's estimation error in test set (in absolute value)
    test_diff_trajec_df = prediction_error_estimation(test_real_trajec_df,
                                                      test_est_trajec_df,
                                                      predict_vars_names)

    return test_diff_trajec_df


"""
INPUT VARIABLES
"""

analysis_dir = opj('/data2/eduardo/results/HD_n_Controls/ROI-feats/'
                   'vol_prediction/TRACK')
regions = ['Thalamus', 'Caudate', 'Putamen']
reg_coef_bias = 1e6
group_analysis = 'preHD'
n_windows = 6
nsubj_window = 20
store_flag = False
N_tpoints = 7

"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':

    # Load subcortical vols (adjusted by ICV); retrieve train and test sets
    vols_df = pd.read_csv(opj(analysis_dir, 'vols_longit_norm.csv'))
    vols_df = vols_df[vols_df.group == group_analysis]
    train_vol_df = vols_df[vols_df['pred_use'] == 'train']
    test_real_vol_df = vols_df[vols_df['pred_use'] == 'test']
    
    # Train, test and estimate prospective errors of AR model
    test_diff_trajec_df = train_test_AR_model(train_vol_df, test_real_vol_df,
                                              group_analysis, reg_coef_bias,
                                              regions, N_tpoints, n_windows,
                                              nsubj_window)
    
    # Store prediction errors of AR model
    if store_flag:
        test_diff_trajec_df.to_csv(opj(analysis_dir, 'AR_absolute_errors.csv'))
