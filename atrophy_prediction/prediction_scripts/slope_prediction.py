#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:22:58 2020

@author: Eduardo Castro

Script to do a linear fit between longitudinal volume measures and time for
pre-HD individuals. Then, prospective predictions for follow-up visits of
subjects in the test set are generated.
"""

import pandas as pd
import numpy as np
from os.path import join as opj
from sklearn import linear_model


"""
FUNCTION DEFINITIONS
"""


def train_slope(data, indep_var, region):
    """
    Estimates average volumetric linear decay for multiple subjects and
    retrieves learned model.
    """

    # Fit linear association between volume and follow-up visits
    lin_mdl = linear_model.Ridge()
    lin_mdl.fit(data[indep_var].values[:, None], data[region].values)

    return lin_mdl


def train_slope_multiregion(data, regions, indep_var):
    """
    Estimate linear rates of change  of volumetric trajectories of
    different brain areas for a group of subjects (multiple data instances).
    It calls train_AR.
    """

    # Estimate AR parameter values for each region for all subjects
    lin_models = pd.DataFrame(index=regions, columns=['model'])

    for col in regions:
        lin_mdl = train_slope(data, indep_var, col)
        lin_models.loc[col, 'model'] = lin_mdl

    return lin_models


def slope_test_estimate(slope_df_groups, test_vol_df, regions):
    """
    Estimate slopess of the provided regions for each test subject. This
    is done after making a linear fit between the mean number of CAG repeats of
    the subjects used for each window and their estimated slopes during
    training.
    """
    # Fit linear association between CAG repeats and per-region AR coefficients
    CAGfit_models = {}
    for reg in regions:
        mdl = linear_model.Ridge()
        mdl.fit(slope_df_groups['CAG_mean'].values[:, None],
                slope_df_groups[reg].map(lambda x: x.coef_).values)
        CAGfit_models[reg] = mdl

    # Estimate slopes for test subjects
    CAG_df = test_vol_df[['subjid', 'CAG']]
    CAG_df = CAG_df.groupby('subjid').first()
    slope_df_subj_test = pd.DataFrame(index=CAG_df.index, columns=regions)
    for reg in regions:
        slope_df_subj_test[reg] = \
            np.squeeze(CAGfit_models[reg].predict(CAG_df.values))

    return slope_df_subj_test


def predict_test_vols_slope(test_real_vol_df, slope_df, regions):
    """
    Estimate future volumes based on estimated slopes of test subjects and
    their baseline volumes.
    Eq: vol[fup_vis] = slope * fup_vis + vol[0]
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
            slp = slope_df.loc[subjid, reg]
            test_est_vol_df.loc[subjid, reg] = \
                slp * test_est_vol_df.loc[subjid,
                                          'follup_visit'].values +\
                test_est_vol_df.loc[subjid, reg].values

    # Get rid of baseline valumes after follow-up visits' estimation
    test_est_vol_df = test_est_vol_df[test_est_vol_df.follup_visit > 0]

    return test_est_vol_df


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


def slope_window_estimation(data, group_analysis, regions, n_windows=6,
                            nsubj_window=20, fixed_stride=5):
    """
    Estimate slopes of different brain areas for various groups of
    subjects that overlap with each other (windows). These are defined by
    either an initial value of CAG repeats (pre-HD) or a fixed window-based
    stride (controls). Details of window-based subject retrieval are described
    in retrieve_subjects_window_estimation. It also allows the option to
    randomize CAG repeats across subjects prior to training to test the null
    hypothesis (CAG repeats do not influence the quality of the prospective
    volume prediction on follow-up visits).
    """
    indep_var = 'follup_visit'
    slope_df_groups = pd.DataFrame(index=range(n_windows), columns=regions)
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

        # Compute slopes
        N_instances = len(data_subset.subjid.unique())
        lin_models = train_slope_multiregion(data_subset, regions, indep_var)

        # Estimate AR parameters; assign subgroup info
        slope_df_groups.loc[window_idx, regions] =\
            lin_models['model'].values
        slope_df_groups.loc[window_idx, 'CAG_mean'] =\
            data_subset.groupby('subjid').first().CAG.mean()
        slope_df_groups.loc[window_idx, 'N_instances'] = N_instances

        if group_analysis == 'preHD':
            CAG_base = CAG_base + 1

    return slope_df_groups


def prediction_error_estimation(test_real_vol_df, test_est_vol_df, regions):
    """
    Calculate absolute error of AR model predictions for each region across
    follow-up visits
    """
    test_real_cp = test_real_vol_df.copy()
    test_est_cp = test_est_vol_df.copy()
    test_est_cp.reset_index(inplace=True)
    test_est_cp.set_index(['subjid', 'follup_visit'], inplace=True)
    test_real_cp.set_index(['subjid', 'follup_visit'], inplace=True)

    # Estimate difference from actual values; take absolute value
    test_diff_vol_df = test_est_cp
    test_diff_vol_df = test_diff_vol_df - test_real_cp[regions]
    test_diff_vol_df = test_diff_vol_df.abs()

    # Get rid of baseline visit
    test_diff_vol_df =\
        test_diff_vol_df[~test_diff_vol_df.index.
                         get_level_values('follup_visit').isin([0])]

    return test_diff_vol_df



def train_test_slope_model(train_trajec_df, test_real_trajec_df,
                           group_analysis, predict_vars_names,
                           n_windows, nsubj_window):

    """
    As the name states, this function trains the AR model and generates
    prospective predictions for subjects in the test set. Then, it estimates
    prediction errors of those predictions in absolute value
    
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
    predict_vars_names: list (str)
        List with the names of the variables of interest (variables to be
        predicted) as they appear in data
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
    slope_df_windows = slope_window_estimation(data=train_trajec_df,
                                               group_analysis=group_analysis,
                                               regions=predict_vars_names,
                                               n_windows=n_windows,
                                               nsubj_window=nsubj_window)

    # Estimate AR parameters for subjects in test set
    slope_df_subj_test = slope_test_estimate(slope_df_groups=slope_df_windows,
                                             test_vol_df=test_real_trajec_df,
                                             regions=predict_vars_names)

    # Predict variables of interest at follow-up visits for subjects in test
    test_est_trajec_df = predict_test_vols_slope(test_real_trajec_df,
                                                 slope_df_subj_test,
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
store_dir = opj(analysis_dir, 'final_results')
regions = ['Thalamus', 'Caudate', 'Putamen']
group_analysis = 'preHD'
n_windows = 6
nsubj_window = 20
store_flag = True

"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':

    # Load subcortical vols (adjusted by ICV); retrieve train and test sets
    vols_df = pd.read_csv(opj(analysis_dir, 'vols_longit_norm.csv'))
    vols_df = vols_df[vols_df.group == group_analysis]
    train_vol_df = vols_df[vols_df['pred_use'] == 'train']
    test_real_vol_df = vols_df[vols_df['pred_use'] == 'test']
    
    # Train, test and estimate prospective errors of slope model
    test_diff_vol_df = train_test_slope_model(train_vol_df, test_real_vol_df,
                                              group_analysis, regions,
                                              n_windows, nsubj_window)
    
    # Store prediction errors of slope model
    if store_flag:
        test_diff_vol_df.to_csv(opj(store_dir, 'slope_absolute_errors.csv'))

