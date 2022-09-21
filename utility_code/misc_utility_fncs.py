#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
File with utility functions used to retrieve FreeSurfer's subcortical
segmentation results and check QA through an automated analysis (QA Tools).
It also includes complementary utilities for brain plotting,
statistical tests, volumetric slopes estimation and detrending of covariates,
among others.

Created on Wed Nov 28 12:25:31 2018

@author: Eduardo Castro
"""

import pickle
from os.path import join as opj
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import pandas as pd
from os.path import exists
from scipy.stats import ttest_ind, ranksums
import os
import nibabel as nib


"""
FUNCTION DEFINITIONS
"""


def retrieve_valid_visits(logs_dir, subj_vst_list):
    """
    Get the list of valid visits of a given subject based on log files (those
    without errors)
    """
    subj_vst_out = []
    for subj_vst in subj_vst_list:
        fname = opj(logs_dir, subj_vst + '_out.log')
        if exists(fname):
            f = open(fname, 'r')
            lines = f.read()
            f.close()
            indx = lines.find('exited with ERRORS')
            if indx == -1:
                subj_vst_out.append(subj_vst)
    return subj_vst_out


def set_fig_bg_contrast(sh, vmin=0, vmax=130):
    """
    Define contrast between figure and background
    sh: nilearm slicer figure object
    """
    for vname, vh in sh.axes.iteritems():
        vax = vh.ax
        ims = vax.get_images()
        ims[1].set_clim(vmin=vmin, vmax=vmax)
    return


def multiple_compar_corr(pvals, alpha, method='FDR'):
    """
    Detect indexes of those features that survive multiple comparisons
    correction, either FDR or Bonferroni, based on the provided confidence
    level.
    """
    N = len(pvals)
    if method == 'bonferroni':
        surv_idx = np.where(pvals < alpha/N)[0]
    elif method == 'FDR':
        idx_sort = np.argsort(pvals)
        pvals_sort = pvals[idx_sort]
        fdr_vals = alpha/N * np.arange(1, N+1)
        surv_idx_sort = np.where(pvals_sort < fdr_vals)[0]
        surv_idx = idx_sort[surv_idx_sort]
    else:
        raise ValueError("invalid value for method")

    p_thr = np.ones(N)
    p_thr[surv_idx] = pvals[surv_idx]

    return p_thr, surv_idx


def make_weigths_aseg_nifti(weights_df_fn, aseg_nii_fn, out_fn):
    """
    Generate a nifti volume by mapping weight values from a dataframe (per
    ROI, aseg-like ROI names format) to the CVS-in-MNI aseg template nifti
    file. Output nifti is stored in defined output file
    """
    aseg_dir = os.path.dirname(aseg_nii_fn)
    aseg_lab_df = pd.read_csv(opj(aseg_dir, 'aseg_labels.csv'))
    aseg_lab_df.set_index('Index', inplace=True)
    weights_df = pd.read_csv(weights_df_fn)
    weights_df.set_index('Brain Region', inplace=True)

    aseg_nii = nib.load(aseg_nii_fn)
    affine = aseg_nii.affine
    hdr = aseg_nii.header

    aseg_vols = aseg_nii.get_data().astype(np.float64)
    aseg_lab_vals = np.unique(aseg_vols.ravel())

    # Reassign values according to weights dataframe
    for wrind in range(len(weights_df)):
        weight_row = weights_df.ix[wrind]
        label = weight_row.name
        weight = weight_row['Weight']
        indx = aseg_lab_df['Brain Region'].str.contains((r'[a-zA-Z-]*%s'
                                                         '[a-zA-Z-]*') % label)
        indx = aseg_lab_df[indx].index.tolist()
        if indx:
            print('Assigning value for region %d (%s)' % (wrind, label))
        for i in indx:
            aseg_vols[aseg_vols == i] = weight

    # When done with weights dataframe values, remove remaining integer vals
    aseg_lab_vals = np.unique(aseg_vols.ravel())
    for val in aseg_lab_vals:
        if int(val) == val:
            aseg_vols[aseg_vols == val] = 0

    # Store weights volume
    weights_nii = nib.Nifti1Image(aseg_vols, affine, header=hdr)
    nib.save(weights_nii, out_fn)


def detrend_vols_covars(vols_df, regions, cont_vars, discr_vars=None,
                        group_var='group', subjid_var=None, visdy_var=None,
                        addl_vars=None):
    """
    Regress out the effects of both continuous and discrete demographic
    variables from the volumes specified in the volumetric dataframe
    """
    ctr_idx = vols_df[group_var] == 'control'
    controls_df = vols_df[ctr_idx]
    nobs = controls_df.shape[0]

    # Retrieve covariates ONLY from controls
    covar_mat_ctr = controls_df[cont_vars].values
    dv_dict = {}
    if discr_vars is not None:
        for dv in discr_vars:
            dv_dict[dv] = controls_df[dv].unique()
            for val in dv_dict[dv][:-1]:    # avoid collinearity
                regressor = (controls_df[dv] == val).astype(int).values
                covar_mat_ctr = np.hstack((covar_mat_ctr,
                                           regressor[:, np.newaxis]))
    # Add column for intercept
    ncovars = covar_mat_ctr.shape[1]
    covar_mat_ctr = np.hstack((covar_mat_ctr, np.ones((nobs, 1))))
    feat_mat = controls_df[regions]
    # Solve feat_mat = covars_mat*beta
    # (solve using statsmodels (not np.linalg) in case there are nan values)
    covars_list = cont_vars
    if discr_vars is not None:
        covars_list = covars_list + discr_vars
    covars_list = covars_list + ['bias']

    beta = np.zeros((ncovars, len(regions)))
    coeff_df = pd.DataFrame(index=covars_list, columns=regions)
    for idx, col in enumerate(regions):
        dv = feat_mat[col]
        mdl = sm.OLS(dv, covar_mat_ctr, missing='drop')
        results = mdl.fit()
        beta[:, idx] = results.params.values[:-1]   # everything but bias
        coeff_df[col] = results.params.values   # linear fit for all regions

    # Subtract linear fit of covariates from features of all subjects
    covar_mat_all = vols_df[cont_vars].values
    if discr_vars is not None:
        for dv in discr_vars:
            for val in dv_dict[dv][:-1]:    # avoid collinearity
                regressor = (vols_df[dv] == val).astype(int).values
                covar_mat_all = np.hstack((covar_mat_all,
                                           regressor[:, np.newaxis]))
    if addl_vars is None:
        vols_adj_df = vols_df[regions] - np.dot(covar_mat_all, beta)
    else:
        vols_adj_df = vols_df[regions + addl_vars].copy()
        vols_adj_df[regions] = vols_adj_df[regions] \
            - np.dot(covar_mat_all, beta)

    # Incorporate additional variables (on top of group) if defined
    vols_adj_df[group_var] = vols_df[group_var]
    if subjid_var is not None:
        vols_adj_df[subjid_var] = vols_df[subjid_var]
    if visdy_var is not None:
        vols_adj_df[visdy_var] = vols_df[visdy_var]

    return vols_adj_df, coeff_df


def univ_test_vols(vols_df, group_var, group_vals=['control', 'preHD'],
                   mult_corr=None, regions=None, stat_method='ttest'):
    """
    Perform a two sample test per ROI volume between groups of subjects
    (by default 'preHD' and 'control')
    """
    stat_function_dict = {'ttest': ttest_ind, 'wilcoxon': ranksums}
    stat_function = stat_function_dict[stat_method]

    if regions is None:
        regions = list(vols_df.columns)
        if group_var in regions:
            regions.remove(group_var)

    stat_vals = np.zeros(len(regions))
    p_vals = np.zeros(len(regions))
    t_vals = np.zeros(len(regions))

    group1 = vols_df[vols_df[group_var] == group_vals[0]][regions]
    if len(group_vals) == 1:
        group2 = vols_df[vols_df[group_var] != group_vals[0]][regions]
    else:
        group2 = vols_df[vols_df[group_var] == group_vals[1]][regions]

    for idx, reg in enumerate(regions):
        stat_vals[idx], p_vals[idx] = stat_function(group2[reg], group1[reg])
        t_vals[idx] = ttest_ind(group2[reg], group1[reg])[0]
    sign_test = np.sign(t_vals)

    if mult_corr is None:
        col_indx = list(np.argsort(p_vals))
    else:
        p_thr, surv_idx = multiple_compar_corr(p_vals, 0.05,
                                               method=mult_corr)
        col_indx = surv_idx

    stats_df = pd.DataFrame(index=np.array(regions)[col_indx])
    stats_df['p-values'] = p_vals[col_indx]
    stats_df['statistic'] = stat_vals[col_indx]
    stats_df['Weight'] = -sign_test[col_indx] * np.log10(p_vals[col_indx])
    stats_df.index.rename('Brain Region', inplace=True)
    stats_df.sort_values('p-values', inplace=True)

    return stats_df


def QA_pass(subj_dir, all_subjs):
    """
    Performs QA assessment of provided subject/visit tuples (list of strings)
    based on FreeSurfer's QA tools
    """
    qa_fn = opj(subj_dir, 'recon_checker.ALL.summary.log')
    outl_subjs = [line.split(' ')[0] for line in open(qa_fn)
                  if ' 0 outlier' in line]
    outl_subjs = list(np.setdiff1d(all_subjs, outl_subjs))
    snr_out_subjs = [line.split('\t')[0].strip() for line in open(qa_fn)
                     if '<---' in line]

    return outl_subjs, snr_out_subjs


def consolidate_aseg_df(vol_raw_df):
    """
    Gives final format to FreeSurfer's volumetric estimates of subcortical
    regions after performing brain segmentation by adding up the halves in both
    left and right hemispheres.
    """
    vol_sum_df = pd.DataFrame(index=vol_raw_df.index)
    for col in list(vol_raw_df.columns):
        if 'Left' in col:
            outcol = col.split('Left-')[-1]
            vol_sum_df[outcol] = vol_raw_df['-'.join(['Left', outcol])] +\
                vol_raw_df['-'.join(['Right', outcol])]
        elif 'lh' in col:
            outcol = col.split('lh')[-1]
            vol_sum_df[outcol] = vol_raw_df['lh' + outcol] +\
                vol_raw_df['rh' + outcol]
        elif ('rh' not in col) and ('Right' not in col):
            vol_sum_df[col] = vol_raw_df[col]

    return vol_sum_df


def volume_slope_estimation(vols_df, regions, addl_vars=None):
    """
    Estimate rate of change of volumes of different brain areas across visits
    for each subject
    """
    slopes_df = pd.DataFrame(columns=regions + ['group'])
    vols_df['visdy'] = vols_df['visdy']/365.

    for col in regions:
        col_alt = ''.join(col.split('-'))
        if col_alt[0].isdigit():
            col_alt = 'a' + col_alt
        adj_df = vols_df.copy()
        adj_df.rename(columns={col: col_alt}, inplace=True)

        col_fit_df = adj_df.groupby('subjid').apply(lambda x:
                                                    smf.ols(formula='%s ~ visdy'
                                                            % col_alt,
                                                            data=x,
                                                            missing='drop').fit().params)

        slopes_df[col] = col_fit_df['visdy']
        slopes_df['group'] = adj_df.groupby('subjid')['group'].first()
        if addl_vars is not None:
            for var in addl_vars:
                # If continuous vars like age, take the baseline value
                slopes_df[var] = adj_df.groupby('subjid')[var].first()

    slopes_df.reset_index(inplace=True)

    return slopes_df


def gen_weights_map_file(results_dir, classifier_name):
    """
    Generate mean weights map (EXPAND!!!)
    """

    classifier_name = classifier_name.replace(" ", "_")
    classifier = classifier_name + '.pkl'
    rank = pd.read_csv(opj(results_dir, 'Ranked_features_per_fold.csv'),
                       index_col=0)
    list_folds = rank.columns.tolist()  # checking the name of features

    top_rank = pd.read_csv(opj(results_dir, 'Results4' + classifier_name +
                               '.csv'), index_col=0)
    top_rank = top_rank.columns.tolist()    # checking the number of topkvalues

    # Initialize the dataframe that will be used to save the weights
    weights_df = pd.DataFrame(index=rank[list_folds[0]], columns=top_rank)

    # The next step is to select the top_rank analyzed. The information is
    # obtained from the generated csv files

    for top_rank_name, used_ntop_indx in zip(top_rank, range(len(top_rank))):
        # Read the pickle file that contains information of the results
        # of the classifier.
        with open(opj(results_dir, classifier), 'r') as po:
            clf_dict = pickle.load(po)
        fcs = clf_dict['FittedClassifiers']

        fcs_per_split = np.array(fcs).T[used_ntop_indx, :]
        weights_per_split = np.array([this_fc.coef_ for this_fc in fcs_per_split]).squeeze()

        if weights_per_split.ndim == 1:
            number_feat = 1
            flagdimension = 1   # for one dimension, the assignment of weight values generates errors
        else:
            flagdimension = 0
            number_feat = weights_per_split.shape[1]

        # Accumulate the weights using the information of rank per fold
        accum_weights = pd.DataFrame(index=list_folds,
                                     columns=rank[list_folds[0]].tolist())

        for col, indx in zip(list_folds, range(len(list_folds))):
            if flagdimension == 0:
                weightvalues = weights_per_split[indx, :]
            else:
                weightvalues = weights_per_split[indx]

            accum_weights.loc[col,
                              rank[col].tolist()[:number_feat]] = weightvalues

        weights_df.loc[accum_weights.mean().index.tolist(),
                       top_rank_name] = accum_weights.mean().values

    # Save the weights for the classifier
    filename = opj(results_dir, 'Weights4' + classifier_name + '.csv')
    weights_df.to_csv(filename)
    message = 'Saving the file:  ' + filename
    print(message)

    return weights_df
