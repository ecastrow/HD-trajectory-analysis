#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to retrieve brain segmentation data retrieved by FreeSurfer from a given
study. This is done in the context of a longitudinal analysis (data from the
longitudinal stream only).

Created on Wed Nov 28 16:12:47 2018

@author: Eduardo Castro
"""

import os
import sys
import numpy as np
import pandas as pd
from os import environ
from os.path import join as opj
from os.path import basename, splitext, dirname
sys.path.append(os.path.abspath(opj(os.path.dirname(__file__), '..', '..')))
from utility_code import misc_utility_fncs as muf


"""
INPUT VARIABLES
"""
study_id = 'TRACK'  # TRACK, PREDICT, IMAGEHD
regions_subset = ['Lateral-Ventricle', 'Thalamus-Proper', 'Caudate',
                  'Putamen', 'Brain-Stem', 'Pallidum', 'Hippocampus',
                  'Amygdala', 'Accumbens-area']
ICV_col = 'EstimatedTotalIntraCranialVol'
if study_id == 'IMAGEHD':
    demog_vars = ['subjid', 'visit', 'age', 'sex', 'group', 'visdy', 'CAP',
                  'CAG']
else:
    demog_vars = ['subjid', 'visit', 'age', 'sex', 'group', 'visdy', 'siteid',
                  'CAG', 'CAP']

# indicate # of restricted visits (either False or number of visits to analyze)
restrict_visits = False
# Are volumes going to be scaled by baseline volume (APR)?
baseline_visit_scaling = False
# Detrend effect of confounds in volumetric measures?
confound_detrend = False

"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':

    # Define source and destination directories and files
    analysis_dir = opj('/data2/eduardo/results/HD_n_Controls',
                       'ROI-feats/slopes_longit/', study_id, 'analysis')
    proc_dir = opj('/data1/cooked/FS_PROCESSED/CCC_output', study_id)

    bad_scans_fn = opj(analysis_dir, 'bad_scans_info.csv')
    aseg_vols_fn = opj(analysis_dir, 'aseg_vols_all_visits_raw.csv')
    aseg_sum_fn = opj(analysis_dir, 'vols_raw_aseg.csv')
    aseg_demog_fn = opj(analysis_dir, 'vols_demog_raw.csv')
    aseg_norm_base_detrend_fn = opj(analysis_dir,
                                    'vols_norm_baseline_detrended.csv')
    aseg_slopes_no_detrend_fn = opj(analysis_dir, 'vols_slopes_nodetrending_'
                                    + study_id + '.csv')
    aseg_slopes_fn = opj(analysis_dir, 'vols_slopes_' + study_id + '.csv')

    demog_clinical_fn = 'imaging_clinical_demog_info_' + study_id + '.csv'
    demog_clinical_fn = opj(analysis_dir, demog_clinical_fn)
    detrend_coeff_fn = opj(analysis_dir, 'detrend_vols_coeff.csv')

    # Retrieve list of valid processed subjects/visits
    subjs_list = os.listdir(proc_dir)
    subjs_list = [sval for sval in subjs_list if '.long.' in sval]
    subj_vst_all = [sval.split('.')[0] for sval in subjs_list]
    logs_dir = opj(proc_dir, 'longit_processing/ccc_logs')
    subj_vst_ok = muf.retrieve_valid_visits(logs_dir, subj_vst_all)
    subj_vst_fault = np.setdiff1d(subj_vst_all, subj_vst_ok).tolist()

    # Remove bad quality scans (FreeSurfer's QAtools); update list of ok scans
    qa_dir = opj(proc_dir, 'QA')
    outl_subjs, snr_out_subjs = muf.QA_pass(qa_dir, subj_vst_ok)
    outl_subjs = list(np.unique(outl_subjs))
    snr_out_subjs = list(np.unique(snr_out_subjs))
    subj_vst_ok = np.setdiff1d(subj_vst_ok, snr_out_subjs+outl_subjs)

    # Store dataframe with bad quality scans
    bad_scans_df = pd.DataFrame(columns=['subjid', 'visit', 'discard_reason',
                                         'all_scans_no', 'discarded_no',
                                         'ok_discarded_frac', 'subj_visit'],
                                index=range(len(outl_subjs + snr_out_subjs
                                                + subj_vst_fault)))
    bad_scans_df['subj_visit'] = outl_subjs + snr_out_subjs + subj_vst_fault
    bad_scans_df.set_index('subj_visit', inplace=True)

    bad_scans_df.loc[subj_vst_fault, 'discard_reason'] = 'Segmentation Failure'
    bad_scans_df.loc[outl_subjs, 'discard_reason'] = 'Segmentation Outlier'
    bad_scans_df.loc[snr_out_subjs, 'discard_reason'] = 'Low WM mean SNR (<16)'
    bad_scans_df.reset_index(inplace=True)
    bad_scans_df[['subjid', 'visit']] = bad_scans_df['subj_visit']\
        .apply(lambda x: x.split('_visit_')).values.tolist()
    bad_scans_df.visit = bad_scans_df.visit.astype(int)
    del bad_scans_df['subj_visit']

    bad_scans_df['all_scans_no'] = len(subj_vst_all)
    bad_scans_df['discarded_no'] = len(snr_out_subjs + outl_subjs
                                       + subj_vst_fault)
    bad_scans_df['ok_discarded_frac'] = float(len(snr_out_subjs + outl_subjs
                                                  + subj_vst_fault))\
        / len(subj_vst_all)
    bad_scans_df.to_csv(bad_scans_fn, index=False)

    # Retrieve subcortical volumetric measures
    environ['SUBJECTS_DIR'] = proc_dir
    subj_vst_ok_full = ['.'.join([svo, 'long', 'templ_'
                                  + svo.split('_vis')[0]])
                        for svo in subj_vst_ok]
    subj_args = ' '.join(subj_vst_ok_full)

    # Generate aseg stats table, read it and polish awkward formatting
    aseg_args = ('asegstats2table --subjects %s -d comma '
                 '--tablefile %s' % (subj_args, aseg_vols_fn))
    stat = os.system(aseg_args)
    aseg_raw_df = pd.read_csv(aseg_vols_fn)

    aseg_raw_df.rename(columns={'Measure:volume': 'subjid'}, inplace=True)
    aseg_raw_df.subjid = aseg_raw_df.subjid.apply(lambda x:
                                                  x.split('.')[0])
    aseg_raw_df.set_index('subjid', inplace=True)

    # Sum up left and right hemisphere halves of the regions, store whole thing
    aseg_sum_df = muf.consolidate_aseg_df(aseg_raw_df)
    aseg_sum_df.reset_index(inplace=True)
    aseg_sum_df['visit'] = int(0)
    aseg_sum_df[['subjid', 'visit']] = aseg_sum_df.subjid.apply(lambda x:
                                                                x.split('_visit_'))\
        .values.tolist()

    # Discard scans from subjects with single visits (after QA)
    aseg_sum_df.set_index('subjid', inplace=True)
    subj_visit_count = aseg_sum_df.groupby('subjid')['visit'].count()
    aseg_sum_df = aseg_sum_df.loc[subj_visit_count > 1, :]
    aseg_sum_df.visit = aseg_sum_df.visit.astype(int)
    aseg_sum_df.reset_index(inplace=True)

    # Reorder columns (subjid and visit first)
    columns_regions = aseg_sum_df.columns.values.tolist()
    columns_regions.remove('subjid')
    columns_regions.remove('visit')
    aseg_sum_df = aseg_sum_df[['subjid', 'visit'] + columns_regions]
    vol_sum_df = aseg_sum_df.copy()
    aseg_sum_df.to_csv(aseg_sum_fn, index=False)

    # Merge demog, clinical and vol.(subset of regions) into single dataframe
    vol_sum_df.set_index(['subjid', 'visit'], inplace=True)
    cols = regions_subset + [ICV_col]
    vol_sum_df = vol_sum_df[cols]
    vol_sum_df.reset_index(inplace=True)

    demog_clinical_df = pd.read_csv(demog_clinical_fn)
    demog_clinical_df = demog_clinical_df[demog_vars]
    if study_id == 'PREDICT':
        demog_clinical_df.subjid = demog_clinical_df.subjid.map('{:06d}'.format)
    vol_demog_df = pd.merge(vol_sum_df, demog_clinical_df, how='inner',
                            on=['subjid', 'visit'])

    # Include region with the sum of striatal structures' volumes
    vol_demog_df['Striatum'] = vol_demog_df['Caudate'] +\
        vol_demog_df['Putamen'] + vol_demog_df['Accumbens-area']
    vol_demog_df.to_csv(aseg_demog_fn, index=False)

    # Correct volumetric measures from demog. confounds for subset of regions
    regions_subset.append('Striatum')   # Include overall striatal vol too
    if confound_detrend:
        cont_vars = ['age']
        discr_vars = ['sex']
        if study_id == 'IMAGEHD':
            addl_vars = ['subjid', 'visdy', 'visit']
        else:
            addl_vars = ['subjid', 'visdy', 'visit', 'siteid']
        vol_adj_df, coeff_df = muf.detrend_vols_covars(vol_demog_df,
                                                       regions_subset,
                                                       cont_vars=cont_vars,
                                                       discr_vars=discr_vars,
                                                       group_var='group',
                                                       addl_vars=addl_vars)
        coeff_df.to_csv(detrend_coeff_fn)
    else:
        vol_adj_df = vol_demog_df

    # Copy adjusted volumetric estimates; retrieve controls and pre-HD only
    vol_adj_df = vol_adj_df[vol_adj_df.group != 'earlyHD']

    # Few visits discarded due to transition to early-HD. Get rid of remaining single-visit subjects
    visit_df = vol_adj_df.groupby('subjid').count().loc[:, 'visit']
    visit_df = visit_df[visit_df > 1]
    subj_ids = visit_df.index
    vol_adj_df = vol_adj_df[vol_adj_df.subjid.isin(subj_ids)]

    vol_adj_df = vol_adj_df.reset_index(drop=True)
    vol_norm_base = vol_adj_df.copy()
    if study_id == 'PREDICT':
        vol_norm_base.visit = vol_norm_base.visit.astype(int)

    # Normalize by baseline volume
    if baseline_visit_scaling:
        idx_base = vol_norm_base.groupby('subjid')['visit'].transform('idxmin')
        base_vols_df = vol_norm_base.iloc[idx_base, :].reset_index(drop=True)
        vol_norm_base[regions_subset] = vol_norm_base[regions_subset] / \
            base_vols_df[regions_subset]
        vol_norm_base.to_csv(aseg_norm_base_detrend_fn)

    # Estimate volumetric slopes
    if study_id == 'IMAGEHD':
        addl_vars = None
    else:
        addl_vars = ['siteid']

    # Evaluate whether a restricted number of visits will be evaluated
    if restrict_visits:
        vol_norm_base = vol_norm_base.groupby('subjid').\
            apply(lambda x: x[x.visit < x.visit.min() + restrict_visits])
        vol_norm_base = vol_norm_base.reset_index(drop=True)
        split_fn = splitext(basename(aseg_slopes_fn))
        dir_fn = dirname(aseg_slopes_fn)
        aseg_slopes_fn = opj(dir_fn, split_fn[0] + '_' +
                             str(restrict_visits) + 'visits' + split_fn[1])
        vol_norm_base.to_csv(aseg_slopes_fn)

    slopes_df = muf.volume_slope_estimation(vol_norm_base, regions_subset,
                                            addl_vars=addl_vars)
    slopes_df['study_id'] = study_id
    slopes_df.to_csv(aseg_slopes_fn)
