#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that runs classification models (training and prediction) and generates
both t-score maps and maps of classification weights

Created on Thu Dec 13 20:45:28 2018

@author: Eduardo Castro
"""

from os.path import join as opj
import numpy as np
import pandas as pd
import imaging_stat_utilities as imu
from matplotlib.colors import LinearSegmentedColormap as LSCM
from nilearn import plotting
from os import environ
import warnings
import os
import sys
sys.path.append(os.path.abspath(opj(os.path.dirname(__file__), '..', '..')))
from classification_core.Main_script_runclassification import execute_pipeline


"""
INPUT VARIABLES
"""
inter_study = False
inter_site = True
univ_flag = False
univ_area = 'Caudate'       # 'Striatum', 'Caudate'
do_ttest_brain_map = False
do_classif_brain_map = False
do_classification = False
fixed_kfolds = 10
mult_corr = 'bonferroni'   # 'bonferroni', 'FDR' or None
fig_type = 'vector_graphics'    # 'raster_graphics', 'vector_graphics'
stat_method = 'wilcoxon'
target_label = 'group'
studyid_list = ['TRACK', 'PREDICT', 'IMAGEHD']
regions_subset = ['Lateral-Ventricle', 'Thalamus-Proper', 'Caudate',
                  'Putamen', 'Brain-Stem', 'Pallidum', 'Hippocampus',
                  'Amygdala', 'Accumbens-area']
fixed_regions = [univ_area]
aseg_dir = '/data2/eduardo/results/HD_n_Controls/ROI-feats/slopes_longit/'

# indicate # of restricted visits (either False or number of visits to analyze)
restrict_visits = False

"""
MAIN CODE
"""

# Run univariate stats for TRACK-HD only
aseg_nii_fn = opj(aseg_dir, 'aseg.nii.gz')
if do_ttest_brain_map:
    study_id = 'TRACK'
    analysis_dir = opj('/data2/eduardo/results/HD_n_Controls',
                       'ROI-feats/slopes_longit/', study_id, 'analysis')
    aseg_slopes_fn = opj(analysis_dir, 'vols_slopes_' + study_id + '.csv')
    ttest_brain_csv_fn = opj(analysis_dir, 'ttest_FDRcorr_brain.csv')
    ttest_brain_nii_fn = opj(analysis_dir, 'ttest_weights_brain.nii.gz')
    slopes_df = pd.read_csv(aseg_slopes_fn, index_col=0)

    # Univariate test per se
    group_var = 'group'
    group_vals = ['control']
    stats_df = imu.univ_test_vols(slopes_df, group_var=group_var,
                                  group_vals=group_vals,
                                  mult_corr=mult_corr,
                                  regions=regions_subset,
                                  stat_method=stat_method)
    stats_df.to_csv(ttest_brain_csv_fn)

    # Generate nifti volume and associated figure
    weights_df_fn = ttest_brain_csv_fn
    imu.make_weigths_aseg_nifti(weights_df_fn, aseg_nii_fn,
                                out_fn=ttest_brain_nii_fn)

    # Generate plot
    thresh = 0.1
    bg_img = opj(environ['FREESURFER_HOME'], 'subjects/cvs_avg35_inMNI152/mri/'
                 'orig.mgz')
    cmap = LSCM.from_list(name='yellow_cyan',
                          colors=['cyan', 'blue', 'grey', 'red', 'yellow'])
    pv = plotting.plot_stat_map(bg_img=bg_img, stat_map_img=ttest_brain_nii_fn,
                                threshold=thresh,
                                cmap=cmap,
                                black_bg=True,
                                cut_coords=[-25, 13, 3])
    imu.set_fig_bg_contrast(pv)

    if fig_type == 'raster_graphics':
        ext_fname = '.png'
    else:
        ext_fname = '.svg'

    pv.savefig(ttest_brain_nii_fn.split('.')[0] + ext_fname, dpi=300)

# Structure code to run classification in three ways (for all regions and striatum only)
if do_classification:
    target_label = 'group'
    # Multivariate?
    if univ_flag:
        list_fix_features = []
        suffix = univ_area
    else:
        list_fix_features = []
        suffix = 'All_Regions'

    # Inter-site? Inter-study?
    if inter_site:
        kfold_type = 'dependent'
        studyid_list.remove('IMAGEHD')
        group_label = 'siteid'
        del_cols = ['subjid', 'study_id']
        kfold_val = None
    elif inter_study:
        kfold_type = 'dependent'
        group_label = 'study_id'
        del_cols = ['subjid', 'siteid']
        kfold_val = None
    else:
        studyid_list = ['IMAGEHD']
        kfold_type = 'fixed'
        group_label = 'subjid'
        del_cols = ['study_id', 'siteid']
        kfold_val = fixed_kfolds

    # If inter-study, run for all studies at once. Else, run for each study separately
    all_slopes_dir = ('/data2/eduardo/results/HD_n_Controls/'
                      'ROI-feats/slopes_longit/ALL_STUDIES/'
                      'analysis')
    if isinstance(restrict_visits, int):
        all_slopes_dir = opj('/data2/eduardo/results/HD_n_Controls/'
                             'ROI-feats/slopes_longit/ALL_STUDIES/',
                             str(restrict_visits) + 'visits',
                             'analysis')
        if not os.path.exists(all_slopes_dir):
            os.makedirs(all_slopes_dir)

    all_slopes_df = pd.DataFrame()
    if inter_study:
        for study_id in studyid_list:
            analysis_dir = opj('/data2/eduardo/results/HD_n_Controls',
                               'ROI-feats/slopes_longit/', study_id,
                               'analysis')
            if isinstance(restrict_visits, int):
                aseg_slopes_fn = opj(analysis_dir,
                                     '_'.join(['vols_slopes', study_id,
                                               str(restrict_visits) +
                                               'visits' + '.csv']))
            elif not restrict_visits:
                aseg_slopes_fn = opj(analysis_dir, 'vols_slopes_' + study_id +
                                     '.csv')
            else:
                warnings.warn('Undefined value for "restrict_visits"')

            slopes_df = pd.read_csv(aseg_slopes_fn, index_col=0)
            all_slopes_df = all_slopes_df.append(slopes_df)

        # Generate complete and for-classification dataframes
        all_slopes_complete_fn = opj(all_slopes_dir,
                                     'vols_slopes_ALL_SLOPES.csv')
        all_slopes_df.to_csv(all_slopes_complete_fn)
        all_slopes_classif_fn = opj(all_slopes_dir,
                                    'vols_slopes_ALL_SLOPES_classif.csv')

        # Delete unwanted columns
        for col in del_cols:
            del all_slopes_df[col]

        # Adjust dataframe based on univariate or multivariate analysis
        if not univ_flag:
            del all_slopes_df['Striatum']
        else:
            if univ_area == 'Caudate':
                del all_slopes_df['Striatum']
            for col in np.setdiff1d(regions_subset, fixed_regions):
                del all_slopes_df[col]
            all_slopes_df['dummy'] = all_slopes_df[fixed_regions]

        # Give group variable numeric values
        all_slopes_df['group'] = all_slopes_df.group.apply(lambda x: -1
                                                           if x == 'control'
                                                           else 1)

        all_slopes_df.to_csv(all_slopes_classif_fn)

        # Generate classification directory
        classif_dir_root = ('/data2/eduardo/results/HD_n_Controls/'
                            'ROI-feats/slopes_longit/ALL_STUDIES/'
                            'classification')
        if isinstance(restrict_visits, int):
            classif_dir_root = opj('/data2/eduardo/results/HD_n_Controls/'
                                   'ROI-feats/slopes_longit/ALL_STUDIES/',
                                   str(restrict_visits) + 'visits',
                                   'classification')

        classif_dir = opj(classif_dir_root, suffix)
        if not os.path.exists(classif_dir):
            os.makedirs(classif_dir)

        # Run classifier
        print('Running classifier!!')
        execute_pipeline(csvfilename=all_slopes_classif_fn, outdir=classif_dir,
                         list_fix_features=list_fix_features,
                         kfold_type=kfold_type, kfold_val=kfold_val,
                         group_label=group_label, target_label=target_label)
    else:
        for study_id in studyid_list:
            analysis_dir = opj('/data2/eduardo/results/HD_n_Controls',
                               'ROI-feats/slopes_longit/', study_id,
                               'analysis')

            if isinstance(restrict_visits, int):
                aseg_slopes_fn = opj(analysis_dir,
                                     '_'.join(['vols_slopes', study_id,
                                               str(restrict_visits) +
                                               'visits' + '.csv']))
                aseg_slopes_classif_fn = opj(analysis_dir,
                                             '_'.join(['vols_slopes', study_id,
                                               str(restrict_visits) +
                                               'visits_classif' + '.csv']))
            elif not restrict_visits:
                aseg_slopes_fn = opj(analysis_dir, 'vols_slopes_' + study_id +
                                     '.csv')
                aseg_slopes_classif_fn = opj(analysis_dir, 'vols_slopes_' +
                                         study_id + '_classif.csv')
            else:
                warnings.warn('Undefined value for "restrict_visits"')

            # Generate complete and for-classification dataframes
            slopes_df = pd.read_csv(aseg_slopes_fn, index_col=0)

            # Delete unwanted columns
            if study_id == 'IMAGEHD':
                del_cols.remove('siteid')
            for col in del_cols:
                del slopes_df[col]

            # Adjust dataframe based on univariate or multivariate analysis
            if not univ_flag:
                del slopes_df['Striatum']
            else:
                if univ_area == 'Caudate':
                    del slopes_df['Striatum']
                for col in np.setdiff1d(regions_subset, fixed_regions):
                    del slopes_df[col]
                slopes_df['dummy'] = slopes_df[fixed_regions]
            slopes_df['group'] = slopes_df.group.apply(lambda x: -1
                                                       if x == 'control'
                                                       else 1)
            slopes_df.to_csv(aseg_slopes_classif_fn)

            classif_dir_root = opj('/data2/eduardo/results/HD_n_Controls',
                                   'ROI-feats/slopes_longit/', study_id,
                                   'classification')
            if isinstance(restrict_visits, int):
                classif_dir_root = opj('/data2/eduardo/results/HD_n_Controls/'
                                       'ROI-feats/slopes_longit/', study_id,
                                       str(restrict_visits) + 'visits',
                                       'classification')

            classif_dir = opj(classif_dir_root, suffix)
            if not os.path.exists(classif_dir):
                os.makedirs(classif_dir)

            # Run classifier
            execute_pipeline(csvfilename=aseg_slopes_classif_fn,
                             outdir=classif_dir,
                             list_fix_features=list_fix_features,
                             kfold_type=kfold_type, kfold_val=kfold_val,
                             group_label=group_label,
                             target_label=target_label)

if do_classif_brain_map:

    # Retrieve maps from classification (TRACK-only, inter-site)
    suffix = 'All_Regions'
    study_id = 'TRACK'
    classif_dir_root = opj('/data2/eduardo/results/HD_n_Controls',
                           'ROI-feats/slopes_longit/', study_id,
                           'classification')
    classif_dir = opj(classif_dir_root, suffix)

    weights_df = imu.gen_weights_map_file(classif_dir, 'Logistic_Regression')
    weights_df.rename(columns={weights_df.columns[-1]: 'Weight'}, inplace=True)
    weights_df.index.rename('Brain Region', inplace=True)
    weights_csv_fn = opj(classif_dir, 'weights_brain_map.csv')
    weights_df.to_csv(weights_csv_fn)

    classif_brain_nii_fn = opj(classif_dir, 'classif_weights_brain.nii.gz')
    imu.make_weigths_aseg_nifti(weights_csv_fn, aseg_nii_fn,
                                out_fn=classif_brain_nii_fn)

    # Generate plot
    thresh = 0.1
    bg_img = opj(environ['FREESURFER_HOME'], 'subjects/cvs_avg35_inMNI152/mri/'
                 'orig.mgz')
    cmap = LSCM.from_list(name='yellow_cyan',
                          colors=['cyan', 'blue', 'grey', 'red', 'yellow'])

    pv = plotting.plot_stat_map(bg_img=bg_img,
                                stat_map_img=classif_brain_nii_fn,
                                threshold=thresh,
                                cmap=cmap,
                                black_bg=True,
                                cut_coords=[-25, 13, 3])     # -25, 9, 3
    imu.set_fig_bg_contrast(pv)

    if fig_type == 'raster_graphics':
        ext_fname = '.png'
    else:
        ext_fname = '.svg'

    pv.savefig(classif_brain_nii_fn.split('.')[0] + '_weights' + ext_fname,
               dpi=300)

