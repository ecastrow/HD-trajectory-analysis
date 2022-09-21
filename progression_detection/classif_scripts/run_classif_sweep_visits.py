#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script that runs classification models (training and prediction) using up to N
visits for ROI slope estimates (models use data from the TRACK study only)

Created on Tue Nov 26 15:27:29 2019

@author: Eduardo Castro
"""
from os.path import join as opj
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(opj(os.path.dirname(__file__), '..', '..')))
from classification_core.Main_script_runclassification import execute_pipeline


study_id = 'TRACK'
univ_area = 'Caudate'           # 'Striatum', 'Caudate'
univ_flag_vals = [False, True]         # [False, True]
restrict_visits_vals = range(2, 8)      # range(2, 8)

regions_subset = ['Lateral-Ventricle', 'Thalamus-Proper', 'Caudate',
                  'Putamen', 'Brain-Stem', 'Pallidum', 'Hippocampus',
                  'Amygdala', 'Accumbens-area']
fixed_regions = [univ_area]


for univ_flag in univ_flag_vals:
    for restrict_visits in restrict_visits_vals:

        target_label = 'group'
        covariate_detrend_params = None
        if univ_flag:
            list_fix_features = []
            suffix = univ_area
        else:
            list_fix_features = []
            suffix = 'All_Regions'

        kfold_type = 'dependent'
        group_label = 'siteid'
        del_cols = ['subjid', 'study_id']
        kfold_val = None

        analysis_dir = opj('/data2/eduardo/results/HD_n_Controls',
                           'ROI-feats/slopes_longit/', study_id,
                           'analysis')

        aseg_slopes_fn = opj(analysis_dir,
                             '_'.join(['vols_slopes', study_id,
                                       str(restrict_visits) +
                                       'visits' + '.csv']))
        aseg_slopes_classif_fn = opj(analysis_dir,
                                     '_'.join(['vols_slopes', study_id,
                                       str(restrict_visits) +
                                       'visits_classif' + '.csv']))

        # Generate complete and for-classification dataframes
        slopes_df = pd.read_csv(aseg_slopes_fn, index_col=0)
        del_cols = ['subjid', 'study_id']

        # Delete unwanted columns
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
                         target_label=target_label,
                         covariate_detrend_params=covariate_detrend_params)
