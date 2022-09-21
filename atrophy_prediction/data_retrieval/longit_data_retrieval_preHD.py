#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:12:28 2020

@author: Eduardo Castro

Code used to retrieve subcortical volumes from good quality images from the
TRACK-HD study (QA determined by FreeSurfer's QA_tools). Data from a subset of
9 subcortical regions, which are used for HD progression detection, are also
used for this analysis. However, this analysis is restricted to 3 brain
regions only: thalamus, caudate and putamen and volumes are scaled by their
respective ICV (ratio normalization method).
"""

import pandas as pd
import numpy as np
from os.path import join as opj


"""
INPUT VARIABLES
"""
study_id = 'TRACK'
whole_vol_col = 'EstimatedTotalIntraCranialVol'
relev_areas = ['Thalamus-Proper', 'Caudate', 'Putamen']
complem_vars = ['subjid', 'visit', 'group', 'CAG']

"""
ANALYSIS SCRIPT
"""

if __name__ == '__main__':

    # Define source and destination directories and files
    analysis_dir = opj('/data2/eduardo/results/HD_n_Controls',
                       'ROI-feats/slopes_longit', study_id, 'analysis')
    store_dir = opj('/data2/eduardo/results/HD_n_Controls/ROI-feats/'
                    'vol_prediction', study_id)
    aseg_demog_fn = opj(analysis_dir, 'vols_demog_raw.csv')

    # Load good QA volumetric data; preserve controls and pre-HD only
    vols_demog_df = pd.read_csv(aseg_demog_fn)
    vols_demog_df = vols_demog_df[vols_demog_df.group != 'earlyHD']
    vols_demog_df.reset_index(drop=True, inplace=True)

    # Get rid of remaining single-visit subjects due to transition to early-HD
    visit_df = vols_demog_df.groupby('subjid').count().loc[:, 'visit']
    visit_df = visit_df[visit_df > 1]
    subj_ids = visit_df.index
    vols_demog_df = vols_demog_df[vols_demog_df['subjid'].isin(subj_ids)]

    # Keep only variables of interest
    vars_interest = complem_vars + relev_areas + [whole_vol_col]
    vols_demog_df = vols_demog_df[vars_interest]
    vols_demog_df.rename(columns={'Thalamus-Proper': 'Thalamus'}, inplace=True)
    relev_areas = ['Thalamus' if 'Thal' in val else val for val in relev_areas]

    # Represent subcortical volumes as percentage of ICV
    vols_demog_df[relev_areas] =\
        vols_demog_df[relev_areas].div(vols_demog_df[whole_vol_col],
                                       axis='index')*100
    del vols_demog_df[whole_vol_col]
    
    # Adjust CAG repeats for controls; use subjects with 7 visits for training
    vols_demog_df['pred_use'] = np.nan
    vols_demog_df.loc[vols_demog_df.group == 'control', 'CAG'] = np.nan
    visit_df = vols_demog_df.groupby('subjid')['visit'].count()
    train_subjid = np.unique(visit_df[visit_df == 7].index.values)
    vols_demog_df.loc[vols_demog_df.subjid.isin(train_subjid), 'pred_use']\
        = 'train'
    vols_demog_df.loc[~vols_demog_df.subjid.isin(train_subjid), 'pred_use']\
        = 'test'

    # Assign 'follow-up visit' index (values relative to first recorded visit)
    idx_base = vols_demog_df.groupby('subjid')['visit'].transform('idxmin')
    vols_demog_df['follup_visit'] = vols_demog_df.loc[idx_base, 'visit'].values
    vols_demog_df['follup_visit'] = vols_demog_df['visit'] -\
        vols_demog_df['follup_visit']
    vols_demog_df.to_csv(opj(store_dir, 'vols_longit_norm.csv'), index=False)
