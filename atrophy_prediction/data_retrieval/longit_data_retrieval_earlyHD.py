#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:42:17 2020

Code used to retrieve subcortical volumes for 3 brain regions (thalamus,
caudate and putamen) for healthy controls and pre-HD subjects that have 7
visits for training purposes, and early HD patients for test.

@author: Eduardo Castro
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

    # Load good QA volumetric data; preserve all three groups
    vols_demog_df = pd.read_csv(aseg_demog_fn)
    vols_demog_df.reset_index(drop=True, inplace=True)

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
    # Optional: Check mean values of each structure across groups
#    vols_demog_df.groupby('group')[relev_areas].mean()

    # Adjust CAG repeats for controls; use subjects with 7 visits for training
    vols_demog_df['pred_use'] = np.nan
    vols_demog_df.loc[vols_demog_df.group == 'control', 'CAG'] = np.nan
    visit_df = vols_demog_df.groupby('subjid')['visit'].count()
    train_subjid = np.unique(visit_df[visit_df == 7].index.values)
    vols_demog_df.loc[vols_demog_df.subjid.isin(train_subjid), 'pred_use']\
        = 'train'
    vols_demog_df.loc[vols_demog_df.group == 'earlyHD', 'pred_use'] =\
        'test_manifest'
    vols_demog_df = vols_demog_df.dropna(subset=['pred_use'])

    # Assign 'follow-up visit' index (values relative to first recorded visit)
    idx_base = vols_demog_df.groupby('subjid')['visit'].transform('idxmin')
    vols_demog_df['follup_visit'] = vols_demog_df.loc[idx_base, 'visit'].values
    vols_demog_df['follup_visit'] = vols_demog_df['visit'] -\
        vols_demog_df['follup_visit']
    vols_demog_df.to_csv(opj(store_dir, 'vols_longit_earlyHD.csv'),
                         index=False)
