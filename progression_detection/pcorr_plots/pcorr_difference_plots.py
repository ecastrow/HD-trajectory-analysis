#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:08:51 2017

Plot partial correlations using a connectivity plot for controls, pre-HD and
thei difference

@author: Eduardo Castro
"""

from sklearn.covariance import LedoitWolf
from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from mne.viz import circular_layout, plot_connectivity_circle

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 18

"""
INPUT PARAMS
"""

# General setup variables
regions = ['Lateral-Ventricle',
           'Brain-Stem',
           'Hippocampus',
           'Amygdala',
           'Accumbens-area',
           'Thalamus-Proper',
           'Caudate',
           'Putamen',
           'Pallidum']

fig_type = 'vector_graphics'    # raster_graphics
save_plots = False
pcorrs = True
age_covariate = False
study_id = 'TRACK'

# Setup for circular connectivity plot
node_colors = plt.cm.Set3(np.linspace(0, 1, 12))
node_col_tup = [tuple(nc[:3]) for nc in node_colors]

regions_cm_sorted = ['Lateral-Ventricle', 'Brain-Stem', 'Hippocampus',
                     'Amygdala', 'Thalamus-Proper', 'Accumbens-area',
                     'Pallidum', 'Caudate', 'Putamen']

analysis_dir = opj('/data2/eduardo/results/HD_n_Controls',
                   'ROI-feats/slopes_longit/', study_id, 'analysis')
in_csv = opj(analysis_dir, 'vols_slopes_' + study_id + '.csv')
group_var = 'group'
groups = ['preHD', 'control']
groups_plot_labels = {'preHD': 'Pre-HD', 'control': 'Control'}

in_circ_csv = opj(analysis_dir, 'final_precision_for_plot.csv')
fig_dir = analysis_dir
plt.close('all')

# Titles for figures
title_connect = ' vs '.join([groups_plot_labels[group]
                             for group in groups])
title_diff = ' - '.join([groups_plot_labels[group]
                         for group in groups])


"""
MAIN FUNCTION
"""

# Load vol dataframe
vol_df = pd.read_csv(in_csv)

# z-score regions with both groups
vol_df = vol_df[regions + [group_var]]
vol_df[regions] = vol_df[regions].apply(zscore)

# Preallocate dataframe for pcorr difference among groups
diff_df = vol_df[regions].cov().copy()
diff_df[regions] = 0

# Estimate difference between partial correlation matrices of both groups
max_val = -5
pcorr_dict = {}
for gindx, group_used in enumerate(groups):
    # Retrieve data from each group separately
    vol_df_group = vol_df.loc[vol_df[group_var] == group_used, regions]
    prec_df = vol_df_group.cov().copy()

    # l2-norm regularized MLE of the covariance with LedoitWolf method
    lw = LedoitWolf()
    lw.fit(vol_df_group)
    cov_fitted = lw.covariance_
    prec_fitted = lw.precision_
    prec_df[prec_df.columns] = prec_fitted

    # Estimate partial correlations (formula from Wikipedia)
    pcorr_df = prec_df.copy()
    if pcorrs:
        pcorr_df[pcorr_df.columns] = 0
        xv, yv = np.meshgrid(np.diagonal(prec_df), np.diagonal(prec_df))
        pcorr_df[pcorr_df.columns] = -prec_df / np.sqrt(xv * yv)
    else:
        pcorr_df[pcorr_df.columns] = lw.covariance_

    pcorr_dict[group_used] = pcorr_df

# Generate pcorr/covariance matrix differences among groups
diff_df = pcorr_dict[groups[0]] - pcorr_dict[groups[1]]
pcorr_dict['diff'] = diff_df


# Circular connectivity diagram based on matrix output by bootstrap analysis
circ_mask = pd.read_csv(in_circ_csv)
circ_mask.set_index('Unnamed: 0', inplace=True)
circ_mask.index.rename('', inplace=True)
circ_mask = circ_mask[circ_mask != 0].isnull()

node_angles = circular_layout(regions, regions_cm_sorted,
                              start_pos=90,
                              group_boundaries=[0, 5])
vabs = max_val

for group in pcorr_dict.keys():
    fh = plt.figure(figsize=(8, 8), facecolor='black')
    circ_vals = pcorr_dict[group]

    if age_covariate:
        circ_vals[np.setdiff1d(circ_vals.columns, 'age')] = 0
        circ_vals.loc['age'] = circ_vals['age']
    else:
        circ_vals[circ_mask] = 0

    n_lines = sum(circ_vals.values.ravel() != 0)/2
    plot_connectivity_circle(con=circ_vals.values, node_names=regions,
                             node_angles=node_angles, facecolor='black',
                             node_colors=node_col_tup, fontsize_title=16,
                             title=group,
                             fontsize_colorbar=10,
                             fontsize_names=12, linewidth=3, n_lines=n_lines,
                             fig=fh, vmin=-vabs, vmax=vabs,
                             colormap='bwr')

    if fig_type == 'raster_graphics':
        fig_fname = '_'.join(['circ', group, 'pcorr.png'])
    else:
        fig_fname = '_'.join(['circ', group, 'pcorr.svg'])
    if save_plots:
        fh.savefig(opj(fig_dir, fig_fname),
                   facecolor='black')
