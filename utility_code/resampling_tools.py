#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:14:50 2019

@author: Eduardo Castro
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample


def gen_resample_indexes(idx_orig, replace=True, random_state=0):
    """
    Generate random indexes from data to be sampled. By default, it generates
    samples with replacement (bootstrap samples). If sampling is not done with
    replacement, then random permutations are drawn.
    """
    idx_resamp = resample(idx_orig, replace=replace, random_state=random_state)

    return idx_resamp


def gen_conf_interval(resamp_values, obs_value, alpha=None):
    """
    Estimates the percentile (and sign) associated to a given observation under
    the distribution of resampled values (null hypothesis). If provided with a
    confidence level, it also generates a confidence interval for the provided
    resampled values.
    """
    # Center distrubution of values in zero prior to estimation of percentile
    median_val = np.median(resamp_values)
    resamp_dm = resamp_values - median_val
    obs_dm = obs_value - median_val

    # Estimate percentile and sign (side of the distribution were the obs is)
    if obs_dm > 0:
        obs_perc = sum(resamp_dm > obs_dm)/float(len(resamp_values))
    else:
        obs_perc = -sum(resamp_dm < obs_dm)/float(len(resamp_values))

    # Minimum value of percentile set to number of samples (avoid 0 assignment)
    if obs_perc == 0:
        obs_perc = 1./len(resamp_values)

    # Estimate confidence interval of null distribution if conf level provided
    if alpha is not None:
        min_ci_val = np.percentile(resamp_values, alpha*100)
        max_ci_val = np.percentile(resamp_values, 100 - alpha*100)
    else:
        min_ci_val = np.nan
        max_ci_val = np.nan

    return obs_perc, [min_ci_val, max_ci_val]
