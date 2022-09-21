# Code repository for Huntington's disease subcortical volumes' trajectories analysis

## Description

This repository includes the code used to perform the analyses listed in our Movement Disorders paper entitled _Predictive Modeling of Huntington's Disease Unfolds Thalamic and Caudate Atrophy Dissociation_ by authors Eduardo Castro, Pablo Polosecki, Dorian Pustina, Andrew Wood, Cristina Sampaio, and Guillermo Cecchi.

This is not a toolbox, but a set of scripts provided _as is_ for reproducibility and transparency purposes. All of the scripts were run with Python 2.7, but they should work on Python 3. The code is mostly self-contained, the only dependency out of this repo being Pyflux, a Bayesian inference library for time series analysis that I modified to meet our project needs (more information in the _Dependencies_ section).

There are 4 directories in this repo. Some of them have subdirectories to make the code better organized, their names being self-explanatory. Here we briefly describe the contents of each directory:
* **atrophy_prediction**: Scripts to perform prospective atrophy prediction of subcortical regions, including data retrieval, evaluation and plot generation.
* **progression_detection**: Scripts to perform progression detection on Huntington's disease individuals before clinical motor diagnosis (before-CMD), including data retrieval, classification and plot generation.
* **utility_code**: Set of various utility functions, designed for tasks such as statistical testing, resampling and classification weights retrieval, among others.
* **classification_core**: Set of classification functions used by the classification tasks defined for before-CMD progression detection.

## Dependencies

All the code runs in Python (tested in version 2.7) and requires the following libraries: scikit-learn, scipy, numpy, mne, nibabel, statsmodels, matplotlib, seaborn and pyflux, a Bayesian inference library for time series analysis. Pyflux was developed by Ross Taylor, but modified by me to work with this code. You can find a copy of this package in my GitHub account.

## References

* Castro E., et al. (in press). Predictive Modeling of Huntington's Disease Unfolds Thalamic and Caudate Atrophy Dissociation. Movement Disorders
* Taylor R. PyFlux: An open source time series library for Python. 2017. Available from: https://pyflux.readthedocs.io/  
