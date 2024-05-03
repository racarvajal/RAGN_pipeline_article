#!/usr/bin/env python

# File with most used
# variables in this project.
# File paths, file names, etc.

# Use (external) LaTeX in plots
use_LaTeX           = True

# Fields properties
# Areas (deg2)
area_HETDEX         = 424
area_S82            = 92

# Model names with train, test, calibration, and validation sub-sets
# Stacked models
star_model         = 'classification_star_no_star_ago_29_2022'
AGN_gal_model      = 'classification_AGN_galaxy_dec_18_2022'
radio_model        = 'classification_LOFAR_detect_dec_19_2022'
full_z_model       = 'regression_z_dec_20_2022'
# Calibrated models
cal_str_model      = 'cal_classification_star_no_star_ago_29_2022.joblib'
cal_AGN_gal_model  = 'cal_classification_AGN_galaxy_dec_18_2022.joblib'
cal_radio_model    = 'cal_classification_LOFAR_detect_dec_19_2022.joblib'

# Seeds
seed               = 42

# Thresholds
# Beta for beta-scores
beta_F             = 1.1  # beta positive real value
# Naive values
naive_star_thresh  = 0.5
naive_AGN_thresh   = 0.5
naive_radio_thresh = 0.5
# Values obtained with train, test, calibration, and validation sub-sets
# PR-optimised models (with train+test sub-set)
star_thresh        = 0.1873511777
AGN_thresh         = 0.5000115951
radio_thresh       = 0.9815369877
# Calibrated and PR-optimised models (with calibration sub-set)
cal_str_thresh     = 0.6007345636412931
cal_AGN_thresh     = 0.34895396724527294
cal_radio_thresh   = 0.2046047064139296
# High redshift limit
high_z_limit       = 2.0  # 3.6

# Colours and colormaps
cmap_bands         = 'cmr.rainforest'
cmap_shap          = 'cmr.ember'  # cmr.pride, cet_CET_R3 cmr.wildfire cmr.guppy
cmap_conf_matr     = 'cmr.neutral_r'
cmap_z_plots       = 'cmr.fall_r'
cmap_dens_plots    = 'cmr.neutral_r'
cmap_hists         = 'cmr.fusion'
