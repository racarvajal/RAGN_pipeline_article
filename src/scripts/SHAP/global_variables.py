#!/usr/bin/env python

# File with most used
# variables in this project.
# File paths, file names, etc.

# Fields properties
# Areas (deg2)
area_HETDEX         = 424
area_S82            = 92
area_COSMOS         = 4  # Not real value. Placeholder

# Model names with train, test, calibration, and validation sub-sets
# Stacked models
star_model         = 'classification_star_no_star_ago_29_2022'
AGN_gal_model      = 'classification_AGN_galaxy_dec_13_2022'
radio_model        = 'classification_LOFAR_detect_dec_14_2022'
full_z_model       = 'regression_z_dec_15_2022'
# Calibrated models
cal_str_model      = 'cal_classification_star_no_star_ago_29_2022.joblib'
cal_AGN_gal_model  = 'cal_classification_AGN_galaxy_ago_30_2022.joblib'
cal_radio_model    = 'cal_classification_LOFAR_detect_sep_07_2022.joblib'

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
AGN_thresh         = 0.4933696292
radio_thresh       = 0.1253210681
# Calibrated and PR-optimised models (with calibration sub-set)
cal_str_thresh     = 0.6007345636412931
cal_AGN_thresh     = 0.40655238948425537
cal_radio_thresh   = 0.20125615854582377
# High redshift limit
high_z_limit       = 2.0  # 3.6

# Colours and colormaps
cmap_bands         = 'cmr.rainforest'
cmap_shap          = 'cmr.ember'  # cmr.pride, cet_CET_R3 cmr.wildfire cmr.guppy
cmap_conf_matr     = 'cmr.neutral_r'
cmap_z_plots       = 'cmr.fall_r'
cmap_dens_plots    = 'cmr.neutral_r'
cmap_hists         = 'cmr.fusion'