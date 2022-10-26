#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import cmasher as cmr
import pandas as pd
import paths
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX  = paths.data / 'HETDEX_for_prediction.h5'
train_test_idx    = np.loadtxt(paths.data / 'indices_train_test.txt')
calibration_idx   = np.loadtxt(paths.data / 'indices_calibration.txt')
validation_idx    = np.loadtxt(paths.data / 'indices_validation.txt')

feats_2_use       = ['ID', 'class', 'LOFAR_detect', 'Score_radio']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
filter_AGN        = np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
filter_train_test  = catalog_HETDEX_df.index.isin(train_test_idx)
filter_calibration = catalog_HETDEX_df.index.isin(calibration_idx)
filter_validation  = catalog_HETDEX_df.index.isin(validation_idx)
# catalog_HETDEX_df = catalog_HETDEX_df.loc[filter_AGN]

fract_positiv_train_test, mean_pred_val_train_test =\
                        calibration_curve(catalog_HETDEX_df.loc[filter_AGN * filter_train_test, 'LOFAR_detect'],\
                                          catalog_HETDEX_df.loc[filter_AGN * filter_train_test, 'Score_radio'],\
                                          n_bins=30, normalize=False, strategy='quantile')

fract_positiv_calib, mean_pred_val_calib =\
                        calibration_curve(catalog_HETDEX_df.loc[filter_AGN * filter_calibration, 'LOFAR_detect'],\
                                          catalog_HETDEX_df.loc[filter_AGN * filter_calibration, 'Score_radio'],\
                                          n_bins=30, normalize=False, strategy='quantile')

fract_positiv_val, mean_pred_val_val =\
                        calibration_curve(catalog_HETDEX_df.loc[filter_AGN * filter_validation, 'LOFAR_detect'],\
                                          catalog_HETDEX_df.loc[filter_AGN * filter_validation, 'Score_radio'],\
                                          n_bins=30, normalize=False, strategy='quantile')

fig             = plt.figure(figsize=(7,6))
ax1             = fig.add_subplot(111)

min_x = np.nanmin([np.nanmin(catalog_HETDEX_df.loc[filter_train_test,  'Score_radio']),\
                   np.nanmin(catalog_HETDEX_df.loc[filter_calibration, 'Score_radio']),\
                   np.nanmin(catalog_HETDEX_df.loc[filter_validation,  'Score_radio'])])
max_x = np.nanmax([np.nanmax(catalog_HETDEX_df.loc[filter_train_test,  'Score_radio']),\
                   np.nanmax(catalog_HETDEX_df.loc[filter_calibration, 'Score_radio']),\
                   np.nanmax(catalog_HETDEX_df.loc[filter_validation,  'Score_radio'])])

ax1.plot(mean_pred_val_train_test, fract_positiv_train_test, ls='-',\
         marker='o', c=plt.get_cmap(gv.cmap_conf_matr)(1.0), lw=2.5, label='Train+test')
ax1.plot(mean_pred_val_calib, fract_positiv_calib, ls='-', marker='s',\
         c=plt.get_cmap(gv.cmap_conf_matr)(0.6), lw=2.5, label='Calibration')
ax1.plot(mean_pred_val_val, fract_positiv_val, ls='-', marker='p',\
         c=plt.get_cmap(gv.cmap_conf_matr)(0.3), lw=2.5, label='Validation')
ax1.plot([0, 1], [0, 1], ls=':', c='k', label="Perfectly calibrated")
ax1.set_xlabel('Predicted score', fontsize=20)
ax1.set_ylabel('Fraction of positives', fontsize=20)
ax1.tick_params(which='both', top=True, right=True, direction='in')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(which='major', length=8, width=1.5)
ax1.tick_params(which='minor', length=4, width=1.5)
plt.setp(ax1.spines.values(), linewidth=2.5)
plt.setp(ax1.spines.values(), linewidth=2.5)
ax1.set_xlim(left=min_x * 0.99999, right=max_x * 1.00001)
plt.legend(loc='best', fontsize=14, title='Sub-sets', title_fontsize=14)
fig.tight_layout()
plt.savefig(paths.figures / 'calib_curves_pre_calib_radio.pdf', bbox_inches='tight')
