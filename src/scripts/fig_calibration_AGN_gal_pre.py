#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import colorcet as cc
import pandas as pd
import paths
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.h5'
train_test_idx   = np.loadtxt(paths.data / 'indices_train_test.txt')
calibration_idx  = np.loadtxt(paths.data / 'indices_calibration.txt')
validation_idx   = np.loadtxt(paths.data / 'indices_validation.txt')

feats_2_use      = ['ID', 'class', 'Score_AGN']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

fract_positiv_train_test, mean_pred_val_train_test =\
                        calibration_curve(catalog_HETDEX_df.loc[train_test_idx, 'class'],\
                                          catalog_HETDEX_df.loc[train_test_idx, 'Score_AGN'],\
                                          n_bins=30, normalize=False, strategy='quantile')

fract_positiv_calib, mean_pred_val_calib =\
                        calibration_curve(catalog_HETDEX_df.loc[calibration_idx, 'class'],\
                                          catalog_HETDEX_df.loc[calibration_idx, 'Score_AGN'],\
                                          n_bins=30, normalize=False, strategy='quantile')

fract_positiv_val, mean_pred_val_val =\
                        calibration_curve(catalog_HETDEX_df.loc[validation_idx, 'class'],\
                                          catalog_HETDEX_df.loc[validation_idx, 'Score_AGN'],\
                                          n_bins=30, normalize=False, strategy='quantile')

fig             = plt.figure(figsize=(7,6))
ax1             = fig.add_subplot(111)

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
plt.legend(loc='best', fontsize=14, title='Sub-sets', title_fontsize=14)
fig.tight_layout()
plt.savefig(paths.figures / 'calib_curves_pre_calib_AGN_galaxy.pdf', bbox_inches='tight')
