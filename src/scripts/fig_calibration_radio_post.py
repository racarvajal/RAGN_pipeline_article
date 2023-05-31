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

file_name_HETDEX     = paths.data / 'HETDEX_for_prediction.parquet'
train_validation_idx = np.loadtxt(paths.data / 'indices_train_validation.txt')
calibration_idx      = np.loadtxt(paths.data / 'indices_calibration.txt')
test_idx             = np.loadtxt(paths.data / 'indices_test.txt')

feats_2_use        = ['ID', 'class', 'LOFAR_detect', 'Prob_radio']

catalog_HETDEX_df  = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)
catalog_HETDEX_df  = catalog_HETDEX_df.set_index(keys=['ID'])
filter_AGN         = np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
filter_train_val   = catalog_HETDEX_df.index.isin(train_validation_idx)
filter_calibration = catalog_HETDEX_df.index.isin(calibration_idx)
filter_test        = catalog_HETDEX_df.index.isin(test_idx)

fract_positiv_train_validation, mean_pred_val_train_validation =\
                        calibration_curve(catalog_HETDEX_df.loc[filter_AGN * filter_train_val, 'LOFAR_detect'], 
                                          catalog_HETDEX_df.loc[filter_AGN * filter_train_val, 'Prob_radio'],
                                          n_bins=30, normalize=False, strategy='quantile')

fract_positiv_calib, mean_pred_val_calib =\
                        calibration_curve(catalog_HETDEX_df.loc[filter_AGN * filter_calibration, 'LOFAR_detect'],
                                          catalog_HETDEX_df.loc[filter_AGN * filter_calibration, 'Prob_radio'],
                                          n_bins=30, normalize=False, strategy='quantile')

fract_positiv_test, mean_pred_val_test =\
                        calibration_curve(catalog_HETDEX_df.loc[filter_AGN * filter_test, 'LOFAR_detect'],
                                          catalog_HETDEX_df.loc[filter_AGN * filter_test, 'Prob_radio'],
                                          n_bins=30, normalize=False, strategy='quantile')

fig             = plt.figure(figsize=(7,6))
ax1             = fig.add_subplot(111)

ax1.plot(mean_pred_val_train_validation, fract_positiv_train_validation, ls='-',
         marker='o', c=plt.get_cmap(gv.cmap_conf_matr)(1.0), lw=2.5, 
         label='Train+\nvalidation')
ax1.plot(mean_pred_val_calib, fract_positiv_calib, ls='-', marker='s',
         c=plt.get_cmap(gv.cmap_conf_matr)(0.6), lw=2.5, label='Calibration')
ax1.plot(mean_pred_val_test, fract_positiv_test, ls='-', marker='p',
         c=plt.get_cmap(gv.cmap_conf_matr)(0.3), lw=2.5, label='Test')
ax1.plot([0, 1], [0, 1], ls=':', c='k', label="Perfectly\ncalibrated")
ax1.set_xlabel('Predicted score', fontsize=36)
ax1.set_ylabel('Fraction of positives', fontsize=36)
ax1.tick_params(which='both', top=True, right=True, direction='in')
ax1.tick_params(axis='both', which='major', labelsize=30)
ax1.tick_params(which='major', length=8, width=1.5)
ax1.tick_params(which='minor', length=4, width=1.5)
plt.setp(ax1.spines.values(), linewidth=2.5)
plt.setp(ax1.spines.values(), linewidth=2.5)
plt.legend(loc='best', fontsize=25, title='Sub-sets', title_fontsize=25, 
           ncol=2, columnspacing=.25, handlelength=0.8, 
           handletextpad=0.1, framealpha=0.55)
fig.tight_layout()
plt.savefig(paths.figures / 'calib_curves_post_calib_radio.pdf', bbox_inches='tight')
