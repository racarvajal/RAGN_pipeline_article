#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import cmasher as cmr
import pandas as pd
import paths
import global_variables as gv

def main():
    mpl.rcdefaults()
    plt.rcParams['text.usetex'] = gv.use_LaTeX

    file_name_HETDEX     = paths.data / 'HETDEX_for_prediction.parquet'
    train_validation_idx = np.loadtxt(paths.data / 'indices_train_validation.txt')
    calibration_idx      = np.loadtxt(paths.data / 'indices_calibration.txt')
    test_idx             = np.loadtxt(paths.data / 'indices_test.txt')

    feats_2_use      = ['ID', 'class', 'Score_AGN']

    catalog_HETDEX_df = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)
    catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

    fract_positiv_train_validation, mean_pred_val_train_validation =\
                            calibration_curve(catalog_HETDEX_df.loc[train_validation_idx, 'class'],
                                              catalog_HETDEX_df.loc[train_validation_idx, 'Score_AGN'],
                                              n_bins=30, normalize=False, strategy='quantile')

    fract_positiv_calib, mean_pred_val_calib =\
                            calibration_curve(catalog_HETDEX_df.loc[calibration_idx, 'class'],
                                              catalog_HETDEX_df.loc[calibration_idx, 'Score_AGN'],
                                              n_bins=30, normalize=False, strategy='quantile')

    fract_positiv_test, mean_pred_val_test =\
                            calibration_curve(catalog_HETDEX_df.loc[test_idx, 'class'],
                                              catalog_HETDEX_df.loc[test_idx, 'Score_AGN'],
                                              n_bins=30, normalize=False, strategy='quantile')

    fig             = plt.figure(figsize=(7,6))
    ax1             = fig.add_subplot(111)

    ax1.plot(mean_pred_val_train_validation, fract_positiv_train_validation, ls='-',
             marker='o', c=plt.get_cmap(gv.cmap_bands)(0.5), lw=2.5, 
             label=r'$\mathrm{Train} + $' + '\n' r'$\mathrm{validation}$')
    ax1.plot(mean_pred_val_calib, fract_positiv_calib, ls='-', marker='s',
             c=plt.get_cmap(gv.cmap_bands)(0.7), lw=2.5, label=r'$\mathrm{Calibration}$')
    ax1.plot(mean_pred_val_test, fract_positiv_test, ls='-', marker='p',
             c=plt.get_cmap(gv.cmap_bands)(0.9), lw=2.5, label=r'$\mathrm{Test}$')
    ax1.plot([0, 1], [0, 1], ls=':', c='k', label=r'$\mathrm{Perfectly}$' + '\n' r'$\mathrm{calibrated}$')
    ax1.set_xlabel(r'$\mathrm{Predicted ~ score}$', fontsize=36)
    ax1.set_ylabel(r'$\mathrm{Fraction ~ of ~ positives}$', fontsize=36)
    ax1.tick_params(which='both', top=True, right=True, direction='in')
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.tick_params(which='major', length=8, width=1.5)
    ax1.tick_params(which='minor', length=4, width=1.5)
    plt.setp(ax1.spines.values(), linewidth=2.5)
    plt.setp(ax1.spines.values(), linewidth=2.5)
    ax1.set_xlim(left=0.49985, right=0.50015)
    plt.legend(loc='best', fontsize=25, title=r'$\mathrm{Sub-sets}$', title_fontsize=25, 
               ncol=2, columnspacing=.25, handlelength=0.8, 
               handletextpad=0.1, framealpha=0.55)
    fig.tight_layout()
    plt.savefig(paths.figures / 'calib_curves_pre_calib_AGN_galaxy.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()