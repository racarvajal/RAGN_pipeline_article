#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc
import cmasher as cmr
import global_variables as gv
import pandas as pd
import paths

def main():
    mpl.rcdefaults()
    plt.rcParams['text.usetex'] = gv.use_LaTeX

    file_name_HETDEX     = paths.data / 'HETDEX_for_prediction.parquet'
    train_idx            = np.loadtxt(paths.data / 'indices_train.txt')
    validation_idx       = np.loadtxt(paths.data / 'indices_validation.txt')
    train_validation_idx = np.loadtxt(paths.data / 'indices_train_validation.txt')
    calibration_idx      = np.loadtxt(paths.data / 'indices_calibration.txt')
    test_idx             = np.loadtxt(paths.data / 'indices_test.txt')

    feats_2_use      = ['ID', 'class', 'Prob_AGN']

    catalog_HETDEX_df = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)
    catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

    prec_train_cal,      recall_train_cal,      thresh_train_cal      =\
                         precision_recall_curve(catalog_HETDEX_df.loc[train_idx, 'class'],
                                                catalog_HETDEX_df.loc[train_idx, 'Prob_AGN'],
                                                pos_label=1)
    prec_validation_cal, recall_validation_cal, thresh_validation_cal =\
                         precision_recall_curve(catalog_HETDEX_df.loc[validation_idx, 'class'],
                                                catalog_HETDEX_df.loc[validation_idx, 'Prob_AGN'],
                                                pos_label=1)
    prec_train_val_cal, recall_train_val_cal, thresh_train_val_cal =\
                         precision_recall_curve(catalog_HETDEX_df.loc[train_validation_idx, 'class'],
                                                catalog_HETDEX_df.loc[train_validation_idx, 'Prob_AGN'],
                                                pos_label=1)
    prec_calib_cal,      recall_calib_cal,      thresh_calib_cal      =\
                         precision_recall_curve(catalog_HETDEX_df.loc[calibration_idx, 'class'],
                                                catalog_HETDEX_df.loc[calibration_idx, 'Prob_AGN'],
                                                pos_label=1)
    prec_test_cal,       recall_test_cal,       thresh_test_cal       =\
                         precision_recall_curve(catalog_HETDEX_df.loc[test_idx, 'class'],
                                                catalog_HETDEX_df.loc[test_idx, 'Prob_AGN'],
                                                pos_label=1)


    auc_pr_train_cal      = auc(recall_train_cal,      prec_train_cal)
    auc_pr_validation_cal = auc(recall_validation_cal, prec_validation_cal)
    auc_pr_train_val_cal  = auc(recall_train_val_cal,  prec_train_val_cal)
    auc_pr_calib_cal      = auc(recall_calib_cal,      prec_calib_cal)
    auc_pr_test_cal       = auc(recall_test_cal,       prec_test_cal)


    fig             = plt.figure(figsize=(7,6))
    ax1             = fig.add_subplot(111)

    viz_train       = PrecisionRecallDisplay(precision=prec_train_cal,      recall=recall_train_cal)
    viz_test        = PrecisionRecallDisplay(precision=prec_test_cal,       recall=recall_test_cal)
    viz_train_val   = PrecisionRecallDisplay(precision=prec_train_val_cal,  recall=recall_train_val_cal)
    viz_calib       = PrecisionRecallDisplay(precision=prec_calib_cal,      recall=recall_calib_cal)
    viz_val         = PrecisionRecallDisplay(precision=prec_validation_cal, recall=recall_validation_cal)
    viz_train.plot(ax=ax1,     lw=4.5, c=cm.get_cmap(gv.cmap_bands)(0.0), alpha=0.75, label=r'$\mathrm{Training}$')
    viz_val.plot(ax=ax1,       lw=4.5, c=cm.get_cmap(gv.cmap_bands)(0.3), alpha=0.75, label=r'$\mathrm{Validation}$')
    viz_train_val.plot(ax=ax1, lw=4.5, c=cm.get_cmap(gv.cmap_bands)(0.5), alpha=0.75, label=r'$\mathrm{Train} + $' + '\n' r'$\mathrm{validation}$')
    viz_calib.plot(ax=ax1,     lw=4.5, c=cm.get_cmap(gv.cmap_bands)(0.7), alpha=0.75, label=r'$\mathrm{Calibration}$')
    viz_test.plot(ax=ax1,      lw=4.5, c=cm.get_cmap(gv.cmap_bands)(0.9), alpha=0.75, label=r'$\mathrm{Test}$')

    no_skill_train     = np.sum(catalog_HETDEX_df.loc[train_idx, 'class'] == 1)            / len(catalog_HETDEX_df.loc[train_idx, 'class'])
    no_skill_test      = np.sum(catalog_HETDEX_df.loc[test_idx, 'class'] == 1)             / len(catalog_HETDEX_df.loc[test_idx, 'class'])
    no_skill_train_val = np.sum(catalog_HETDEX_df.loc[train_validation_idx, 'class'] == 1) / len(catalog_HETDEX_df.loc[train_validation_idx, 'class'])
    no_skill_calib     = np.sum(catalog_HETDEX_df.loc[calibration_idx, 'class'] == 1)      / len(catalog_HETDEX_df.loc[calibration_idx, 'class'])
    no_skill_val       = np.sum(catalog_HETDEX_df.loc[validation_idx, 'class'] == 1)       / len(catalog_HETDEX_df.loc[validation_idx , 'class'])
    ax1.plot([0, 1], [no_skill_train,     no_skill_train],     ls='--', marker=None, c=cm.get_cmap(gv.cmap_bands)(0.0), alpha=0.4, lw=3.5)
    ax1.plot([0, 1], [no_skill_val,       no_skill_val],       ls='--', marker=None, c=cm.get_cmap(gv.cmap_bands)(0.4), alpha=0.4, lw=3.5)
    ax1.plot([0, 1], [no_skill_train_val, no_skill_train_val], ls='--', marker=None, c=cm.get_cmap(gv.cmap_bands)(0.6), alpha=0.4, lw=3.5)
    ax1.plot([0, 1], [no_skill_calib,     no_skill_calib],     ls='--', marker=None, c=cm.get_cmap(gv.cmap_bands)(0.8), alpha=0.4, lw=3.5)
    ax1.plot([0, 1], [no_skill_test,      no_skill_test],      ls='--', marker=None, c=cm.get_cmap(gv.cmap_bands)(1.0), alpha=0.4, lw=3.5)
    ax1.plot([1, 1], [1, 1], ls=':', marker=None, c='Gray', alpha=0.9, lw=4.5, label=r'$\mathrm{No ~ Skill}$', zorder=0)

    ax1.set_xlabel(r'$\mathrm{Recall}$', fontsize=36)
    ax1.set_ylabel(r'$\mathrm{Precision}$', fontsize=36)
    ax1.tick_params(which='both', top=True, right=True, direction='in')
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.tick_params(which='major', length=8, width=1.5)
    ax1.tick_params(which='minor', length=4, width=1.5)
    plt.setp(ax1.spines.values(), linewidth=3.0)
    plt.setp(ax1.spines.values(), linewidth=3.0)
    plt.legend(loc=6, fontsize=24, title=r'$\mathrm{AGN/Galaxy}$' + '\n' + r'$\mathrm{classification}$', 
               title_fontsize=24, ncol=2, columnspacing=.25, handlelength=0.8, 
               handletextpad=0.1, framealpha=0.55)
    ax1.set_aspect('equal', 'datalim')
    # ax1.set_title('Calibrated PR curve. HETDEX field.', fontsize=16)
    fig.tight_layout()
    plt.savefig(paths.figures / 'PR_cal_curve_AGN_gal.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()