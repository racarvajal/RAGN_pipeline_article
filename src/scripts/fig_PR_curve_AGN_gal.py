#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc
import colorcet as cc
import pandas as pd
import paths

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.h5'
train_idx        = np.loadtxt(paths.data / 'indices_train.txt')
test_idx         = np.loadtxt(paths.data / 'indices_test.txt')
train_test_idx   = np.loadtxt(paths.data / 'indices_train_test.txt')
calibration_idx  = np.loadtxt(paths.data / 'indices_calibration.txt')
validation_idx   = np.loadtxt(paths.data / 'indices_validation.txt')

feats_2_use      = ['ID', 'class', 'Prob_AGN']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

prec_train_cal,      recall_train_cal,      thresh_train_cal      =\
                     precision_recall_curve(catalog_HETDEX_df.loc[train_idx, 'class'],
                                            catalog_HETDEX_df.loc[train_idx, 'Prob_AGN'],
                                            pos_label=1)
prec_test_cal,       recall_test_cal,       thresh_test_cal       =\
                     precision_recall_curve(catalog_HETDEX_df.loc[test_idx, 'class'],
                                            catalog_HETDEX_df.loc[test_idx, 'Prob_AGN'],
                                            pos_label=1)
prec_train_test_cal, recall_train_test_cal, thresh_train_test_cal =\
                     precision_recall_curve(catalog_HETDEX_df.loc[train_test_idx, 'class'],
                                            catalog_HETDEX_df.loc[train_test_idx, 'Prob_AGN'],
                                            pos_label=1)
prec_calib_cal,      recall_calib_cal,      thresh_calib_cal      =\
                     precision_recall_curve(catalog_HETDEX_df.loc[calibration_idx, 'class'],
                                            catalog_HETDEX_df.loc[calibration_idx, 'Prob_AGN'],
                                            pos_label=1)
prec_validation_cal, recall_validation_cal, thresh_validation_cal =\
                     precision_recall_curve(catalog_HETDEX_df.loc[validation_idx, 'class'],
                                            catalog_HETDEX_df.loc[validation_idx, 'Prob_AGN'],
                                            pos_label=1)


auc_pr_train_cal      = auc(recall_train_cal,      prec_train_cal)
auc_pr_test_cal       = auc(recall_test_cal,       prec_test_cal)
auc_pr_train_test_cal = auc(recall_train_test_cal, prec_train_test_cal)
auc_pr_calib_cal      = auc(recall_calib_cal,      prec_calib_cal)
auc_pr_validation_cal = auc(recall_validation_cal, prec_validation_cal)


fig             = plt.figure(figsize=(7,6))
ax1             = fig.add_subplot(111)

viz_train       = PrecisionRecallDisplay(precision=prec_train_cal,      recall=recall_train_cal)
viz_test        = PrecisionRecallDisplay(precision=prec_test_cal,       recall=recall_test_cal)
viz_train_test  = PrecisionRecallDisplay(precision=prec_train_test_cal, recall=recall_train_test_cal)
viz_calib       = PrecisionRecallDisplay(precision=prec_calib_cal,      recall=recall_calib_cal)
viz_val         = PrecisionRecallDisplay(precision=prec_validation_cal, recall=recall_validation_cal)
viz_train.plot(ax=ax1,       lw=3.5, c=cm.get_cmap('cet_CET_C2s')(0.0), alpha=0.6, label=f"Training    (AUC = {auc_pr_train_cal:.4f})")
viz_test.plot(ax=ax1,        lw=3.5, c=cm.get_cmap('cet_CET_C2s')(0.3), alpha=0.6, label=f"Test        (AUC = {auc_pr_test_cal:.4f})")
viz_train_test.plot(ax=ax1,  lw=3.5, c=cm.get_cmap('cet_CET_C2s')(0.5), alpha=0.6, label=f"Train+Test  (AUC = {auc_pr_train_test_cal:.4f})")
viz_calib.plot(ax=ax1,       lw=3.5, c=cm.get_cmap('cet_CET_C2s')(0.7), alpha=0.6, label=f"Calibration (AUC = {auc_pr_calib_cal:.4f})")
viz_val.plot(ax=ax1,         lw=3.5, c=cm.get_cmap('cet_CET_C2s')(0.9), alpha=0.6, label=f"Validation  (AUC = {auc_pr_validation_cal:.4f})")

no_skill_train      = np.sum(catalog_HETDEX_df.loc[train_idx, 'class'] == 1)       / len(catalog_HETDEX_df.loc[train_idx, 'class'])
no_skill_test       = np.sum(catalog_HETDEX_df.loc[test_idx, 'class'] == 1)        / len(catalog_HETDEX_df.loc[test_idx, 'class'])
no_skill_train_test = np.sum(catalog_HETDEX_df.loc[train_test_idx, 'class'] == 1)  / len(catalog_HETDEX_df.loc[train_test_idx, 'class'])
no_skill_calib      = np.sum(catalog_HETDEX_df.loc[calibration_idx, 'class'] == 1) / len(catalog_HETDEX_df.loc[calibration_idx, 'class'])
no_skill_val        = np.sum(catalog_HETDEX_df.loc[validation_idx, 'class'] == 1)  / len(catalog_HETDEX_df.loc[validation_idx , 'class'])
ax1.plot([0, 1], [no_skill_train,      no_skill_train],      ls='--', marker=None, c=cm.get_cmap('cet_CET_C2s')(0.0), alpha=0.8, lw=3.5)
ax1.plot([0, 1], [no_skill_test,       no_skill_test],       ls='--', marker=None, c=cm.get_cmap('cet_CET_C2s')(0.4), alpha=0.8, lw=3.5)
ax1.plot([0, 1], [no_skill_train_test, no_skill_train_test], ls='--', marker=None, c=cm.get_cmap('cet_CET_C2s')(0.6), alpha=0.8, lw=3.5)
ax1.plot([0, 1], [no_skill_calib,      no_skill_calib],      ls='--', marker=None, c=cm.get_cmap('cet_CET_C2s')(0.8), alpha=0.8, lw=3.5)
ax1.plot([0, 1], [no_skill_val,        no_skill_val],        ls='--', marker=None, c=cm.get_cmap('cet_CET_C2s')(1.0), alpha=0.8, lw=3.5)
ax1.plot([1, 1], [1, 1], ls='--', marker=None, c='Gray', alpha=0.7, lw=3.5, label='No Skill', zorder=0)

ax1.set_xlabel('Recall', fontsize=20)
ax1.set_ylabel('Precision', fontsize=20)
ax1.tick_params(which='both', top=True, right=True, direction='in')
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.tick_params(which='major', length=8, width=1.5)
ax1.tick_params(which='minor', length=4, width=1.5)
plt.setp(ax1.spines.values(), linewidth=3.0)
plt.setp(ax1.spines.values(), linewidth=3.0)
plt.legend(loc=3, fontsize=14, title='AGN/Galaxy classification', title_fontsize=14)
ax1.set_aspect('equal', 'datalim')
ax1.set_title('Calibrated PR curve. HETDEX field.', fontsize=16)
fig.tight_layout()
plt.savefig(paths.figures / 'PR_cal_curve_AGN_gal.pdf', bbox_inches='tight')
