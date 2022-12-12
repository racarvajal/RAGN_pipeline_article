#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cmasher as cmr
import pandas as pd
import paths
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.h5'
file_name_S82    = paths.data / 'S82_for_prediction.h5'
test_idx         = np.loadtxt(paths.data / 'indices_test.txt')

feats_2_use      = ['ID', 'class', 'pred_prob_class',
                    'LOFAR_detect', 'pred_prob_radio', 'Z', 'pred_Z']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
catalog_HETDEX_df['is_AGN'] = np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)

catalog_S82_df    = pd.read_hdf(file_name_S82, key='df').loc[:, feats_2_use]

HETDEX_known_filter     = np.array(catalog_HETDEX_df.loc[:, 'class'] == 0)           |\
                          np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
S82_known_filter        = np.array(catalog_S82_df.loc[:, 'class'] == 0)              |\
                          np.array(catalog_S82_df.loc[:, 'class'] == 1)
HETDEX_true_rAGN_filter = np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)           &\
                          np.array(catalog_HETDEX_df.loc[:, 'LOFAR_detect'] == 1)
HETDEX_pred_rAGN_filter = np.array(catalog_HETDEX_df.loc[:, 'pred_prob_class'] == 1) &\
                          np.array(catalog_HETDEX_df.loc[:, 'pred_prob_radio'] == 1)
S82_true_rAGN_filter    = np.array(catalog_S82_df.loc[:, 'class'] == 1)              &\
                          np.array(catalog_S82_df.loc[:, 'LOFAR_detect'] == 1)
S82_pred_rAGN_filter    = np.array(catalog_S82_df.loc[:, 'pred_prob_class'] == 1)    &\
                          np.array(catalog_S82_df.loc[:, 'pred_prob_radio'] == 1)

fig             = plt.figure(figsize=(9,3.5))
ax1             = fig.add_subplot(111, xscale='linear', yscale='log')

min_z_HETDEX_pred = catalog_HETDEX_df.loc[HETDEX_pred_rAGN_filter * ~HETDEX_known_filter, 'pred_Z'].min()
max_z_HETDEX_pred = catalog_HETDEX_df.loc[HETDEX_pred_rAGN_filter * ~HETDEX_known_filter, 'pred_Z'].max()
min_z_HETDEX_true = catalog_HETDEX_df.loc[HETDEX_true_rAGN_filter * HETDEX_known_filter, 'Z'].min()
max_z_HETDEX_true = catalog_HETDEX_df.loc[HETDEX_true_rAGN_filter * HETDEX_known_filter, 'Z'].max()
min_z_S82_pred    = catalog_S82_df.loc[S82_pred_rAGN_filter * ~S82_known_filter, 'pred_Z'].min()
max_z_S82_pred    = catalog_S82_df.loc[S82_pred_rAGN_filter * ~S82_known_filter, 'pred_Z'].max()
min_z_S82_true    = catalog_S82_df.loc[S82_true_rAGN_filter * S82_known_filter, 'Z'].min()
max_z_S82_true    = catalog_S82_df.loc[S82_true_rAGN_filter * S82_known_filter, 'Z'].max()
full_z_bins       = np.linspace(np.nanmin([min_z_HETDEX_pred, min_z_HETDEX_true, min_z_S82_pred, min_z_S82_true]),\
                                np.nanmax([max_z_HETDEX_pred, max_z_HETDEX_true, max_z_S82_pred, max_z_S82_true]), 50)

counts_HETDEX_pred, edges_HETDEX_pred = np.histogram(catalog_HETDEX_df.loc[HETDEX_pred_rAGN_filter * ~HETDEX_known_filter, 'pred_Z'],
                                                     bins=full_z_bins)
counts_S82_pred,    edges_S82_pred    = np.histogram(catalog_S82_df.loc[S82_pred_rAGN_filter * ~S82_known_filter, 'pred_Z'],
                                                     bins=full_z_bins)
counts_HETDEX_true, edges_HETDEX_true = np.histogram(catalog_HETDEX_df.loc[HETDEX_true_rAGN_filter * HETDEX_known_filter, 'Z'],
                                                     bins=full_z_bins)
counts_S82_true,    edges_S82_true    = np.histogram(catalog_S82_df.loc[S82_true_rAGN_filter * S82_known_filter, 'Z'],
                                                     bins=full_z_bins)
colour_a    = list(plt.get_cmap(gv.cmap_hists)(0.3))
colour_b    = list(plt.get_cmap(gv.cmap_hists)(0.7))
colour_a[3] = 0.65
colour_b[3] = 0.65
colour_a    = tuple(colour_a)
colour_b    = tuple(colour_b)

ax1.stairs(counts_S82_pred / gv.area_S82, edges_S82_pred, fill=True, ec='k', lw=1.5,\
           fc=colour_b, label='S82')
ax1.stairs(counts_HETDEX_pred / gv.area_HETDEX, edges_HETDEX_pred, fill=True, ec='k', lw=1.5,\
           fc=colour_a, label='HETDEX')
ax1.stairs(counts_HETDEX_true / gv.area_HETDEX, edges_HETDEX_true, fill=True, ec='k', lw=1.5,\
           fc=colour_a, label='HETDEX', hatch='///')
ax1.stairs(counts_S82_true / gv.area_S82, edges_S82_true, fill=True, ec='k', lw=1.5,\
           fc=colour_b, label='S82', hatch='///')


HETDEX_patch       = mpatches.Patch(fc=plt.get_cmap(gv.cmap_hists)(0.3), ec='k', label='HETDEX', lw=2.0, alpha=0.65)
S82_patch          = mpatches.Patch(fc=plt.get_cmap(gv.cmap_hists)(0.7), ec='k', label='S82', lw=2.0, alpha=0.65)
pred_patch         = mpatches.Patch(fc='None', ec='k', label='Predicted', lw=2.0)
true_patch         = mpatches.Patch(fc='None', ec='k', label='True', hatch='///', lw=2.0)
ax1.legend(handles=[HETDEX_patch, S82_patch, pred_patch, true_patch], loc=1, fontsize=20, ncol=1,
           handletextpad=0.3, handlelength=1.0, columnspacing=0.5, framealpha=0.75)

ax1.tick_params(which='both', top=True, right=True, direction='in')
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.tick_params(which='major', length=8, width=1.5)
ax1.tick_params(which='minor', length=4, width=1.5)
ax1.set_xlabel('Redshift', size=28)
ax1.set_ylabel('Normalised\nfrequency [$\mathrm{deg}^{-2}$]', size=26)
plt.setp(ax1.spines.values(), linewidth=3.5)
plt.setp(ax1.spines.values(), linewidth=3.5)
fig.tight_layout()
plt.savefig(paths.figures / 'hist_pred_true_z_HETDEX_all_S82_nonimputed.pdf', bbox_inches='tight')
