#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorcet as cc
import pandas as pd
import paths
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.h5'
file_name_S82    = paths.data / 'S82_for_prediction.h5'
validation_idx   = np.loadtxt(paths.data / 'indices_validation.txt')

feats_2_use      = ['ID', 'class', 'pred_prob_class',
                    'pred_prob_radio', 'W4mag']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

catalog_S82_df    = pd.read_hdf(file_name_S82, key='df').loc[:, feats_2_use]

HETDEX_known_filter     = np.array(catalog_HETDEX_df.loc[:, 'class'] == 0) |\
                          np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
S82_known_filter        = np.array(catalog_S82_df.loc[:, 'class'] == 0)    |\
                          np.array(catalog_S82_df.loc[:, 'class'] == 1)

HETDEX_pred_rAGN_filter      = np.array(catalog_HETDEX_df.loc[:, 'pred_prob_class'] == 1) &\
                               np.array(catalog_HETDEX_df.loc[:, 'pred_prob_radio'] == 1)
HETDEX_pred_AGN_norad_filter = np.array(catalog_HETDEX_df.loc[:, 'pred_prob_class'] == 1) &\
                               np.array(catalog_HETDEX_df.loc[:, 'pred_prob_radio'] == 0)

S82_pred_rAGN_filter      = np.array(catalog_S82_df.loc[:, 'pred_prob_class'] == 1) &\
                            np.array(catalog_S82_df.loc[:, 'pred_prob_radio'] == 1)
S82_pred_AGN_norad_filter = np.array(catalog_S82_df.loc[:, 'pred_prob_class'] == 1) &\
                            np.array(catalog_S82_df.loc[:, 'pred_prob_radio'] == 0)

fig             = plt.figure(figsize=(9,3.5))
ax1             = fig.add_subplot(111, xscale='linear', yscale='log')

min_z_HETDEX_radio = catalog_HETDEX_df.loc[HETDEX_pred_rAGN_filter      * ~HETDEX_known_filter, 'W4mag'].min()
max_z_HETDEX_radio = catalog_HETDEX_df.loc[HETDEX_pred_rAGN_filter      * ~HETDEX_known_filter, 'W4mag'].max()
min_z_HETDEX_norad = catalog_HETDEX_df.loc[HETDEX_pred_AGN_norad_filter * ~HETDEX_known_filter, 'W4mag'].min()
max_z_HETDEX_norad = catalog_HETDEX_df.loc[HETDEX_pred_AGN_norad_filter * ~HETDEX_known_filter, 'W4mag'].max()
min_z_S82_radio    = catalog_S82_df.loc[S82_pred_rAGN_filter            * ~S82_known_filter, 'W4mag'].min()
max_z_S82_radio    = catalog_S82_df.loc[S82_pred_rAGN_filter            * ~S82_known_filter, 'W4mag'].max()
min_z_S82_norad    = catalog_S82_df.loc[S82_pred_AGN_norad_filter       * ~S82_known_filter, 'W4mag'].min()
max_z_S82_norad    = catalog_S82_df.loc[S82_pred_AGN_norad_filter       * ~S82_known_filter, 'W4mag'].max()
full_z_bins        = np.linspace(np.nanmin([min_z_HETDEX_radio, min_z_HETDEX_norad, min_z_S82_radio, min_z_S82_norad]),\
                                np.nanmax([max_z_HETDEX_radio, max_z_HETDEX_norad, max_z_S82_radio, max_z_S82_norad]), 50)

counts_HETDEX_radio, edges_HETDEX_radio = np.histogram(catalog_HETDEX_df.loc[HETDEX_pred_rAGN_filter * ~HETDEX_known_filter, 'W4mag'],
                                                       bins=full_z_bins)
counts_S82_radio,    edges_S82_radio    = np.histogram(catalog_S82_df.loc[S82_pred_rAGN_filter * ~S82_known_filter, 'W4mag'],
                                                       bins=full_z_bins)
counts_HETDEX_norad, edges_HETDEX_norad = np.histogram(catalog_HETDEX_df.loc[HETDEX_pred_AGN_norad_filter * ~HETDEX_known_filter, 'W4mag'],
                                                       bins=full_z_bins)
counts_S82_norad,    edges_S82_norad    = np.histogram(catalog_S82_df.loc[S82_pred_AGN_norad_filter * ~S82_known_filter, 'W4mag'],
                                                       bins=full_z_bins)

ax1.stairs(counts_HETDEX_radio / gv.area_HETDEX, edges_HETDEX_radio, fill=True, ec='k', lw=3.5,\
           fc=plt.get_cmap('cet_bmy',3)(0.0), label='HETDEX', alpha=0.65)
ax1.stairs(counts_S82_radio / gv.area_S82, edges_S82_radio, fill=True, ec='k', lw=3.5,\
           fc=plt.get_cmap('cet_bmy',3)(0.5), label='S82', alpha=0.65)

ax1.stairs(counts_S82_norad / gv.area_S82, edges_S82_norad, fill=True, ec='k', lw=3.5,\
           fc=plt.get_cmap('cet_bmy',3)(0.5), label='S82', hatch='//', alpha=0.65)
ax1.stairs(counts_HETDEX_norad / gv.area_HETDEX, edges_HETDEX_norad, fill=True, ec='k', lw=3.5,\
           fc=plt.get_cmap('cet_bmy',3)(0.0), label='HETDEX', hatch='//', alpha=0.65)


HETDEX_patch       = mpatches.Patch(fc=plt.get_cmap('cet_bmy',3)(0.0), ec='k', label='HETDEX', lw=2.0, alpha=0.65)
S82_patch          = mpatches.Patch(fc=plt.get_cmap('cet_bmy',3)(0.5), ec='k', label='S82', lw=2.0, alpha=0.65)
pred_patch         = mpatches.Patch(fc='None', ec='k', label='Pred AGN + Radio', lw=2.0)
true_patch         = mpatches.Patch(fc='None', ec='k', label='Pred AGN + No Radio', hatch='///', lw=2.0)
ax1.legend(handles=[HETDEX_patch, S82_patch, pred_patch, true_patch], loc=1, fontsize=16, ncol=1,
           handletextpad=0.3, handlelength=1.0, columnspacing=0.5)
ax1.invert_xaxis()
ax1.tick_params(which='both', top=True, right=True, direction='in')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(which='major', length=8, width=1.5)
ax1.tick_params(which='minor', length=4, width=1.5)
ax1.set_xlabel('$\mathrm{mag}_{\mathrm{W4}}\,[\mathrm{AB}]$', size=20)
ax1.set_ylabel('Normalised\nfrequency [$\mathrm{deg}^{-2}$]', size=20)
plt.setp(ax1.spines.values(), linewidth=3.5)
plt.setp(ax1.spines.values(), linewidth=3.5)
fig.tight_layout()
plt.savefig(paths.figures / 'hist_W4_radio_non_radio_HETDEX_S82_nonimputed.pdf', bbox_inches='tight')
