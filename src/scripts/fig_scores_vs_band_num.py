#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_scatter_density
import cmasher as cmr
import pandas as pd
import paths
import global_functions as gf

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.h5'
test_idx         = np.loadtxt(paths.data / 'indices_test.txt')

feats_2_use      = ['ID', 'pred_prob_class', 'pred_prob_radio', 
                    'Prob_AGN', 'Prob_radio', 'Z', 'pred_Z', 'band_num']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
catalog_HETDEX_df = catalog_HETDEX_df.loc[test_idx]

score_to_use      = 'Prob_AGN'  # 'Prob_AGN', 'Prob_radio', 'pred_Z'

# filter_rAGN = np.array(catalog_HETDEX_df.loc[:, 'pred_prob_class'] == 1) &\
#               np.array(catalog_HETDEX_df.loc[:, 'pred_prob_radio'] == 1)
filter_rAGN = np.ones_like(catalog_HETDEX_df.loc[:, score_to_use]).astype(bool)

fig             = plt.figure(figsize=(7.1,5.8))
ax1             = fig.add_subplot(111, projection='scatter_density', xscale='linear', yscale='linear')

_ = gf.plot_scores_band_num(catalog_HETDEX_df.loc[filter_rAGN, 'band_num'],
                            catalog_HETDEX_df.loc[filter_rAGN, score_to_use], 
                            ax_pre=ax1, title=None, dpi=10, show_clb=True, log_stretch=False)

vals_band_num = np.unique(catalog_HETDEX_df.loc[filter_rAGN, 'band_num'])
mean_scores   = np.zeros_like(vals_band_num, dtype=float)
std_scores    = np.zeros_like(vals_band_num, dtype=float)
for count, num in enumerate(vals_band_num):
    filter_band_n      = np.array(catalog_HETDEX_df.loc[filter_rAGN, 'band_num'] == num)
    mean_scores[count] = np.nanmean(catalog_HETDEX_df.loc[filter_rAGN * filter_band_n, score_to_use])
    std_scores[count]  = np.nanstd(catalog_HETDEX_df.loc[filter_rAGN * filter_band_n, score_to_use])
ax1.errorbar(vals_band_num, mean_scores, yerr=std_scores, ls='None')

plt.savefig(paths.figures / 'prob_class_band_num_rAGN_HETDEX_test.pdf', bbox_inches='tight')
