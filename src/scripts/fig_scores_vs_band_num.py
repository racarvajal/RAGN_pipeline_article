#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
import pandas as pd
import paths
import global_functions as gf

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.h5'
test_idx         = np.loadtxt(paths.data / 'indices_test.txt')

feats_2_use      = ['ID', 'class', 'LOFAR_detect', 'pred_prob_class', 
                    'pred_prob_radio', 'Prob_AGN', 'Prob_radio', 
                    'Z', 'pred_Z', 'band_num']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
catalog_HETDEX_df = catalog_HETDEX_df.loc[test_idx]

score_to_use_1    = 'Prob_AGN'  # 'Prob_AGN', 'Prob_radio', 'pred_Z'
score_to_use_2    = 'Prob_radio'  # 'Prob_AGN', 'Prob_radio', 'pred_Z'
score_to_use_3    = 'pred_Z'  # 'Prob_AGN', 'Prob_radio', 'pred_Z'

true_value        = ['class', 'LOFAR_detect', 'Z']
predicted_value   = ['pred_prob_class', 'pred_prob_radio', 'pred_Z']

filter_rAGN = [np.ones_like(catalog_HETDEX_df.loc[:, score_to_use_1]).astype(bool),
               np.array(catalog_HETDEX_df.loc[:, predicted_value[0]] == 1),
               np.array(catalog_HETDEX_df.loc[:, predicted_value[0]] == 1) &\
               np.array(catalog_HETDEX_df.loc[:, predicted_value[1]] == 1)]

fig             = plt.figure(figsize=(9.5, 10))
grid            = fig.add_gridspec(ncols=1, nrows=3, height_ratios=[1, 1, 1],
                                   hspace=0.0, wspace=0.0)
axs             = {}
axs[0]          = fig.add_subplot(grid[0, 0], xscale='linear', yscale='linear')
axs[1]          = fig.add_subplot(grid[1, 0], yscale='linear', sharex=axs[0])
axs[2]          = fig.add_subplot(grid[2, 0], yscale='linear', sharex=axs[0])

boxprops      = dict(linewidth=2.0)
whiskerprops  = dict(linewidth=2.0)
capprops      = dict(linewidth=2.0)
meanprops     = dict(linewidth=2.0)
medianprops   = dict(linewidth=2.0)

xlabels       = [None, None, '$\mathtt{band\_num}$']
ylabels       = ['$\mathrm{AGN\:prob.}$', '$\mathrm{Radio\:prob.}$', '$\mathrm{Redshift}$']
top_plots     = [True, False, False]
bottom_plots  = [False, False, True]

bins_x        = [1.8, 2.2, 2.8, 3.2, 3.8, 4.2, 4.8, 5.2, 5.8, 
                 6.2, 6.8, 7.2, 7.8, 8.2, 8.8, 9.2, 9.8, 10.2, 
                 10.8, 11.2, 11.8, 12.2]

for count, score_to_use in enumerate([score_to_use_1, score_to_use_2, score_to_use_3]):
    bins_y        = np.linspace(np.nanmin(catalog_HETDEX_df.loc[filter_rAGN[count], score_to_use]), 
                                np.nanmax(catalog_HETDEX_df.loc[filter_rAGN[count], score_to_use]), 
                                14)
    _ = gf.plot_scores_band_num(catalog_HETDEX_df.loc[filter_rAGN[count], 'band_num'],
                            catalog_HETDEX_df.loc[filter_rAGN[count], score_to_use], 
                            ax_pre=axs[count], title=None, bins=[bins_x, bins_y], 
                            show_clb=True, log_stretch=False, xlabel=xlabels[count], 
                            ylabel=ylabels[count], top_plot=top_plots[count], 
                            bottom_plot=bottom_plots[count])

    vals_band_num = np.unique(catalog_HETDEX_df.loc[filter_rAGN[count], 'band_num'])

    all_scores    = []
    for count_b, num in enumerate(vals_band_num):
        filter_band_n      = np.array(catalog_HETDEX_df.loc[:, 'band_num'] == num)
        all_scores.append(catalog_HETDEX_df.loc[filter_rAGN[count] * filter_band_n, score_to_use])

    axs[count].boxplot(all_scores, positions=vals_band_num, showfliers=False, showmeans=True, 
                       meanline=True, widths=0.4, boxprops=boxprops, whiskerprops=whiskerprops, 
                       capprops=capprops, meanprops=meanprops, medianprops=medianprops)
    for count_b, num in enumerate(vals_band_num):
        filter_band_n      = np.array(catalog_HETDEX_df.loc[:, 'band_num'] == num)
        if count in [0, 1]:
            tmp_conf_mat = gf.conf_mat_func(catalog_HETDEX_df.loc[filter_rAGN[count] * filter_band_n, 
                                                                  true_value[count]], 
                                            catalog_HETDEX_df.loc[filter_rAGN[count] * filter_band_n, 
                                                                  predicted_value[count]])
            tmp_metric   = gf.Recall_from_CM(tmp_conf_mat)
        if count == 2:
            tmp_metric   = gf.sigma_nmad(catalog_HETDEX_df.loc[filter_rAGN[count] * filter_band_n, 
                                                               true_value[count]], 
                                         catalog_HETDEX_df.loc[filter_rAGN[count] * filter_band_n, 
                                                               predicted_value[count]])
        axs[count].annotate(text=f'{tmp_metric:.2f} -'.replace('nan', ' '), 
                            xy=(num, 0.7 * np.nanmax(np.hstack(all_scores))), 
                            xycoords='data', fontsize=18, ha='center', va='top',  rotation='vertical', 
                            path_effects=gf.pe2)
        axs[count].annotate(text=f'{np.sum(filter_rAGN[count] * filter_band_n):,}'.replace(',', '$\,$'), 
                            xy=(num, 0.935 * np.nanmax(np.hstack(all_scores))), 
                            xycoords='data', fontsize=18, ha='center', va='top', 
                            rotation='vertical', path_effects=gf.pe2)

axs[0].set_xlim(left=1.5, right=12.5)
axs[2].set_xticks(np.unique(catalog_HETDEX_df.loc[:, 'band_num']))
axs[2].set_xticklabels(np.unique(catalog_HETDEX_df.loc[:, 'band_num']))
plt.savefig(paths.figures / 'predicted_probs_band_num_HETDEX_test.pdf', bbox_inches='tight')
