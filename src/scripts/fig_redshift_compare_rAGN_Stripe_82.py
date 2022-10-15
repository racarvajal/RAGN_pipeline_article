#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_scatter_density
import colorcet as cc
import pandas as pd
import paths
import global_functions as gf

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_S82 = paths.data / 'S82_for_prediction.h5'

feats_2_use      = ['ID', 'class', 'pred_prob_class', 'pred_prob_radio', 'Z', 'pred_Z']

catalog_S82_df = pd.read_hdf(file_name_S82, key='df').loc[:, feats_2_use]
catalog_S82_df = catalog_S82_df.set_index(keys=['ID'])
filter_known   = np.array(catalog_S82_df.loc[:, 'class'] == 0) |\
                 np.array(catalog_S82_df.loc[:, 'class'] == 1)
catalog_S82_df = catalog_S82_df.loc[filter_known]

filter_rAGN = np.array(catalog_S82_df.loc[:, 'pred_prob_class'] == 1) &\
              np.array(catalog_S82_df.loc[:, 'pred_prob_radio'] == 1)

fig             = plt.figure(figsize=(6.8,5.3))
ax1             = fig.add_subplot(111, projection='scatter_density', xscale='log', yscale='log')
_ = gf.plot_redshift_compare(catalog_S82_df.loc[filter_rAGN, 'Z'], catalog_S82_df.loc[filter_rAGN, 'pred_Z'],\
                      ax_pre=ax1, title=None, dpi=10, show_clb=True, log_stretch=False)

plt.savefig(paths.figures / 'compare_redshift_rAGN_Stripe82.pdf', bbox_inches='tight')
