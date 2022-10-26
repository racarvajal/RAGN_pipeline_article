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

file_name_S82 = paths.data / 'S82_for_prediction.h5'

feats_2_use      = ['ID', 'class', 'radio_AGN', 'pred_prob_rAGN']

catalog_S82_df = pd.read_hdf(file_name_S82, key='df').loc[:, feats_2_use]
filter_known   = np.array(catalog_S82_df.loc[:, 'class'] == 0) |\
                 np.array(catalog_S82_df.loc[:, 'class'] == 1)
catalog_S82_df = catalog_S82_df.loc[filter_known]


cm_rAGN = gf.conf_mat_func(catalog_S82_df.loc[:, 'radio_AGN'], 
                           catalog_S82_df.loc[:, 'pred_prob_rAGN'])

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1 = gf.plot_conf_mat(cm_rAGN, title='S82', axin=ax1,
                       display_labels=['No\nRadio AGN', 'Radio AGN'],
                       log_stretch=False)
ax1.texts[1].set_color('black')
ax1.texts[2].set_color('black')
ax1.texts[3].set_color('black')
plt.savefig(paths.figures / 'conf_matrix_rAGN_Stripe82.pdf', bbox_inches='tight')
