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
validation_idx   = np.loadtxt(paths.data / 'indices_validation.txt')

feats_2_use      = ['ID', 'radio_AGN', 'pred_prob_rAGN']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
catalog_HETDEX_df = catalog_HETDEX_df.loc[validation_idx]


cm_rAGN = gf.conf_mat_func(catalog_HETDEX_df.loc[:, 'radio_AGN'], 
                           catalog_HETDEX_df.loc[:, 'pred_prob_rAGN'])

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1 = gf.plot_conf_mat(cm_rAGN, title='HETDEX', axin=ax1,
                       display_labels=['No\nRadio AGN', 'Radio AGN'],
                       log_stretch=False)
ax1.texts[1].set_color('black')
ax1.texts[2].set_color('black')
ax1.texts[3].set_color('black')
plt.savefig(paths.figures / 'conf_matrix_rAGN_HETDEX_validation.pdf', bbox_inches='tight')
