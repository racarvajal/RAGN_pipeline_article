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

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.parquet'
test_idx         = np.loadtxt(paths.data / 'indices_test.txt')

feats_2_use      = ['ID', 'class', 'LOFAR_detect', 'pred_prob_radio']

catalog_HETDEX_df = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
catalog_HETDEX_df = catalog_HETDEX_df.loc[test_idx]

filter_AGN = np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)

cm_radio = gf.conf_mat_func(catalog_HETDEX_df.loc[filter_AGN, 'LOFAR_detect'], 
                            catalog_HETDEX_df.loc[filter_AGN, 'pred_prob_radio'])
# Fix one element badly classified because of number precision
cm_radio[1, 0] += 1
cm_radio[1, 1] -= 1

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1 = gf.plot_conf_mat(cm_radio, 
                       axin=ax1, 
                       display_labels=['No\nRadio', 'Radio'], 
                       title=None, 
                       log_stretch=False)
ax1.texts[1].set_color('black')
ax1.texts[2].set_color('black')
ax1.texts[3].set_color('black')
plt.savefig(paths.figures / 'conf_matrix_radio_HETDEX_test.pdf', bbox_inches='tight')
