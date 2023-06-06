#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
import pandas as pd
import paths
import global_functions as gf
import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.parquet'
test_idx         = np.loadtxt(paths.data / 'indices_test.txt')

feats_2_use      = ['ID', 'class', 'pred_prob_class']

catalog_HETDEX_df = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
catalog_HETDEX_df = catalog_HETDEX_df.loc[test_idx]

cm_AGN_gal = gf.conf_mat_func(catalog_HETDEX_df.loc[:, 'class'], 
                              catalog_HETDEX_df.loc[:, 'pred_prob_class'])

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(111)
ax1 = gf.plot_conf_mat(cm_AGN_gal, 
                       axin=ax1, 
                       display_labels=['Galaxy', 'AGN'], 
                       title=None, 
                       log_stretch=False)
ax1.texts[1].set_color('black')
ax1.texts[2].set_color('black')
ax1.texts[3].set_color('white')
plt.savefig(paths.figures / 'conf_matrix_AGN_HETDEX_test.pdf', bbox_inches='tight')
