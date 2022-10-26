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
validation_idx   = np.loadtxt(paths.data / 'indices_validation.txt')

feats_2_use      = ['ID', 'class', 'LOFAR_detect', 'Z', 'pred_Z']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
catalog_HETDEX_df = catalog_HETDEX_df.loc[validation_idx]
catalog_HETDEX_df['is_AGN'] = np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)

filter_rAGN = np.array(catalog_HETDEX_df.loc[:, 'is_AGN'] == 1) &\
              np.array(catalog_HETDEX_df.loc[:, 'LOFAR_detect'] == 1)

fig             = plt.figure(figsize=(7.5,6))
ax1             = fig.add_subplot(111, projection='scatter_density', xscale='log', yscale='log')
_ = gf.plot_redshift_compare(catalog_HETDEX_df.loc[filter_rAGN, 'Z'], catalog_HETDEX_df.loc[filter_rAGN, 'pred_Z'],\
                      ax_pre=ax1, title='Validation set', dpi=10, show_clb=True, log_stretch=False)

plt.savefig(paths.figures / 'compare_redshift_HETDEX_validation.pdf', bbox_inches='tight')
