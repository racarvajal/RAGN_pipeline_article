#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
import pandas as pd
import paths
import global_functions as gf
import global_variables as gv

def main():
    mpl.rcdefaults()
    plt.rcParams['text.usetex'] = gv.use_LaTeX

    file_name_HETDEX = paths.data / 'HETDEX_for_prediction.parquet'
    test_idx         = np.loadtxt(paths.data / 'indices_test.txt')

    feats_2_use      = ['ID', 'radio_AGN', 'pred_prob_rAGN']

    catalog_HETDEX_df = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)
    catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
    catalog_HETDEX_df = catalog_HETDEX_df.loc[test_idx]


    cm_rAGN = gf.conf_mat_func(catalog_HETDEX_df.loc[:, 'radio_AGN'], 
                               catalog_HETDEX_df.loc[:, 'pred_prob_rAGN'])

    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(111)
    ax1 = gf.plot_conf_mat(cm_rAGN, axin=ax1,
                           display_labels=[r'$\mathrm{No}$' + '\n' + r'$\mathrm{Radio}$' + '\n' + r'$\mathrm{AGN}$', r'$\mathrm{Radio}$' + '\n' + r'$\mathrm{AGN}$'], 
                           title=None,
                           log_stretch=False)
    ax1.texts[1].set_color('black')
    ax1.texts[2].set_color('black')
    ax1.texts[3].set_color('black')
    plt.savefig(paths.figures / 'conf_matrix_rAGN_HETDEX_test.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()