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

    file_name_S82 = paths.data / 'S82_for_prediction.parquet'

    feats_2_use      = ['ID', 'class', 'pred_prob_class', 'pred_prob_radio', 'Z', 'pred_Z']

    catalog_S82_df = pd.read_parquet(file_name_S82, engine='fastparquet', columns=feats_2_use)
    catalog_S82_df = catalog_S82_df.set_index(keys=['ID'])
    filter_known   = np.array(catalog_S82_df.loc[:, 'class'] == 0) |\
                     np.array(catalog_S82_df.loc[:, 'class'] == 1)
    catalog_S82_df = catalog_S82_df.loc[filter_known]

    filter_rAGN = np.array(catalog_S82_df.loc[:, 'pred_prob_class'] == 1) &\
                  np.array(catalog_S82_df.loc[:, 'pred_prob_radio'] == 1)

    fig             = plt.figure(figsize=(7.1,5.8))
    ax1             = fig.add_subplot(111, xscale='log', yscale='log')
    _ = gf.plot_redshift_compare(catalog_S82_df.loc[filter_rAGN, 'Z'], 
                                 catalog_S82_df.loc[filter_rAGN, 'pred_Z'],
                                 ax_pre=ax1, title=None, dpi=10, show_clb=True, log_stretch=False)

    plt.savefig(paths.figures / 'compare_redshift_rAGN_Stripe82.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()