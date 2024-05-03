#!/usr/bin/env python

import schemdraw
from schemdraw import flow
import numpy as np
import pandas as pd
import paths
import matplotlib as mpl
import matplotlib.pyplot as plt
import global_variables as gv

def main():
    mpl.rcdefaults()
    plt.rcParams['text.usetex'] = gv.use_LaTeX
    if gv.use_LaTeX:
        str_replace = '$\,$'
    else:
        str_replace = '\,'

    file_name_S82  = paths.data / 'S82_for_prediction.parquet'

    feats_2_use    = ['ID', 'class']

    catalog_S82_df  = pd.read_parquet(file_name_S82, engine='fastparquet', columns=feats_2_use)
    filter_known    = np.array(catalog_S82_df.loc[:, 'class'] == 0) |\
                  np.array(catalog_S82_df.loc[:, 'class'] == 1)
    full_size_S82   = len(catalog_S82_df)

    size_labelled   = np.sum(filter_known, dtype=int)
    size_unlabelled = full_size_S82 - size_labelled

    with schemdraw.Drawing(show=False, fontsize=14, lw=2.5) as S:
        S += (HETDEX := flow.Terminal(w=3, h=1.5).label(f'S82\n{full_size_S82:,}'.replace(',', str_replace)))
        S += schemdraw.elements.lines.Gap().at(HETDEX.S)

        S += (Labelled := flow.RoundBox(w=3, h=1.5, anchor='ENE').label(f'Labelled\n{size_labelled:,}'.replace(',', str_replace)))
        S += (Unlabelled := flow.RoundBox(w=3, h=1.5, anchor='WNW').label(f'Unlabelled\n{size_unlabelled:,}'.replace(',', str_replace)))
        S += flow.Arrow().length(S.unit/3).at(HETDEX.S).to(Labelled.N)
        S += flow.Arrow().length(S.unit/3).at(HETDEX.S).to(Unlabelled.N)

        S.save(paths.figures / 'flowchart_S82_subsets.pdf')

if __name__ == "__main__":
    main()