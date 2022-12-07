#!/usr/bin/env python

import numpy as np
import paths

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
import cmasher as cmr
import pandas as pd
import global_variables as gv
from global_functions import pe2

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX = paths.data / 'HETDEX_mags_imputed.h5'
file_name_S82    = paths.data / 'S82_mags_imputed.h5'

feats_2_use = ['gmag', 'rmag', 'imag', 'zmag', 'ymag', 'Jmag',
              'Hmag', 'Kmag', 'W1mproPM', 'W2mproPM', 'W3mag', 'W4mag']

mag_cols_lim = {'W1mproPM': 20.13, 'W2mproPM': 19.81, 'W3mag': 16.67,
                'W4mag': 14.62, 'gmag': 23.3, 'rmag': 23.2, 'imag': 23.1,
                'zmag': 22.3, 'ymag': 21.4, 'Jmag': 17.45, 'Hmag': 17.24,
                'Kmag': 16.59}  # Proper (5-sigma) limits

mag_cols_names   = {'W1mproPM': 'W1', 'W2mproPM': 'W2', 'W3mag': 'W3',
                    'W4mag': 'W4', 'gmag': 'g', 'rmag': 'r',
                    'imag': 'i', 'zmag': 'z', 'ymag': 'y',
                    'Jmag': 'J', 'Hmag': 'H', 'Kmag': 'K'}  # For strings in plot

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_S82_df    = pd.read_hdf(file_name_S82,    key='df').loc[:, feats_2_use]

n_cols = 2
n_rows = int(np.ceil((len(feats_2_use)) / 2))

min_magnitude_HETDEX  = catalog_HETDEX_df.min().min()
max_magnitude_HETDEX  = catalog_HETDEX_df.max().max()
min_magnitude_S82     = catalog_S82_df.min().min()
max_magnitude_S82     = catalog_S82_df.max().max()

min_magnitude  = np.nanmin([min_magnitude_HETDEX, min_magnitude_S82])
max_magnitude  = np.nanmax([max_magnitude_HETDEX, max_magnitude_S82])
mag_bins_both  = np.linspace(min_magnitude, max_magnitude, 50)

fig            = plt.figure(figsize=(4 * n_cols, 1.4 * n_rows), constrained_layout=True)

grid           = fig.add_gridspec(ncols=n_cols, nrows=n_rows, width_ratios=[1]*n_cols,
                                   height_ratios=[1]*n_rows, hspace=0.0, wspace=0.0)
axs            = {}
norm_z         = mcolors.Normalize(vmin=0, vmax=8)

for count, band in enumerate(feats_2_use):
    if count == 0:
        axs[count] = fig.add_subplot(grid[int(np.floor(count / n_cols)), int(count % n_cols)],
                                     xscale='linear', yscale='log')
    elif count != 0:
        axs[count] = fig.add_subplot(grid[int(np.floor(count / n_cols)), int(count % n_cols)],
                                     sharex=axs[0], sharey=axs[0])
    
    filt_lims_HETDEX = np.array(catalog_HETDEX_df.loc[:, band] < mag_cols_lim[band]) & np.isfinite(catalog_HETDEX_df.loc[:, band])
    filt_lims_S82    = np.array(catalog_S82_df.loc[:, band] < mag_cols_lim[band]) & np.isfinite(catalog_S82_df.loc[:, band])
    
    counts_HETDEX, edges_HETDEX = np.histogram(catalog_HETDEX_df.loc[filt_lims_HETDEX, band], bins=mag_bins_both)
    counts_S82,    edges_S82    = np.histogram(catalog_S82_df.loc[filt_lims_S82, band], bins=mag_bins_both)
    
    face_colour    = list(plt.get_cmap(gv.cmap_bands, len(feats_2_use) + 2)(count + 1))
    face_colour[3] = 0.45  # alpha=0.65
    face_colour    = tuple(face_colour)
    axs[count].stairs(counts_HETDEX / gv.area_HETDEX, edges_HETDEX, fill=True, ec='k', lw=2.0,
                          fc=face_colour, label=f'{band}')
    axs[count].stairs(counts_S82 / gv.area_S82, edges_S82, fill=False, ec='brown', lw=2.0,
                          fc=face_colour, label=f'{band}')
    
    axs[count].tick_params(which='both', top=True, right=True, direction='in')
    axs[count].tick_params(axis='both', which='major', labelsize=22)
    axs[count].tick_params(which='major', length=8, width=1.5)
    axs[count].tick_params(which='minor', length=4, width=1.5)
    axs[count].xaxis.set_minor_locator(AutoMinorLocator())
    plt.setp(axs[count].spines.values(), linewidth=2.5)
    plt.setp(axs[count].spines.values(), linewidth=2.5)
    if count % n_cols != 0:
        plt.setp(axs[count].get_yticklabels(), visible=False)
    if count < (len(feats_2_use) - n_cols):
        plt.setp(axs[count].get_xticklabels(), visible=False)
    axs[count].annotate(text=f'{mag_cols_names[band]}', xy=(0.02, 0.9),\
                         xycoords='axes fraction', fontsize=20, ha='left', va='top', path_effects=pe2)
    axs[count].annotate(text=f'HETDEX N={np.sum(filt_lims_HETDEX):,}'.replace(',','$\,$'), xy=(0.98, 0.9),\
                         xycoords='axes fraction', fontsize=16, ha='right', va='top', path_effects=pe2)
    axs[count].annotate(text=f'S82 N={np.sum(filt_lims_S82):,}'.replace(',','$\,$'), xy=(0.98, 0.7),\
                         xycoords='axes fraction', fontsize=16, ha='right', va='top', path_effects=pe2)
    axs[0].set_xlim(left=np.floor(min_magnitude), right=np.ceil(max_magnitude))
    axs[0].invert_xaxis()
HETDEX_patch = mpatches.Patch(fc='None', ec='k', label='HETDEX', lw=2.5)
S82_patch    = mpatches.Patch(fc='None', ec='brown', label='Stripe 82', lw=2.5)#
axs[len(feats_2_use) - 1].legend(handles=[HETDEX_patch, S82_patch], loc=3, fontsize=14,\
               handletextpad=0.3, handlelength=1.0, borderpad=0.3)

fig.supxlabel('$m\, \mathrm{[AB]}$', fontsize=26, x=0.52, y=0.045)
fig.supylabel('Normalised frequency [$\mathrm{deg}^{-2}$]', fontsize=26, va='center', y=0.49, x=0.04)
fig.tight_layout()
plt.savefig(paths.figures / 'hists_bands_norm_HETDEX_S82_imputed.pdf', bbox_inches='tight')

