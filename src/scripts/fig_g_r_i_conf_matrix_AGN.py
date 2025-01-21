#!/usr/bin/env python

import os 
os.environ["PATH"] += os.pathsep + '/usr/bin'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import PowerStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import cmasher as cmr
import pandas as pd
import paths
import global_variables as gv
import global_functions as gf

mpl.rcdefaults()
plt.rcParams['text.usetex'] = gv.use_LaTeX

mag_cols_lim = {'W1mproPM': 20.13, 'W2mproPM': 19.81, 'W3mag': 16.67,
                'W4mag': 14.62, 'gmag': 23.3, 'rmag': 23.2, 'imag': 23.1,
                'zmag': 22.3, 'ymag': 21.4, 'Jmag': 17.45, 'Hmag': 17.24,
                'Kmag': 16.59}  # Proper (5-sigma) limits
for key in mag_cols_lim.keys():
    mag_cols_lim[key] = np.float32(mag_cols_lim[key])

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.parquet'
file_name_S82    = paths.data / 'S82_for_prediction.parquet'
test_idx         = np.loadtxt(paths.data / 'indices_test.txt')

cm_gradient_HETDEX = gf.create_colour_gradient('#1E88E5')
cm_gradient_S82    = gf.create_colour_gradient('#D32F2F')
cm_gradient_black  = gf.create_colour_gradient('#000000')

# colours would be: X-axis: mag_a - mag_b, Y-axis: mag_b - mag_c
mag_a = 'gmag'
mag_b = 'rmag'
mag_c = 'imag'

feats_2_use      = ['ID', 'class', 'pred_prob_class', 'pred_prob_radio', 
                    'Z', 'pred_Z', mag_a, mag_b, mag_c]

catalog_HETDEX_df = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])
filter_known_HTDX = np.array(catalog_HETDEX_df.loc[:, 'class'] == 0) |\
                    np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
unknown_HTDX_df   = catalog_HETDEX_df.loc[~filter_known_HTDX].copy()
catalog_HETDEX_df = catalog_HETDEX_df.loc[test_idx]

catalog_S82_df    = pd.read_parquet(file_name_S82, engine='fastparquet', columns=feats_2_use)
catalog_S82_df    = catalog_S82_df.set_index(keys=['ID'])
filter_known_S82  = np.array(catalog_S82_df.loc[:, 'class'] == 0) |\
                    np.array(catalog_S82_df.loc[:, 'class'] == 1)
catalog_S82_df    = catalog_S82_df.loc[filter_known_S82]

for col in feats_2_use[1::]:
    if catalog_HETDEX_df.loc[:, col].dtype == 'float64':
        catalog_HETDEX_df.loc[:, col] = catalog_HETDEX_df.loc[:, col].astype('float32')
    if catalog_S82_df.loc[:, col].dtype == 'float64':
        catalog_S82_df.loc[:, col] = catalog_S82_df.loc[:, col].astype('float32')
    if unknown_HTDX_df.loc[:, col].dtype == 'float64':
        unknown_HTDX_df.loc[:, col] = unknown_HTDX_df.loc[:, col].astype('float32')

n_rows = 2
n_cols = 2

fig             = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows), constrained_layout=False)

grid            = fig.add_gridspec(ncols=n_cols, nrows=n_rows, width_ratios=[1]*n_cols,
                                   height_ratios=[1]*n_rows, hspace=0.0, wspace=0.0)
axs             = {}
axs_twinx       = {}
axs_twiny       = {}

unknown_HTDX_df = pd.concat([catalog_HETDEX_df, unknown_HTDX_df], ignore_index=True)

correcting_filter_X = np.array(unknown_HTDX_df.loc[:, mag_a] - unknown_HTDX_df.loc[:, mag_b] > -2.25) & np.array(unknown_HTDX_df.loc[:, mag_a] - unknown_HTDX_df.loc[:, mag_b] < 3.0)
correcting_filter_Y = np.array(unknown_HTDX_df.loc[:, mag_b] - unknown_HTDX_df.loc[:, mag_c] > -1.75) & np.array(unknown_HTDX_df.loc[:, mag_b] - unknown_HTDX_df.loc[:, mag_c] < 3.0)

unknown_HTDX_df = unknown_HTDX_df.loc[correcting_filter_X & correcting_filter_Y]

filt_colours_m = np.array(unknown_HTDX_df.loc[:, mag_a]  != mag_cols_lim[mag_a]) &\
                 np.array(unknown_HTDX_df.loc[:, mag_b]  != mag_cols_lim[mag_b]) &\
                 np.array(unknown_HTDX_df.loc[:, mag_c]  != mag_cols_lim[mag_c])

filt_plot      = np.isfinite(unknown_HTDX_df.loc[:, mag_a] -
                             unknown_HTDX_df.loc[:, mag_b]) &\
                 np.isfinite(unknown_HTDX_df.loc[:, mag_b] -
                             unknown_HTDX_df.loc[:, mag_c])

dens_plot_data_x = (unknown_HTDX_df.loc[filt_colours_m * filt_plot, mag_a] -\
                    unknown_HTDX_df.loc[filt_colours_m * filt_plot, mag_b])
dens_plot_data_y = (unknown_HTDX_df.loc[filt_colours_m * filt_plot, mag_b] -\
                    unknown_HTDX_df.loc[filt_colours_m * filt_plot, mag_c])

cm_mat_AGN_filter_HETDEX = np.array([[(np.array(catalog_HETDEX_df['class'] == 0)   & np.array(catalog_HETDEX_df['pred_prob_class'] == 0)),
                                      (np.array(catalog_HETDEX_df['class'] == 0)   & np.array(catalog_HETDEX_df['pred_prob_class'] == 1))],
                                      [(np.array(catalog_HETDEX_df['class'] == 1)   & np.array(catalog_HETDEX_df['pred_prob_class'] == 0)),
                                       (np.array(catalog_HETDEX_df['class'] == 1)   & np.array(catalog_HETDEX_df['pred_prob_class'] == 1))]])

cm_mat_AGN_filter_S82    = np.array([[(np.array(catalog_S82_df['class'] == 0)   & np.array(catalog_S82_df['pred_prob_class'] == 0)),
                                      (np.array(catalog_S82_df['class'] == 0)   & np.array(catalog_S82_df['pred_prob_class'] == 1))],
                                      [(np.array(catalog_S82_df['class'] == 1)   & np.array(catalog_S82_df['pred_prob_class'] == 0)),
                                       (np.array(catalog_S82_df['class'] == 1)   & np.array(catalog_S82_df['pred_prob_class'] == 1))]])
dens_plts = {}
cont_plts = {}

x_axis_dens_AGN_HETDEX = {}
y_axis_dens_AGN_HETDEX = {}
x_axis_dens_AGN_S82 = {}
y_axis_dens_AGN_S82 = {}

AB_lims_x = (-1.0, 2.5)
AB_lims_y = (-1.3, 1.7)

contour_levels  = [1, 2, 3]  # in sigmas
sigmas_perc     = [0.39346934, 0.86466472, 0.988891, 0.99966454]  # [0.39346934, 0.86466472, 0.988891, 0.99966454, 0.99999627]
sigmas_perc_inv = [1. - sigma for sigma in sigmas_perc][::-1]  # 1, 2, 3, 4 sigma
# sigmas_perc     = [0.5, 0.75, 0.999]  # [50%, 75%, 99.9% of data]
nstep           = len(sigmas_perc)
seq_cont        = np.linspace(0, 1, nstep)
n_bins_large    = [[30, 30], [30, 30], [30, 30], [30, 30]]

num_levels_dens = 20
txt_x_positions = [0.67, 0.60, 0.60, 0.65]

# norm_val  = mcolors.CenteredNorm(vcenter=0.5)
try:
    norm_dens = ImageNormalize(vmin=0, stretch=PowerStretch(0.35))
except:
    pass
# norm_cont = ImageNormalize(vmin=contour_levels[0], vmax=contour_levels[-1], stretch=PowerStretch(0.35)) # PowerStretch(0.35), LogStretch()

for count, idx_ax in enumerate(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])):
    if count == 0:
        axs[count] = fig.add_subplot(grid[tuple(idx_ax)])
    if count != 0:
        axs[count] = fig.add_subplot(grid[tuple(idx_ax)], sharex=axs[0], sharey=axs[0])
    
    _, _, _, dens_plts[count] = axs[count].hist2d(dens_plot_data_x, dens_plot_data_y, 
                                                  bins=[50, 50], cmin=1, norm=norm_dens,
                                                  cmap=plt.get_cmap(gv.cmap_dens_plots),
                                                  zorder=0)
    # Create a Rectangle patch
    rect_S23 = Rectangle((-0.2, -0.2), 1.0, 0.8, linewidth=3,
                        edgecolor='k', facecolor='none')
    axs[count].add_patch(rect_S23)

    x_axis_dens_AGN_HETDEX[count] = (catalog_HETDEX_df[cm_mat_AGN_filter_HETDEX[tuple(idx_ax)] ][mag_a] -\
               catalog_HETDEX_df[cm_mat_AGN_filter_HETDEX[tuple(idx_ax)] ][mag_b])
    y_axis_dens_AGN_HETDEX[count] = (catalog_HETDEX_df[cm_mat_AGN_filter_HETDEX[tuple(idx_ax)] ][mag_b] -\
               catalog_HETDEX_df[cm_mat_AGN_filter_HETDEX[tuple(idx_ax)]][mag_c])
    
    x_axis_dens_AGN_S82[count] = (catalog_S82_df[cm_mat_AGN_filter_S82[tuple(idx_ax)] ][mag_a] -\
               catalog_S82_df[cm_mat_AGN_filter_S82[tuple(idx_ax)]][mag_b])
    y_axis_dens_AGN_S82[count] = (catalog_S82_df[cm_mat_AGN_filter_S82[tuple(idx_ax)]][mag_b] -\
               catalog_S82_df[cm_mat_AGN_filter_S82[tuple(idx_ax)]][mag_c])
    
    min_X  = np.nanmin([np.nanmin(x_axis_dens_AGN_HETDEX[count]), np.nanmin(x_axis_dens_AGN_S82[count])])
    max_X  = np.nanmax([np.nanmax(x_axis_dens_AGN_HETDEX[count]), np.nanmax(x_axis_dens_AGN_S82[count])])
    min_Y  = np.nanmin([np.nanmin(y_axis_dens_AGN_HETDEX[count]), np.nanmin(y_axis_dens_AGN_S82[count])])
    max_Y  = np.nanmax([np.nanmax(y_axis_dens_AGN_HETDEX[count]), np.nanmax(y_axis_dens_AGN_S82[count])])
    n_bins = n_bins_large[count]
    # bins_X = np.linspace(min_X, max_X, n_bins[1])
    # bins_Y = np.linspace(min_Y, max_Y, n_bins[0])
    bins_X = np.linspace(-3, 3, n_bins[1])
    bins_Y = np.linspace(-3, 3, n_bins[0])

    H_HET, xedges_HET, yedges_HET = np.histogram2d(x_axis_dens_AGN_HETDEX[count],
                                                    y_axis_dens_AGN_HETDEX[count],
                                                    bins=n_bins, density=False)
    H_S82, xedges_S82, yedges_S82 = np.histogram2d(x_axis_dens_AGN_S82[count],
                                                    y_axis_dens_AGN_S82[count],
                                                    bins=n_bins, density=False)
    # Flatten histogram and sort by density
    H_flat_HET   = H_HET.ravel()
    H_flat_S82   = H_S82.ravel()
    H_sorted_HET = np.sort(H_flat_HET)[::-1]  # Sort descending
    H_sorted_S82 = np.sort(H_flat_S82)[::-1]  # Sort descending
    cumulative_density_HET = np.cumsum(H_sorted_HET) / np.sum(H_sorted_HET)
    cumulative_density_S82 = np.cumsum(H_sorted_S82) / np.sum(H_sorted_S82)

    # Find bin count thresholds for the desired percentiles
    target_percentiles_HET = np.array(sigmas_perc) * np.sum(H_HET)
    target_percentiles_S82 = np.array(sigmas_perc) * np.sum(H_S82)
    density_thresholds_HET = []
    density_thresholds_S82 = []
    for target in target_percentiles_HET:
        idx = np.argmin(np.abs(cumulative_density_HET * np.sum(H_HET) - target))
        threshold = H_sorted_HET[idx]
        density_thresholds_HET.append(threshold)
    for target in target_percentiles_S82:
        idx = np.argmin(np.abs(cumulative_density_S82 * np.sum(H_S82) - target))
        threshold = H_sorted_S82[idx]
        density_thresholds_S82.append(threshold)
    density_thresholds_HET = sorted(set(density_thresholds_HET), reverse=True)  # Ensure unique, decreasing values
    density_thresholds_S82 = sorted(set(density_thresholds_S82), reverse=True)  # Ensure unique, decreasing values

    # fix probable lines not closing
    H_HET, x_centers_HET, y_centers_HET = gf.pad_matrix_zeros(H_HET, xedges_HET, yedges_HET)
    H_S82, x_centers_S82, y_centers_S82 = gf.pad_matrix_zeros(H_S82, xedges_S82, yedges_S82)

    # Plot contours using the computed density thresholds
    CS_HET = axs[count].contour(x_centers_HET, y_centers_HET, H_HET.T,
                        levels=density_thresholds_HET[::-1],
                        colors=cm_gradient_HETDEX.reversed()(seq_cont),
                        linewidths=2.25, zorder=3)
    CS_S82 = axs[count].contour(x_centers_S82, y_centers_S82, H_S82.T,
                        levels=density_thresholds_S82[::-1],
                        colors=cm_gradient_S82.reversed()(seq_cont),
                        linewidths=2.25, zorder=3)
    # Calculate percentage levels from density thresholds
    # density_percentages = [100.0 * cumulative_density[np.searchsorted(np.sort(H_flat), level)] for level in density_thresholds]
    density_percentages = [100 * sigma_perc for sigma_perc in sigmas_perc]
    n_sources_HETDEX   = np.sum(cm_mat_AGN_filter_HETDEX[tuple(idx_ax)])
    n_sources_S82   = np.sum(cm_mat_AGN_filter_S82[tuple(idx_ax)])
    outlier_number_HETDEX  = int(np.floor(n_sources_HETDEX * (1 - sigmas_perc[-1]))) # expected number of points outside last contour
    outlier_number_S82     = int(np.floor(n_sources_S82 * (1 - sigmas_perc[-1]))) # expected number of points outside last contour

    try:
        p_HET = CS_HET.get_paths()
    except Exception as e:
        p_HET = CS_HET.collections[0].get_paths()
    try:
        p_S82 = CS_S82.get_paths()
    except Exception as e:
        p_S82 = CS_S82.collections[0].get_paths()
    inside_HET = np.full_like(x_axis_dens_AGN_HETDEX[count], False, dtype=bool)
    inside_S82 = np.full_like(x_axis_dens_AGN_S82[count], False, dtype=bool)
    for level in p_HET:
        inside_HET |= level.contains_points(np.column_stack((x_axis_dens_AGN_HETDEX[count],
                                                            y_axis_dens_AGN_HETDEX[count])))
    for level in p_S82:
        inside_S82 |= level.contains_points(np.column_stack((x_axis_dens_AGN_S82[count],
                                                            y_axis_dens_AGN_S82[count])))
    print(f'For cell {count}: -------------------------')
    print(f'There are {np.sum(inside_HET)} HETDEX sources inside contours')
    print(f'There are {np.sum(~inside_HET)} HETDEX sources outside contours')
    print(f'It is expected to have {outlier_number_HETDEX} HETDEX sources outside contours')
    print(f'There are {np.sum(inside_S82)} S82 sources inside contours')
    print(f'There are {np.sum(~inside_S82)} S82 sources outside contours')
    print(f'It is expected to have {outlier_number_S82} S82 sources outside contours')
    out_HET, = axs[count].plot(x_axis_dens_AGN_HETDEX[count].loc[~inside_HET], 
                        y_axis_dens_AGN_HETDEX[count].loc[~inside_HET],
                        marker='x', ls='None', color=gf.colour_AGN, zorder=2, alpha=0.8)
    out_S82, = axs[count].plot(x_axis_dens_AGN_S82[count].loc[~inside_S82], 
                        y_axis_dens_AGN_S82[count].loc[~inside_S82],
                        marker='x', ls='None', color=gf.colour_SFG, zorder=2, alpha=0.8)

    n_sources_HETDEX   = np.sum(cm_mat_AGN_filter_HETDEX[tuple(idx_ax)])
    n_sources_S82   = np.sum(cm_mat_AGN_filter_S82[tuple(idx_ax)])
    outlier_number_HETDEX  = int(np.floor(n_sources_HETDEX * (1 - sigmas_perc[-1]))) # expected number of points outside last contour
    outlier_number_S82     = int(np.floor(n_sources_S82 * (1 - sigmas_perc[-1]))) # expected number of points outside last contour
    axs[count].annotate(text=rf'$\mathrm{{HETDEX - N}} = {n_sources_HETDEX: >6,d}$'.replace(',','$\,$') + '\n' + rf'$\mathrm{{S82 - N}} = {n_sources_S82: >6,d}$'.replace(',','$\,$'),
                        xy=(txt_x_positions[count], 0.96), xycoords='axes fraction', fontsize=18, 
                        ha='right', va='top', path_effects=gf.pe2, zorder=11)
    
    axs[count].tick_params(which='both', top=True, right=True,\
                            bottom=True, left=True, direction='in')
    axs[count].tick_params(axis='both', which='major', labelsize=20)
    axs[count].tick_params(axis='both', which='minor', labelsize=20)
    axs[count].tick_params(which='major', length=8, width=1.5)
    axs[count].tick_params(which='minor', length=4, width=1.5)
    plt.setp(axs[count].spines.values(), linewidth=2.5)
    plt.setp(axs[count].spines.values(), linewidth=2.5)

# Colorbar density
#axins0 = inset_axes(axs[1], width='100%', height='100%', bbox_transform=axs[1].transAxes,\
#                    loc=1, bbox_to_anchor=(0.9, 0.08, 0.05, 0.85), borderpad=0)
axins0 = make_axes_locatable(axs[1])

#clb_dens    = fig.colorbar(dens_plts[1], cax=axins0, orientation='vertical', 
#                           cmap=plt.get_cmap(gv.cmap_dens_plots), 
#                           norm=norm_dens)#, format=lambda x, pos: f"{x:,.0f}".replace(",", "$\,$"))
clb_dens    = fig.colorbar(dens_plts[1], cax=axs[1].inset_axes((0.9, 0.05, 0.05, 0.85)), 
                           orientation='vertical', cmap=plt.get_cmap(gv.cmap_dens_plots), 
                           norm=norm_dens, format=lambda x, pos: f"${x:,.0f}$".replace(",", "$\,$"))
#axins0.yaxis.set_ticks_position('left')
#axins0.yaxis.set_ticklabels(axins0.yaxis.get_ticklabels(), path_effects=gf.pe2)
clb_dens.ax.yaxis.set_ticks_position('left')
clb_dens.ax.yaxis.set_ticklabels(clb_dens.ax.yaxis.get_ticklabels(), path_effects=gf.pe2)
clb_dens.ax.tick_params(labelsize=20)
clb_dens.outline.set_linewidth(2.5)

axs[2].plot([-3], [-3], marker=None, ls='-', lw=4.5, c='k', label=r'$\mathrm{S23}$', zorder=0)

axs[1].plot([-3], [-3], marker='s', ls='None', c=plt.get_cmap(gv.cmap_dens_plots)(1.1), label=r'$\mathrm{HETDEX}$', zorder=0)

cont_S82 = axs[3].scatter([-1], [10], marker='o', edgecolor=gf.colour_SFG, 
               color=gf.colour_SFG_shade, s=80, 
               label=r'$\mathrm{MQC - SDSS}$' + '\n' + r'$\mathrm{S82}$', linewidths=2.5)
cont_HET = axs[3].scatter([-1], [10], marker='o', 
               edgecolor=gf.colour_AGN, color=gf.colour_AGN_shade, s=80, 
               label=r'$\mathrm{MQC - SDSS}$' + '\n' + r'$\mathrm{HETDEX}$', linewidths=2.5)

# axs[0].set_xlim(left=AB_lims_x[0], right=AB_lims_x[1])
# axs[0].set_ylim(bottom=AB_lims_y[0], top=AB_lims_y[1])

axs[0].xaxis.set_ticks_position('top')
axs[1].xaxis.set_ticks_position('top')
axs[1].yaxis.set_ticks_position('right')
axs[3].yaxis.set_ticks_position('right')
    
# plt.setp(axs[0].get_xticklabels(), visible=False)
# plt.setp(axs[1].get_yticklabels(), visible=False)
# plt.setp(axs[1].get_xticklabels(), visible=False)
# plt.setp(axs[3].get_yticklabels(), visible=False)

axs[0].set_ylabel(r'$\mathrm{SFG}$', fontsize=22, rotation='horizontal', labelpad=25)
axs[2].set_xlabel(r'$\mathrm{SFG}$', fontsize=22)
axs[2].set_ylabel(r'$\mathrm{AGN}$', fontsize=22, rotation='horizontal', labelpad=25)
axs[3].set_xlabel(r'$\mathrm{AGN}$', fontsize=22)

axs[1].legend(loc=3, fontsize=18, ncol=1, columnspacing=.25, 
              handletextpad=0.2, handlelength=0.8, framealpha=0.75)
axs[2].legend(loc=3, fontsize=18, ncol=1, columnspacing=.25, 
              handletextpad=0.2, handlelength=0.8, framealpha=0.75)
# axs[3].legend(loc=4, fontsize=14, ncol=2, columnspacing=.25, 
#               handletextpad=0.2, handlelength=0.8, framealpha=0.75)
arr_legends = [r'$\mathrm{MQC - SDSS}$' + '\n' + r'$\mathrm{HETDEX}$', r'$\mathrm{MQC - SDSS}$' + '\n' + r'$\mathrm{S82}$']
axs[3].legend([(cont_HET, out_HET), (cont_S82, out_S82)], arr_legends,
              scatterpoints=1, numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=14,
              ncol=2, columnspacing=.30, handletextpad=0.30,
              handlelength=1.10, framealpha=0.75, loc=4)

fig.supxlabel('$m_{\mathrm{g}} - m_{\mathrm{r}}\, \mathrm{[AB]}$\n$\mathrm{Predicted ~ class}$', 
              fontsize=25, ha='left', x=0.49, y=0.05)
fig.supylabel('$\mathrm{True ~ class}$\n$m_{\mathrm{r}} - m_{\mathrm{i}}\, \mathrm{[AB]}$', 
              fontsize=26, x=0.09, y=0.58, va='center', ha='center')
# fig.suptitle('AGN prediction', fontsize=20, x=0.55)
fig.tight_layout()
save_filename = f'PS1_colour_colour_conf_matrix_AGN_HETDEX_test_S82_all.pdf'
plt.savefig(paths.figures / save_filename, bbox_inches='tight')

