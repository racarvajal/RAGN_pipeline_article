#!/usr/bin/env python

import paths
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.visualization import PowerStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import cmasher as cmr
import pandas as pd
import global_functions as gf
import global_variables as gv

mpl.rcdefaults()
plt.rcParams['text.usetex'] = gv.use_LaTeX

mag_cols_lim = {'W1mproPM': 20.13, 'W2mproPM': 19.81, 'W3mag': 16.67,
                'W4mag': 14.62, 'gmag': 23.3, 'rmag': 23.2, 'imag': 23.1,
                'zmag': 22.3, 'ymag': 21.4, 'Jmag': 17.45, 'Hmag': 17.24,
                'Kmag': 16.59}  # Proper (5-sigma) limits
vega_shift   = {'W1mproPM': 2.699, 'W2mproPM': 3.339, 'W1mag': 2.699, 'W2mag': 3.339,
                'W3mag': 5.174, 'W4mag': 6.620, 'Jmag': 0.910, 'Hmag': 1.390, 
                'Kmag': 1.850, 'gmag': 0.4810, 'rmag': 0.6170, 'imag': 0.7520, 
                'zmag': 0.8660, 'ymag': 0.9620}
for band in mag_cols_lim.keys():
    mag_cols_lim[band] = np.float32(mag_cols_lim[band])
    vega_shift[band]   = np.float32(vega_shift[band])

file_name_HETDEX = paths.data / 'HETDEX_for_prediction.parquet'

feats_2_use      = ['class', 'gmag', 'rmag', 'W1mproPM', 'W2mproPM', 'g_r', 'W1_W2']

catalog_HETDEX_df = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)

for col in catalog_HETDEX_df.columns:
    if catalog_HETDEX_df.loc[:, col].dtype == 'float64':
        catalog_HETDEX_df.loc[:, col] = catalog_HETDEX_df.loc[:, col].astype('float32')

filter_AGN_HETDEX = np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
filter_gal_HETDEX = np.array(catalog_HETDEX_df.loc[:, 'class'] == 0)

filter_imputed_HETDEX = np.array(catalog_HETDEX_df.loc[:, 'W1mproPM'] != mag_cols_lim['W1mproPM']) &\
                        np.array(catalog_HETDEX_df.loc[:, 'W2mproPM'] != mag_cols_lim['W2mproPM']) &\
                        np.array(catalog_HETDEX_df.loc[:, 'gmag']     != mag_cols_lim['gmag']) &\
                        np.array(catalog_HETDEX_df.loc[:, 'rmag']     != mag_cols_lim['rmag'])

filter_valid_colours  = np.isfinite(catalog_HETDEX_df.loc[:, 'W1mproPM'] - catalog_HETDEX_df.loc[:, 'W2mproPM']) &\
                        np.isfinite(catalog_HETDEX_df.loc[:, 'gmag'] - catalog_HETDEX_df.loc[:, 'rmag'])

filter_used_data      = filter_valid_colours & filter_imputed_HETDEX

fig                = plt.figure(figsize=(7.2,6.5))
ax1                = fig.add_subplot(111, xscale='linear', yscale='linear')

num_levels_dens    = 20

AB_lims_x          = (-2.5, 4.1)
AB_lims_y          = (-1.3, 1.8)

try:
    norm_dens = ImageNormalize(vmin=0, vmax=1500, stretch=PowerStretch(0.35))
except:
    pass

dens_plot_data_x = (catalog_HETDEX_df.loc[filter_used_data, 'gmag'] -
                        catalog_HETDEX_df.loc[filter_used_data, 'rmag'])
dens_plot_data_y = (catalog_HETDEX_df.loc[filter_used_data, 'W1mproPM'] -
                        catalog_HETDEX_df.loc[filter_used_data, 'W2mproPM'])

_, _, _, dens_CW_HETDEX = ax1.hist2d(dens_plot_data_x, dens_plot_data_y, 
                                     bins=[400, 250], cmin=1, norm=norm_dens, 
                                     cmap=plt.get_cmap(gv.cmap_dens_plots), zorder=0)
n_sources_CW     = np.nansum(filter_used_data)

min_X  = np.nanmin(catalog_HETDEX_df.loc[filter_used_data, 'g_r'])
max_X  = np.nanmax(catalog_HETDEX_df.loc[filter_used_data, 'g_r'])
min_Y  = np.nanmin(catalog_HETDEX_df.loc[filter_used_data, 'W1_W2'])
max_Y  = np.nanmax(catalog_HETDEX_df.loc[filter_used_data, 'W1_W2'])
n_bins = [50, 75]
bins_X = np.linspace(min_X, max_X, n_bins[1])
bins_Y = np.linspace(min_Y, max_Y, n_bins[0])

sigmas_perc     = [0.39346934, 0.86466472, 0.988891]  # [0.39346934, 0.86466472, 0.988891, 0.99966454, 0.99999627]
sigmas_perc_inv = [1. - sigma for sigma in sigmas_perc][::-1]  # 1, 2, 3 sigma (2-dimensional)

nstep = 4
seq_cont   = np.logspace(-1.5, 0.0, nstep)
seq_cont   = np.insert(seq_cont, 0, 0.0)
seq_fill   = np.logspace(-1.5, 0.0, nstep+2)
seq_fill   = np.insert(seq_fill, 0, 0.0)

cm_gradient_AGN = gf.create_colour_gradient(mcolors.to_hex(plt.get_cmap(gv.cmap_hists)(0.2)))
cm_gradient_Gal = gf.create_colour_gradient(mcolors.to_hex(plt.get_cmap(gv.cmap_hists)(0.8)))

H_AGN, xedges_AGN, yedges_AGN = np.histogram2d(catalog_HETDEX_df.loc[filter_used_data & filter_AGN_HETDEX, 'g_r'], 
                                   catalog_HETDEX_df.loc[filter_used_data & filter_AGN_HETDEX, 'W1_W2'], bins=n_bins)
H_Gal, xedges_Gal, yedges_Gal = np.histogram2d(catalog_HETDEX_df.loc[filter_used_data & filter_gal_HETDEX, 'g_r'], 
                                   catalog_HETDEX_df.loc[filter_used_data & filter_gal_HETDEX, 'W1_W2'], bins=n_bins)

H_AGN_smooth = gf.clean_and_smooth_matrix(H_AGN, sigma=0.9)
H_Gal_smooth = gf.clean_and_smooth_matrix(H_Gal, sigma=0.9)

Z_AGN = (H_AGN_smooth.T - H_AGN_smooth.T.min())/(H_AGN_smooth.T.max() - H_AGN_smooth.T.min())
Z_Gal = (H_Gal_smooth.T - H_Gal_smooth.T.min())/(H_Gal_smooth.T.max() - H_Gal_smooth.T.min())

# fix probable lines not closing
Z_AGN, x_centers_AGN, y_centers_AGN = gf.pad_matrix_zeros(Z_AGN, xedges_AGN, yedges_AGN)
Z_Gal, x_centers_Gal, y_centers_Gal = gf.pad_matrix_zeros(Z_Gal, xedges_Gal, yedges_Gal)

CS_AGN_f = ax1.contourf(x_centers_AGN, y_centers_AGN, Z_AGN, levels=sigmas_perc_inv, 
                            colors=cm_gradient_AGN(seq_fill[::-1]), extend='max', 
                            alpha=0.2, antialiased=True, zorder=3)
CS_AGN = ax1.contour(x_centers_AGN, y_centers_AGN, Z_AGN, levels=sigmas_perc_inv, 
                         colors=cm_gradient_AGN(seq_cont[::-1]), linewidths=3.5, zorder=4)
for count in np.arange(0):  # times to run next command, more labels
    labels_AGN = ax1.clabel(CS_AGN, CS_AGN.levels, inline=True, fmt=gf.fmt, 
                            fontsize=10, inline_spacing=-8)
    [txt.set_bbox(dict(boxstyle='square, pad=0', fc='None', ec='None')) for txt in labels_AGN]

CS_Gal_f = ax1.contourf(x_centers_Gal, y_centers_Gal, Z_Gal, levels=sigmas_perc_inv, 
                            colors=cm_gradient_Gal(seq_fill[::-1]), extend='max', 
                            alpha=0.2, antialiased=True, zorder=3)
CS_Gal = ax1.contour(x_centers_Gal, y_centers_Gal, Z_Gal, levels=sigmas_perc_inv, 
                         colors=cm_gradient_Gal(seq_cont[::-1]), linewidths=3.5, zorder=4)
for count in np.arange(0):  # times to run next command, more labels
    labels_Gal = ax1.clabel(CS_Gal, CS_Gal.levels, inline=True, fmt=gf.fmt, 
                            fontsize=10, inline_spacing=-8)
    [txt.set_bbox(dict(boxstyle='square, pad=0', fc='None', ec='None')) for txt in labels_Gal]

base_dens, = ax1.plot([-3], [-3], marker='s', ls='None', c=plt.get_cmap(gv.cmap_dens_plots)(1.1), 
                      label=rf'$\mathrm{{CW ~ - ~ N}} = {n_sources_CW:,}$'.replace(',','$\,$'),
                      zorder=0, ms=8)
cont_SFG = ax1.scatter([-1], [10], marker='o', edgecolor=gf.colour_SFG, 
               color=gf.colour_SFG_shade, s=80, 
               label=rf'$\mathrm{{SDSS ~ Gal ~ - ~ N}} =      {np.sum(filter_used_data & filter_gal_HETDEX):,}$'.replace(',','$\,$'), linewidths=2.5)
cont_AGN = ax1.scatter([-1], [10], marker='o', 
               edgecolor=gf.colour_AGN, color=gf.colour_AGN_shade, s=80, 
               label=rf'$\mathrm{{MQC ~ AGN ~ - ~ N}} =      {np.sum(filter_used_data & filter_AGN_HETDEX):,}$'.replace(',','$\,$'), linewidths=2.5)

# Identify and plot points outside contours
p_AGN = CS_AGN.collections[0].get_paths()
p_Gal = CS_Gal.collections[0].get_paths()
inside_AGN = np.full_like(catalog_HETDEX_df.loc[filter_used_data & filter_AGN_HETDEX, 'g_r'], False, dtype=bool)
inside_Gal = np.full_like(catalog_HETDEX_df.loc[filter_used_data & filter_gal_HETDEX, 'g_r'], False, dtype=bool)
for level in p_AGN:
    inside_AGN |= level.contains_points(catalog_HETDEX_df.loc[filter_used_data & filter_AGN_HETDEX, ['g_r', 'W1_W2']])
for level in p_Gal:
    inside_Gal |= level.contains_points(catalog_HETDEX_df.loc[filter_used_data & filter_gal_HETDEX, ['g_r', 'W1_W2']])
out_AGN, = ax1.plot(catalog_HETDEX_df.loc[filter_used_data & filter_AGN_HETDEX].loc[~inside_AGN, 'g_r'], 
                    catalog_HETDEX_df.loc[filter_used_data & filter_AGN_HETDEX].loc[~inside_AGN, 'W1_W2'],
                     marker='x', ls='None', color=gf.colour_AGN, zorder=2, alpha=0.8)
out_SFG, = ax1.plot(catalog_HETDEX_df.loc[filter_used_data & filter_gal_HETDEX].loc[~inside_Gal, 'g_r'], 
                    catalog_HETDEX_df.loc[filter_used_data & filter_gal_HETDEX].loc[~inside_Gal, 'W1_W2'],
                    marker='x', ls='None', color=gf.colour_SFG, zorder=2, alpha=0.8)

# Colorbar density
axins0 = inset_axes(ax1, width='100%', height='100%', bbox_transform=ax1.transAxes,
                    loc=1, bbox_to_anchor=(0.9, 0.15, 0.05, 0.80), borderpad=0)
clb_dens    = fig.colorbar(dens_CW_HETDEX, cax=axins0, orientation='vertical', 
                           cmap=plt.get_cmap(gv.cmap_dens_plots, num_levels_dens), 
                           norm=norm_dens, extend='max', 
                           format=lambda x, pos: rf"${x:,.0f}$".replace(",", "$\,$"))
axins0.yaxis.set_ticks_position('left')
clb_dens.ax.tick_params(labelsize=20)
clb_dens.outline.set_linewidth(2.5)
# clb_dens.ax.set_ylabel('Elements per bin', size=12, path_effects=pe2)

x_Vega   = np.array(AB_lims_x) - vega_shift['gmag']     + vega_shift['rmag']      # Vega
y_Vega   = np.array(AB_lims_y) - vega_shift['W1mproPM'] + vega_shift['W2mproPM']  # Vega
x_Vega_shift = vega_shift['gmag'] - vega_shift['rmag']
y_Vega_shift = vega_shift['W1mproPM'] - vega_shift['W2mproPM']
# New criterion by Carvajal+
points_C23 = np.array([[-0.76, -0.76, 1.8, 1.8], [y_Vega[-1], 0.26, 0.84, y_Vega[-1]]])
points_C23_AB = np.array([[-0.76 + x_Vega_shift, -0.76 + x_Vega_shift, 1.8 + x_Vega_shift, 1.8 + x_Vega_shift], [np.array(AB_lims_y)[-1], 0.26 + y_Vega_shift, 0.84 + y_Vega_shift, np.array(AB_lims_y)[-1]]])
points_C23_AB_constrained = [[-0.6, -0.6, 1.2, 1.2], [np.array(AB_lims_y)[-1], 0.0, 0.3, np.array(AB_lims_y)[-1]]]
C23_plot, = ax1.plot(points_C23[0] + vega_shift['gmag'] - vega_shift['rmag'], 
        points_C23[1] + vega_shift['W1mproPM'] - vega_shift['W2mproPM'], 
        label=r'$\mathrm{This ~ work}$', c=plt.get_cmap(gv.cmap_bands)(0.75), zorder=6, lw=3.5)
# ax1.plot(points_C23_AB_constrained[0], 
#          points_C23_AB_constrained[1], 
#          label=r'$\mathrm{This ~ work}$', c=plt.get_cmap(gv.cmap_bands)(0.75), zorder=2, lw=3.5)
ax1.set_xlim(AB_lims_x)
ax1.set_ylim(AB_lims_y)

ax2 = ax1.twinx()
ax2.set_ylim(tuple(np.array(ax1.get_ylim()) - 2.699 + 3.339))
ax2.tick_params(which='both', top=False, right=True, direction='in')
ax2.tick_params(which='both', bottom=False, left=False, direction='in')
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(which='major', length=8, width=1.5)
ax2.tick_params(which='minor', length=4, width=1.5)
ax2.set_ylabel('$m_{\mathrm{W1}} - m_{\mathrm{W2}}\, \mathrm{[Vega]}$', size=28)

ax3 = ax1.twiny()
ax3.set_xlim(tuple(np.array(ax1.get_xlim()) - 0.481 + 0.617))
ax3.tick_params(which='both', top=True, right=False, direction='in')
ax3.tick_params(which='both', bottom=False, left=False, direction='in')
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.tick_params(which='major', length=8, width=1.5)
ax3.tick_params(which='minor', length=4, width=1.5)
ax3.set_xlabel('$m_{\mathrm{g}} - m_{\mathrm{r}}\, \mathrm{[Vega]}$', size=28, labelpad=10)

ax1.tick_params(which='both', top=False, right=False, direction='in')
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.tick_params(which='major', length=8, width=1.5)
ax1.tick_params(which='minor', length=4, width=1.5)
ax1.set_xlabel('$m_{\mathrm{g}} - m_{\mathrm{r}}\, \mathrm{[AB]}$', size=28)
ax1.set_ylabel('$m_{\mathrm{W1}} - m_{\mathrm{W2}}\, \mathrm{[AB]}$', size=28)
plt.setp(ax1.spines.values(), linewidth=3.5)
plt.setp(ax1.spines.values(), linewidth=3.5)
# ax1.legend(loc=2, fontsize=20, ncol=1, columnspacing=.5, handletextpad=0.2, handlelength=0.8, framealpha=0.75)
arr_legends = [rf'$\mathrm{{CW ~ - ~ N}} = {n_sources_CW:,}$'.replace(',','$\,$'), rf'$\mathrm{{MQC ~ AGN ~ - ~ N}} =      {np.sum(filter_used_data & filter_AGN_HETDEX):,}$'.replace(',','$\,$'), rf'$\mathrm{{SDSS ~ Gal ~ - ~ N}} =      {np.sum(filter_used_data & filter_gal_HETDEX):,}$'.replace(',','$\,$'), '$\mathrm{This ~ work}$']
ax1.legend([base_dens, (cont_AGN, out_AGN), (cont_SFG, out_SFG), C23_plot], arr_legends, scatterpoints=1,
           numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc=2, fontsize=18,
           ncol=1, columnspacing=.5, handletextpad=0.2, handlelength=0.8, framealpha=0.75).set_zorder(10)
# plt.tight_layout()
plt.savefig(paths.figures / 'g_r_W1_W2_AGN_gal_HETDEX_nonimputed.pdf', bbox_inches='tight')
