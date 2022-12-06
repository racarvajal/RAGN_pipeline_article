#!/usr/bin/env python

# File with most used
# functions and derived
# variables.

# Initial imports
import numpy as np
import pandas as pd
import shap
import sklearn.pipeline as skp
from sklearn.metrics import ConfusionMatrixDisplay
from astropy.visualization import LogStretch, PowerStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from pycaret import classification as pyc
from pycaret import regression as pyr
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import global_variables as gv

##########################################
##########################################
# Obtain classification metrics from confusion matrices
def conf_mat_func(true_class_arr, predicted_class_arr):
    cm = np.array([[np.sum(np.array(true_class_arr == 0) & np.array(predicted_class_arr == 0)),\
                    np.sum(np.array(true_class_arr == 0) & np.array(predicted_class_arr == 1))],\
                   [np.sum(np.array(true_class_arr == 1) & np.array(predicted_class_arr == 0)),\
                    np.sum(np.array(true_class_arr == 1) & np.array(predicted_class_arr == 1))]])
    return cm

def flatten_CM(cm_array, **kwargs):
    try:
        TN, FP, FN, TP = cm_array.flatten().astype('float32')
    except:
        TN, FP, FN, TP = cm_array.flatten()
    return TN, FP, FN, TP

# Recall
def Recall_from_CM(cm_array, **kwargs):
    TN, FP, FN, TP = flatten_CM(cm_array)
    Recall = TP / (TP + FN)
    return Recall
##########################################
# Define metrics for regression
def sigma_nmad(z_true, z_pred, **kwargs):
    dif  = (z_true - z_pred)
    frac = dif / (1 + z_true).values
    try:
        med  = np.nanmedian(np.abs(frac)).astype('float32')
    except:
        med  = np.nanmedian(np.abs(frac))
    return 1.48 * med
##########################################
# Methods using Pycaret pipelines
def get_final_column_names(pycaret_pipeline, sample_df, verbose=False):
    if isinstance(pycaret_pipeline, skp.Pipeline):
        for (name, method) in pycaret_pipeline.named_steps.items():
            if method != 'passthrough' and name != 'trained_model':
                if verbose:
                    print(f'Running {name}')
                sample_df = method.transform(sample_df)
        return sample_df.columns.tolist()
    else:
        try:
            for (name, method) in pyr.get_config('prep_pipe').named_steps.items():
                if method != 'passthrough' and name != 'trained_model':
                    if verbose:
                        print(f'Running {name}')
                    sample_df = method.transform(sample_df)
        except:
            for (name, method) in pyc.get_config('prep_pipe').named_steps.items():
                if method != 'passthrough' and name != 'trained_model':
                    if verbose:
                        print(f'Running {name}')
                    sample_df = method.transform(sample_df)
        return sample_df.columns.tolist()

# Feature importance (or mean of) from meta model (or base models)

def get_base_estimators_names(pycaret_pipeline):
    if isinstance(pycaret_pipeline, skp.Pipeline):
        estimators  = pycaret_pipeline['trained_model'].estimators
    else:
        estimators  = pycaret_pipeline.estimators

    estimators_list = [estimator[0] for estimator in estimators]
    return estimators_list

def get_base_estimators_models(pycaret_pipeline):
    if isinstance(pycaret_pipeline, skp.Pipeline):
        estimators_  = pycaret_pipeline['trained_model'].estimators_
    else:
        estimators_  = pycaret_pipeline.estimators_
    return estimators_

# Run data through previous steps of pipeline
def preprocess_data(pycaret_pipeline, data_df, base_models_names, verbose=False):
    processed_data = data_df.loc[:, get_final_column_names(pycaret_pipeline, data_df)].copy()
    processed_idx_data  = processed_data.index
    # processed_cols_data  = processed_data.columns
    processed_cols_data = processed_data.columns.insert(0, base_models_names[0])
    if len(base_models_names) > 1:
        for est_name in base_models_names[1::]:
            processed_cols_data = processed_cols_data.insert(0, est_name)
    if isinstance(pycaret_pipeline, skp.Pipeline):
        prep_steps = pycaret_pipeline.named_steps.items()
    else:
        prep_steps = pyc.get_config('prep_pipe').named_steps.items()

    for (name, method) in prep_steps:
        if method != 'passthrough':  # and name != 'trained_model':
            if verbose:
                print(f'Running {name}')
            processed_data = method.transform(processed_data)
    processed_data_df = pd.DataFrame(processed_data, columns=processed_cols_data, index=processed_idx_data)
    return processed_data_df

##########################################
# Plotting methods

# Path effects for labels and plots.
pe1            = [mpe.Stroke(linewidth=5.0, foreground='black'),
                  mpe.Stroke(foreground='white', alpha=1),
                  mpe.Normal()]
pe2            = [mpe.Stroke(linewidth=3.0, foreground='white'),
                  mpe.Stroke(foreground='white', alpha=1),
                  mpe.Normal()]

# Plot confusion matrix
def plot_conf_mat(confusion_matrix, axin, display_labels=['0', '1'], title=None, cmap=gv.cmap_conf_matr, show_clb=False, log_stretch=False):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)

    if log_stretch:
        norm = ImageNormalize(stretch=LogStretch())
    if not log_stretch:
        norm = ImageNormalize(stretch=PowerStretch(0.35))

    # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    disp_b = disp.plot(include_values=True, cmap=cm.get_cmap(cmap),\
             ax=axin, xticks_rotation='horizontal', values_format=',')

    for text_val in disp_b.text_.flatten():
        text_val.set_fontsize(28)
        text_val.set_text( text_val.get_text().replace(',','$\,$'))
    clb = plt.gca().images[-1].colorbar
    clb.ax.tick_params(labelsize=26)
    clb.ax.ticklabel_format(style='sci', scilimits=(0, 0))
    clb.outline.set_linewidth(2.5)
    clb.ax.set_ylabel('Elements in bin', size=20)
    if not show_clb:
        clb.remove()

    disp_b.im_.norm = norm

    axin.xaxis.get_label().set_fontsize(26)
    axin.yaxis.get_label().set_fontsize(26)

    axin.tick_params(axis='both', which='major', labelsize=20)

    plt.setp(axin.spines.values(), linewidth=2.5)
    plt.setp(axin.spines.values(), linewidth=2.5)
    axin.set_title(title, fontsize=22)
    plt.tight_layout()
    return axin

# Plot true and estimated/predicted redshifts
def plot_redshift_compare(true_z, predicted_z, ax_pre, title=None, dpi=10, cmap=gv.cmap_z_plots, show_clb=False, log_stretch=False):
    if log_stretch:
        norm = ImageNormalize(vmin=0., stretch=LogStretch())
    if not log_stretch:
        norm = ImageNormalize(vmin=0., stretch=PowerStretch(0.5))

    filt_pair_z   = np.isfinite(true_z) & np.isfinite(predicted_z)
    max_for_range = np.nanmax([np.nanmax(1 + true_z.loc[filt_pair_z]), np.nanmax(1 + predicted_z.loc[filt_pair_z])])

    # Fix colormap to have 0=>white
    cmap_m       = plt.get_cmap(cmap)
    cmap_list    = [cmap_m(i) for i in range(cmap_m.N)]
    cmap_list[0] = (1., 1., 1., 1.)
    cmap_mod     = mcolors.LinearSegmentedColormap.from_list(cmap + '_mod', cmap_list, cmap_m.N)

    dens_1 = ax_pre.scatter_density((1 + true_z.sample(frac=1, random_state=gv.seed)),\
            (1 + predicted_z.sample(frac=1, random_state=gv.seed)),\
            cmap=cmap_mod, zorder=0, dpi=dpi, norm=norm, alpha=0.93)
    
    ax_pre.axline((2., 2.), (3., 3.), ls='--', marker=None, c='Gray', alpha=0.8, lw=3.0, zorder=20)
    ax_pre.axline(xy1=(1., 1.15), xy2=(2., 2.3), ls='-.', marker=None, c='slateblue', alpha=0.6, lw=3.0, zorder=20)
    ax_pre.axline(xy1=(1., 0.85), xy2=(2., 1.7), ls='-.', marker=None, c='slateblue', alpha=0.6, lw=3.0, zorder=20)

    if show_clb:
        clb = plt.colorbar(dens_1, extend='neither', norm=norm, ticks=mtick.MaxNLocator(integer=True))
        clb.ax.tick_params(labelsize=26)
        clb.outline.set_linewidth(2.5)
        clb.ax.set_ylabel('Elements per pixel', size=28, path_effects=pe2)

    # Inset axis with residuals
    axins = inset_axes(ax_pre, width='35%', height='20%', loc=2)
    res_z_z = (predicted_z - true_z) / (1 + true_z)
    axins.hist(res_z_z, histtype='stepfilled', fc='grey', ec='k', bins=50, lw=2.5)
    axins.axvline(x=np.nanpercentile(res_z_z, [15.9]), ls='--', lw=2.5, c='royalblue')
    axins.axvline(x=np.nanpercentile(res_z_z, [84.1]), ls='--', lw=2.5, c='royalblue')
    axins.set_xlabel('$\Delta \mathit{z} / (1 + \mathit{z}_{\mathrm{True}})$', 
                    fontsize=19, path_effects=pe2)
    axins.tick_params(labelleft=False, labelbottom=True)
    axins.tick_params(which='both', top=True, right=True, direction='in')
    axins.tick_params(axis='both', which='major', labelsize=19)
    axins.tick_params(which='major', length=8, width=1.5)
    axins.tick_params(which='minor', length=4, width=1.5)
    plt.setp(axins.spines.values(), linewidth=2.5)
    plt.setp(axins.spines.values(), linewidth=2.5)
    axins.set_xlim(left=-0.9, right=0.9)
    ##
    ax_pre.set_xlabel('$1 + \mathit{z}_{\mathrm{True}}$', fontsize=32)
    ax_pre.set_ylabel('$1 + \mathit{z}_{\mathrm{Predicted}}$', fontsize=32)
    ax_pre.tick_params(which='both', top=True, right=True, direction='in')
    ax_pre.tick_params(axis='both', which='minor', labelsize=24.5)
    ax_pre.tick_params(axis='both', which='major', labelsize=24.5)
    ax_pre.tick_params(which='major', length=8, width=1.5)
    ax_pre.tick_params(which='minor', length=4, width=1.5)
    # ax_pre.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    # ax_pre.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax_pre.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=False))
    ax_pre.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=False))
    ax_pre.xaxis.set_minor_formatter(mtick.ScalarFormatter(useMathText=False))
    ax_pre.yaxis.set_minor_formatter(mtick.ScalarFormatter(useMathText=False))
    plt.setp(ax_pre.spines.values(), linewidth=2.5)
    plt.setp(ax_pre.spines.values(), linewidth=2.5)
    ax_pre.set_xlim(left=1., right=np.ceil(max_for_range))
    ax_pre.set_ylim(bottom=1., top=np.ceil(max_for_range))
    ax_pre.set_title(title, fontsize=22)
    plt.tight_layout()
    return ax_pre

# Plot predicted scores (or Z) vs number of measured bands per source
def plot_scores_band_num(pred_scores, band_num, ax_pre, title=None, dpi=10, cmap=gv.cmap_z_plots, 
                         show_clb=False, log_stretch=False, xlabel=None, ylabel=None, 
                         top_plot=True, bottom_plot=True):
    if log_stretch:
        norm = ImageNormalize(vmin=0., stretch=LogStretch())
    if not log_stretch:
        norm = ImageNormalize(vmin=0., stretch=PowerStretch(0.5))

    filt_vals     = np.isfinite(pred_scores) & np.isfinite(band_num)
    min_x         = np.nanmin(pred_scores.loc[filt_vals])
    min_y         = np.nanmin(band_num.loc[filt_vals])
    max_x         = np.nanmax(pred_scores.loc[filt_vals])
    max_y         = np.nanmax(band_num.loc[filt_vals])

    # Fix colormap to have 0=>white
    cmap_m       = plt.get_cmap(cmap)
    cmap_list    = [cmap_m(i) for i in range(cmap_m.N)]
    cmap_list[0] = (1., 1., 1., 1.)
    cmap_mod     = mcolors.LinearSegmentedColormap.from_list(cmap + '_mod', cmap_list, cmap_m.N)

    dens_1 = ax_pre.scatter_density(pred_scores.sample(frac=1, random_state=gv.seed), 
                                    band_num.sample(frac=1, random_state=gv.seed), 
                                    cmap=cmap_mod, zorder=0, dpi=dpi, norm=norm, alpha=1.0)
    
    # ax_pre.axline((2., 2.), (3., 3.), ls='--', marker=None, c='Gray', alpha=0.8, lw=3.0, zorder=20)
    # ax_pre.axline(xy1=(1., 1.15), xy2=(2., 2.3), ls='-.', marker=None, c='slateblue', alpha=0.6, lw=3.0, zorder=20)
    # ax_pre.axline(xy1=(1., 0.85), xy2=(2., 1.7), ls='-.', marker=None, c='slateblue', alpha=0.6, lw=3.0, zorder=20)

    if show_clb:
        clb = plt.colorbar(dens_1, extend='neither', norm=norm, ax=ax_pre, 
                           ticks=mtick.MaxNLocator(nbins=5, integer=True, prune='lower'), 
                           format=lambda x, pos: f"{x:,.0f}".replace(",", "$\,$"))
        clb.ax.tick_params(labelsize=26)
        clb.outline.set_linewidth(2.5)
        clb.ax.set_ylabel('Sources per pixel', size=28, path_effects=pe2)
    if bottom_plot:
        ax_pre.set_xlabel(xlabel, fontsize=30)
    ax_pre.set_ylabel(ylabel, fontsize=30)
    ax_pre.tick_params(which='both', top=True, right=True, direction='in')
    ax_pre.tick_params(axis='both', which='minor', labelsize=24.5)
    ax_pre.tick_params(axis='both', which='major', labelsize=24.5)
    ax_pre.tick_params(which='major', length=8, width=1.5)
    ax_pre.tick_params(which='minor', length=4, width=1.5)
    # ax_pre.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    # ax_pre.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax_pre.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=False))
    ax_pre.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=False))
    ax_pre.xaxis.set_minor_formatter(mtick.ScalarFormatter(useMathText=False))
    ax_pre.yaxis.set_minor_formatter(mtick.ScalarFormatter(useMathText=False))
    plt.setp(ax_pre.spines.values(), linewidth=2.5)
    plt.setp(ax_pre.spines.values(), linewidth=2.5)
    if top_plot:
        ax_pre.set_xlim(left=np.floor(min_x)-0.5, right=np.ceil(max_x)+0.5)
    ax_pre.set_ylim(bottom=np.floor(min_y)-0.1, top=np.ceil(max_y)+0.1)
    if not bottom_plot:
        ax_pre.xaxis.set_ticklabels([])
    #ax_pre.set_ylim(bottom=-0.1, top=1.1)
    ax_pre.set_title(title, fontsize=22)
    plt.tight_layout()
    return ax_pre