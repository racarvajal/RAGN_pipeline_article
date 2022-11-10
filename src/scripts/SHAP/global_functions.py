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
import matplotlib.patheffects as mpe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import global_variables as gv

##########################################
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

# Plot SHAP decision
def plot_shap_decision(pred_type, model_name, shap_values, shap_explainer,
                       col_names, ax, link, cmap=gv.cmap_shap, 
                       new_base_value=None, base_meta='', xlim=None, highlight=None):
    if np.ndim(shap_values.values) == 2:
        shap.plots.decision(base_value=shap_explainer.expected_value,
                            shap_values=shap_values.values,
                            feature_names=col_names.to_list(),
                            link=link, plot_color=plt.get_cmap(cmap),
                            highlight=highlight, auto_size_plot=False,
                            show=False, xlim=xlim,
                            feature_display_range=slice(-1, -(len(shap_values.feature_names) +1), -1),
                            new_base_value=new_base_value)
    if np.ndim(shap_values.values) > 2:
        shap.plots.decision(base_value=shap_explainer.expected_value[-1],
                            shap_values=(shap_values.values)[:, :, 1],
                            feature_names=col_names.to_list(),
                            link=link, plot_color=plt.get_cmap(cmap),
                            highlight=highlight, auto_size_plot=False,
                            show=False, xlim=None,
                            feature_display_range=slice(-1, -(len(shap_values.feature_names) +1), -1),
                            new_base_value=new_base_value)
    ax.tick_params('x', labelsize=38)
    ax.xaxis.get_offset_text().set_fontsize(28)
    #ax1.xaxis.get_offset_text().set_position((0,1))
    ax.tick_params('y', labelsize=38)
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.xaxis.label.set_size(38)
    plt.title(f'{pred_type}: {base_meta}-learner - {model_name}', fontsize=38)
    plt.tight_layout()
    return ax