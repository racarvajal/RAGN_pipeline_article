#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
from pycaret import classification as pyc
import pandas as pd
import fasttreeshap
import paths
import global_variables as gv
import global_functions as gf
import os
from pathlib import Path
os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX  = paths.data / 'HETDEX_for_prediction.parquet'
model_AGN_name    = paths.models / gv.AGN_gal_model
AGN_gal_full      = pyc.load_model(model_AGN_name, verbose=False)
AGN_gal_clf       = AGN_gal_full.named_steps['trained_model'].estimators_[1]

feats_2_use       = ['ID', 'class', 'LOFAR_detect', 'Z', 
                     'band_num', 'W4mag', 'g_r', 'r_i', 
                     'r_J', 'i_z', 'i_y',  'z_y', 'z_W2', 
                     'y_J', 'y_W1', 'y_W2', 'J_H', 'H_K', 
                     'H_W3', 'W1_W2', 'W1_W3', 'W3_W4']

catalog_HETDEX_df = pd.read_parquet(file_name_HETDEX, engine='fastparquet', columns=feats_2_use)
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

filter_hiz        = np.array(catalog_HETDEX_df.loc[:, 'Z'] >= 4.0) &\
                    np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
catalog_HETDEX_df = catalog_HETDEX_df.loc[filter_hiz]
filter_radio      = np.array(catalog_HETDEX_df.loc[:, 'LOFAR_detect'] == 1)

base_models_AGN   = gf.get_base_estimators_names(AGN_gal_full)
reduced_data_df   = gf.preprocess_data(AGN_gal_full, 
                                       catalog_HETDEX_df.drop(columns=['class', 'Z', 'LOFAR_detect']),
                                                              base_models_AGN)
reduced_cols      = reduced_data_df.columns.drop(base_models_AGN)

base_logit_AGN    = np.log(0.5 / (1 - 0.5))

explainer_AGN     = fasttreeshap.TreeExplainer(AGN_gal_clf,
                                               data=None, feature_perturbation='tree_path_dependent',
                                               model_output='raw', feature_dependence='independent',
                                               algorithm='auto', n_jobs=12)
shap_values_AGN   = explainer_AGN(reduced_data_df.drop(columns=base_models_AGN), check_additivity=False)

size_side         = 8
fig               = plt.figure(figsize=(size_side,size_side * 3/2))
ax1               = fig.add_subplot(111, xscale='linear', yscale='linear')
_ = gf.plot_shap_decision('AGN/Galaxy class', 'rf', shap_values_AGN, explainer_AGN, 
                          reduced_cols, ax1, 'identity', new_base_value=0.5, 
                          base_meta='Base', highlight=filter_radio)
plt.savefig(paths.figures / 'SHAP/SHAP_decision_AGN_base_rf_HETDEX_highz.pdf', bbox_inches='tight')
