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

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX  = paths.data / 'HETDEX_for_prediction.h5'
model_radio_name  = paths.data / 'models' / gv.radio_model
radio_det_full    = pyc.load_model(model_radio_name, verbose=False)
radio_det_clf     = radio_det_full.named_steps['trained_model'].estimators_[0]

feats_2_use       = ['ID', 'class', 'LOFAR_detect', 'Z', 'band_num', 
                     'W4mag', 'g_r', 'r_i', 'r_z', 'i_z', 'i_y', 
                     'z_y', 'z_W1', 'y_J', 'y_W1', 'J_H', 'H_K', 
                     'K_W3', 'K_W4', 'W1_W2', 'W2_W3']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

filter_hiz        = np.array(catalog_HETDEX_df.loc[:, 'Z'] >= 4.0) &\
                    np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
catalog_HETDEX_df = catalog_HETDEX_df.loc[filter_hiz]
filter_radio      = np.array(catalog_HETDEX_df.loc[:, 'LOFAR_detect'] == 1)

base_models_radio = gf.get_base_estimators_names(radio_det_full)
reduced_data_df   = gf.preprocess_data(radio_det_full, 
                                       catalog_HETDEX_df.drop(columns=['class', 'Z', 'LOFAR_detect']),
                                                              base_models_radio)
reduced_cols      = reduced_data_df.columns.drop(base_models_radio)

base_logit_radio  = np.log(0.5 / (1 - 0.5))

explainer_radio   = fasttreeshap.TreeExplainer(radio_det_clf,\
                                               data=None, feature_perturbation='tree_path_dependent',\
                                               model_output='raw', feature_dependence='independent',\
                                               algorithm='auto', n_jobs=12)
shap_values_radio = explainer_radio(reduced_data_df.drop(columns=base_models_radio), check_additivity=False)

#xlims_plt         = (0.4999, 0.5001)
size_side         = 8
fig               = plt.figure(figsize=(size_side,size_side * 3/2))
ax1               = fig.add_subplot(111, xscale='linear', yscale='linear')
_ = gf.plot_shap_decision('Radio detection', 'xgboost', shap_values_radio, explainer_radio, 
                          reduced_cols, ax1, 'logit', new_base_value=base_logit_radio, 
                          base_meta='Base', highlight=filter_radio)
plt.savefig(paths.figures / 'SHAP/SHAP_decision_radio_base_xgboost_HETDEX_highz.pdf', bbox_inches='tight')
