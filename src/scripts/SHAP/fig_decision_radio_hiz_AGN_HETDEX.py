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
radio_det_clf     = pyc.load_model(model_radio_name, verbose=False)

feats_2_use       = ['ID', 'class', 'LOFAR_detect', 'Z', 
                     'band_num', 'W4mag', 'g_r', 'g_i', 
                     'r_i', 'r_z', 'i_z', 'z_y', 'z_W1', 
                     'y_J', 'y_W1', 'J_H', 'H_K', 'K_W3', 
                     'K_W4', 'W1_W2', 'W2_W3']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

filter_hiz        = np.array(catalog_HETDEX_df.loc[:, 'Z'] >= 4.0) &\
                    np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
catalog_HETDEX_df = catalog_HETDEX_df.loc[filter_hiz]
filter_radio      = np.array(catalog_HETDEX_df.loc[:, 'LOFAR_detect'] == 1)

base_models_radio = gf.get_base_estimators_names(radio_det_clf)
reduced_data_df   = gf.preprocess_data(radio_det_clf, 
                                       catalog_HETDEX_df.drop(columns=['class', 'Z', 'LOFAR_detect']),
                                                              base_models_radio)
reduced_cols      = reduced_data_df.columns

base_logit_radio  = np.log(gv.radio_thresh / (1 - gv.radio_thresh))

explainer_radio   = fasttreeshap.TreeExplainer(radio_det_clf.named_steps['trained_model'].final_estimator_,\
                                               data=None, feature_perturbation='tree_path_dependent',\
                                               model_output='raw', feature_dependence='independent',\
                                               algorithm='auto', n_jobs=12)
shap_values_radio = explainer_radio(reduced_data_df, check_additivity=False)

xlims_plt         = (0.4999, 0.5001)
size_side         = 8
fig               = plt.figure(figsize=(size_side,size_side * 3/2))
ax1               = fig.add_subplot(111, xscale='linear', yscale='linear')
_ = gf.plot_shap_decision('Radio detection', 'GradientBoosting', shap_values_radio, explainer_radio, 
                          reduced_cols, ax1, 'logit', new_base_value=base_logit_radio, 
                          base_meta='Meta', highlight=filter_radio)
plt.savefig(paths.figures / 'SHAP/SHAP_decision_radio_meta_learner_HETDEX_highz.pdf', bbox_inches='tight')
