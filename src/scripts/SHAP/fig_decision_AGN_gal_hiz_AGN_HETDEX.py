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
model_AGN_name    = paths.data / 'models' / gv.AGN_gal_model
AGN_gal_clf       = pyc.load_model(model_AGN_name, verbose=False)

feats_2_use       = ['ID', 'class', 'Z', 'W4mag', 'g_r',
                     'g_J', 'r_i', 'r_z', 'r_W1', 
                     'i_z', 'i_y', 'z_y', 'y_J', 'y_W2', 
                     'J_H', 'H_K', 'K_W3', 'W1_W2', 'W1_W3']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

filter_hiz        = np.array(catalog_HETDEX_df.loc[:, 'Z'] >= 4.0) &\
                    np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
catalog_HETDEX_df = catalog_HETDEX_df.loc[filter_hiz]

base_models_AGN   = gf.get_base_estimators_names(AGN_gal_clf)
reduced_data_df   = gf.preprocess_data(AGN_gal_clf, 
                                       catalog_HETDEX_df.drop(columns=['class', 'Z']),
                                                              base_models_AGN)
reduced_cols      = reduced_data_df.columns

base_logit_AGN    = np.log(gv.AGN_thresh / (1 - gv.AGN_thresh))

explainer_AGN     = fasttreeshap.TreeExplainer(AGN_gal_clf.named_steps['trained_model'].final_estimator_,\
                                               data=None, feature_perturbation='tree_path_dependent',\
                                               model_output='raw', feature_dependence='independent',\
                                               algorithm='auto', n_jobs=12)
shap_values_AGN   = explainer_AGN(reduced_data_df, check_additivity=False)

size_side         = 8
fig               = plt.figure(figsize=(size_side,size_side * 3/2))
ax1               = fig.add_subplot(111, xscale='linear', yscale='linear')
_ = gf.plot_shap_decision('AGN/Galaxy class', 'CatBoost', shap_values_AGN, explainer_AGN,\
                          reduced_cols, ax1, 'logit', new_base_value=base_logit_AGN, base_meta='Meta')
plt.savefig(paths.figures / 'SHAP/SHAP_decision_AGN_meta_learner_HETDEX_highz.pdf', bbox_inches='tight')
