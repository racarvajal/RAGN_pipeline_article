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

file_name_HETDEX  = paths.data / 'HETDEX_for_prediction.parquet'
model_AGN_name    = paths.models / gv.AGN_gal_model
AGN_gal_clf       = pyc.load_model(model_AGN_name, verbose=False)

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

base_models_AGN   = gf.get_base_estimators_names(AGN_gal_clf)
reduced_data_df   = gf.preprocess_data(AGN_gal_clf, 
                                       catalog_HETDEX_df.drop(columns=['class', 'Z', 'LOFAR_detect']),
                                                              base_models_AGN)
reduced_cols      = reduced_data_df.columns

base_logit_AGN    = np.log(gv.AGN_thresh / (1 - gv.AGN_thresh))

explainer_AGN     = fasttreeshap.TreeExplainer(AGN_gal_clf.named_steps['trained_model'].final_estimator_,
                                               data=None, feature_perturbation='tree_path_dependent',
                                               model_output='raw', feature_dependence='independent',
                                               algorithm='auto', n_jobs=12)
shap_values_AGN   = explainer_AGN(reduced_data_df, check_additivity=False)

xlims_plt         = (0.49987, 0.50013)
size_side         = 8
fig               = plt.figure(figsize=(size_side,size_side * 3/2))
ax1               = fig.add_subplot(111, xscale='linear', yscale='linear')
_ = gf.plot_shap_decision('AGN/Galaxy class', 'CatBoost', shap_values_AGN, explainer_AGN, 
                          reduced_cols, ax1, 'logit', new_base_value=base_logit_AGN, 
                          base_meta='Meta', xlim=xlims_plt, highlight=filter_radio)
plt.savefig(paths.figures / 'SHAP/SHAP_decision_AGN_meta_learner_HETDEX_highz.pdf', bbox_inches='tight')
