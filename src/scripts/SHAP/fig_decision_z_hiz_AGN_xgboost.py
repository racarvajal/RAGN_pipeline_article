#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
from pycaret import regression as pyr
import pandas as pd
import fasttreeshap
import paths
import global_variables as gv
import global_functions as gf

mpl.rcdefaults()
plt.rcParams['text.usetex'] = True

file_name_HETDEX  = paths.data / 'HETDEX_for_prediction.h5'
model_z_name      = paths.data / 'models' / gv.full_z_model
redshift_reg_full = pyr.load_model(model_z_name, verbose=False)
redshift_reg      = redshift_reg_full.named_steps['trained_model'].estimators_[2]

feats_2_use       = ['ID', 'class', 'LOFAR_detect', 'Z', 'pred_Z',
                     'band_num', 'W4mag', 'g_r', 'g_J', 'r_i', 
                     'r_z', 'i_z', 'i_y', 'z_y', 'y_J', 'y_W1', 
                     'J_H', 'H_K', 'K_W3', 'K_W4', 'W1_W2', 'W1_W3']

catalog_HETDEX_df = pd.read_hdf(file_name_HETDEX, key='df').loc[:, feats_2_use]
catalog_HETDEX_df = catalog_HETDEX_df.set_index(keys=['ID'])

filter_hiz        = np.array(catalog_HETDEX_df.loc[:, 'Z'] >= 4.0) &\
                    np.array(catalog_HETDEX_df.loc[:, 'class'] == 1)
catalog_HETDEX_df = catalog_HETDEX_df.loc[filter_hiz]
filter_radio      = np.array(catalog_HETDEX_df.loc[:, 'LOFAR_detect'] == 1)

base_models_z     = gf.get_base_estimators_names(redshift_reg_full)
reduced_data_df   = gf.preprocess_data(redshift_reg_full, 
                                       catalog_HETDEX_df.drop(columns=['class', 'Z']),
                                                              base_models_z)
reduced_cols      = reduced_data_df.columns.drop(base_models_z)

explainer_z       = fasttreeshap.TreeExplainer(redshift_reg,\
                                               data=None, feature_perturbation='tree_path_dependent',\
                                               model_output='raw', feature_dependence='independent',\
                                               algorithm='auto', n_jobs=12)
shap_values_z     = explainer_z(reduced_data_df.drop(columns=base_models_z), check_additivity=False)

xlims_plt         = (-0.5, catalog_HETDEX_df.loc[reduced_data_df.index, ['Z', 'pred_Z']].max().max() + 0.8)
size_side         = 8
fig               = plt.figure(figsize=(size_side,size_side * 3/2))
ax1               = fig.add_subplot(111, xscale='linear', yscale='linear')
_ = gf.plot_shap_decision('Redshift prediction', 'xgboost', shap_values_z, explainer_z, 
                          reduced_cols, ax1, 'identity', new_base_value=0.0, 
                          base_meta='Base', xlim=xlims_plt, highlight=filter_radio)
plt.savefig(paths.figures / 'SHAP/SHAP_decision_z_base_xgboost_HETDEX_highz.pdf', bbox_inches='tight')
