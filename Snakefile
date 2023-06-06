rule flowchart_HETDEX:
    input:
        "src/data/indices_known.txt",
        "src/data/indices_train.txt",
        "src/data/indices_test.txt",
        "src/data/indices_train_validation.txt",
        "src/data/indices_calibration.txt",
        "src/data/indices_validation.txt"
    output:
        "src/tex/figures/flowchart_HETDEX_subsets.pdf"
    script:
        "src/scripts/fig_flowchart_HETDEX.py"

rule flowchart_S82:
    input:
        "src/data/S82_for_prediction.parquet"
    output:
        "src/tex/figures/flowchart_S82_subsets.pdf"
    script:
        "src/scripts/fig_flowchart_Stripe82.py"

rule conf_mat_AGN_val:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_test.txt",
        "src/scripts/global_functions.py"
    output:
        "src/tex/figures/conf_matrix_AGN_HETDEX_test.pdf"
    script:
        "src/scripts/fig_conf_matrix_AGN_gal_test.py"

rule conf_mat_radio_val:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_test.txt",
        "src/scripts/global_functions.py"
    output:
        "src/tex/figures/conf_matrix_radio_HETDEX_test.pdf"
    script:
        "src/scripts/fig_conf_matrix_radio_test.py"

rule redshift_compare_val:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_test.txt",
        "src/scripts/global_functions.py"
    output:
        "src/tex/figures/compare_redshift_HETDEX_test.pdf"
    script:
        "src/scripts/fig_redshift_compare_test.py"

rule conf_mat_radio_AGN_HETDEX:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_test.txt",
        "src/scripts/global_functions.py"
    output:
        "src/tex/figures/conf_matrix_rAGN_HETDEX_test.pdf"
    script:
        "src/scripts/fig_conf_matrix_radio_AGN_test.py"

rule redshift_compare_HETDEX:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_test.txt",
        "src/scripts/global_functions.py"
    output:
        "src/tex/figures/compare_redshift_rAGN_HETDEX_test.pdf"
    script:
        "src/scripts/fig_redshift_compare_rAGN_test.py"

rule conf_mat_radio_AGN_S82:
    input:
        "src/data/S82_for_prediction.parquet",
        "src/scripts/global_functions.py"
    output:
        "src/tex/figures/conf_matrix_rAGN_Stripe82.pdf"
    script:
        "src/scripts/fig_conf_matrix_radio_AGN_Stripe_82.py"

rule redshift_compare_S82:
    input:
        "src/data/S82_for_prediction.parquet",
        "src/scripts/global_functions.py"
    output:
        "src/tex/figures/compare_redshift_rAGN_Stripe82.pdf"
    script:
        "src/scripts/fig_redshift_compare_rAGN_Stripe_82.py"

rule calibration_AGN_pre:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_train_validation.txt",
        "src/data/indices_calibration.txt",
        "src/data/indices_test.txt",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/calib_curves_pre_calib_AGN_galaxy.pdf"
    script:
        "src/scripts/fig_calibration_AGN_gal_pre.py"

rule calibration_radio_pre:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_train_validation.txt",
        "src/data/indices_calibration.txt",
        "src/data/indices_test.txt",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/calib_curves_pre_calib_radio.pdf"
    script:
        "src/scripts/fig_calibration_radio_pre.py"

rule calibration_AGN_post:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_train_validation.txt",
        "src/data/indices_calibration.txt",
        "src/data/indices_test.txt",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/calib_curves_post_calib_AGN_galaxy.pdf"
    script:
        "src/scripts/fig_calibration_AGN_gal_post.py"

rule calibration_radio_post:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_train_validation.txt",
        "src/data/indices_calibration.txt",
        "src/data/indices_test.txt",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/calib_curves_post_calib_radio.pdf"
    script:
        "src/scripts/fig_calibration_radio_post.py"

rule PR_curve_AGN:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_train.txt",
        "src/data/indices_test.txt",
        "src/data/indices_train_validation.txt",
        "src/data/indices_calibration.txt",
        "src/data/indices_validation.txt"
    output:
        "src/tex/figures/PR_cal_curve_AGN_gal.pdf"
    script:
        "src/scripts/fig_PR_curve_AGN_gal.py"

rule PR_curve_radio:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/indices_train.txt",
        "src/data/indices_test.txt",
        "src/data/indices_train_validation.txt",
        "src/data/indices_calibration.txt",
        "src/data/indices_validation.txt"
    output:
        "src/tex/figures/PR_cal_curve_radio.pdf"
    script:
        "src/scripts/fig_PR_curve_radio.py"

rule decision_AGN_xgboost:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/classification_AGN_galaxy_dec_18_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_AGN_base_xgboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_AGN_gal_hiz_AGN_xgboost.py"

rule decision_AGN_rf:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/classification_AGN_galaxy_dec_18_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_AGN_base_rf_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_AGN_gal_hiz_AGN_rf.py"

rule decision_AGN_et:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/classification_AGN_galaxy_dec_18_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_AGN_base_et_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_AGN_gal_hiz_AGN_et.py"

rule decision_AGN_gbc:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/classification_AGN_galaxy_dec_18_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_AGN_base_gbc_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_AGN_gal_hiz_AGN_gbc.py"

rule decision_radio_xgboost:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/classification_LOFAR_detect_dec_19_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_radio_base_xgboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_radio_hiz_AGN_xgboost.py"

rule decision_radio_catboost:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/classification_LOFAR_detect_dec_19_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_radio_base_catboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_radio_hiz_AGN_catboost.py"

rule decision_radio_rf:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/classification_LOFAR_detect_dec_19_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_radio_base_rf_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_radio_hiz_AGN_rf.py"

rule decision_radio_et:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/classification_LOFAR_detect_dec_19_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_radio_base_et_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_radio_hiz_AGN_et.py"

rule decision_redshift_rf:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/regression_z_dec_20_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_z_base_rf_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_z_hiz_AGN_rf.py"

rule decision_redshift_catboost:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/regression_z_dec_20_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_z_base_catboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_z_hiz_AGN_catboost.py"

rule decision_redshift_xgboost:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/regression_z_dec_20_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_z_base_xgboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_z_hiz_AGN_xgboost.py"

rule decision_redshift_gbr:
    input:
        "src/data/HETDEX_for_prediction.parquet",
        "src/data/models/regression_z_dec_20_2022.pkl",
        "src/scripts/global_functions.py",
        "src/scripts/global_variables.py"
    output:
        "src/tex/figures/SHAP/SHAP_decision_z_base_gbr_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_z_hiz_AGN_gbr.py"