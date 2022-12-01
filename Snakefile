rule flowchart_HETDEX:
    output:
        "src/tex/figures/flowchart_HETDEX_subsets.pdf"
    script:
        "src/scripts/fig_flowchart_HETDEX.py"

rule flowchart_S82:
    output:
        "src/tex/figures/flowchart_S82_subsets.pdf"
    script:
        "src/scripts/fig_flowchart_Stripe82.py"

rule conf_mat_AGN_val:
    output:
        "src/tex/figures/conf_matrix_AGN_HETDEX_test.pdf"
    script:
        "src/scripts/fig_conf_matrix_AGN_gal_test.py"

rule conf_mat_radio_val:
    output:
        "src/tex/figures/conf_matrix_radio_HETDEX_test.pdf"
    script:
        "src/scripts/fig_conf_matrix_radio_test.py"

rule redshift_compare_val:
    output:
        "src/tex/figures/compare_redshift_HETDEX_test.pdf"
    script:
        "src/scripts/fig_redshift_compare_test.py"

rule conf_mat_radio_AGN_HETDEX:
    output:
        "src/tex/figures/conf_matrix_rAGN_HETDEX_test.pdf"
    script:
        "src/scripts/fig_conf_matrix_radio_AGN_test.py"

rule redshift_compare_HETDEX:
    output:
        "src/tex/figures/compare_redshift_rAGN_HETDEX_test.pdf"
    script:
        "src/scripts/fig_redshift_compare_rAGN_test.py"

rule conf_mat_radio_AGN_S82:
    output:
        "src/tex/figures/conf_matrix_rAGN_Stripe82.pdf"
    script:
        "src/scripts/fig_conf_matrix_radio_AGN_Stripe_82.py"

rule redshift_compare_S82:
    output:
        "src/tex/figures/compare_redshift_rAGN_Stripe82.pdf"
    script:
        "src/scripts/fig_redshift_compare_rAGN_Stripe_82.py"

rule calibration_AGN_pre:
    output:
        "src/tex/figures/calib_curves_pre_calib_AGN_galaxy.pdf"
    script:
        "src/scripts/fig_calibration_AGN_gal_pre.py"

rule calibration_radio_pre:
    output:
        "src/tex/figures/calib_curves_pre_calib_radio.pdf"
    script:
        "src/scripts/fig_calibration_radio_pre.py"

rule calibration_AGN_post:
    output:
        "src/tex/figures/calib_curves_post_calib_AGN_galaxy.pdf"
    script:
        "src/scripts/fig_calibration_AGN_gal_post.py"

rule calibration_radio_post:
    output:
        "src/tex/figures/calib_curves_post_calib_radio.pdf"
    script:
        "src/scripts/fig_calibration_radio_post.py"

rule PR_curve_AGN:
    output:
        "src/tex/figures/PR_cal_curve_AGN_gal.pdf"
    script:
        "src/scripts/fig_PR_curve_AGN_gal.py"

rule PR_curve_radio:
    output:
        "src/tex/figures/PR_cal_curve_radio.pdf"
    script:
        "src/scripts/fig_PR_curve_radio.py"

rule decision_AGN_rf:
    output:
        "src/tex/figures/SHAP/SHAP_decision_AGN_base_rf_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_AGN_gal_hiz_AGN_rf.py"

rule decision_AGN_gbc:
    output:
        "src/tex/figures/SHAP/SHAP_decision_AGN_base_gbc_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_AGN_gal_hiz_AGN_gbc.py"

rule decision_AGN_et:
    output:
        "src/tex/figures/SHAP/SHAP_decision_AGN_base_et_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_AGN_gal_hiz_AGN_et.py"

rule decision_AGN_xgboost:
    output:
        "src/tex/figures/SHAP/SHAP_decision_AGN_base_xgboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_AGN_gal_hiz_AGN_xgboost.py"

rule decision_radio_xgboost:
    output:
        "src/tex/figures/SHAP/SHAP_decision_radio_base_xgboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_radio_hiz_AGN_xgboost.py"

rule decision_radio_et:
    output:
        "src/tex/figures/SHAP/SHAP_decision_radio_base_et_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_radio_hiz_AGN_et.py"

rule decision_radio_rf:
    output:
        "src/tex/figures/SHAP/SHAP_decision_radio_base_rf_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_radio_hiz_AGN_rf.py"

rule decision_radio_gbc:
    output:
        "src/tex/figures/SHAP/SHAP_decision_radio_base_gbc_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_radio_hiz_AGN_gbc.py"

rule decision_redshift_xgboost:
    output:
        "src/tex/figures/SHAP/SHAP_decision_z_base_xgboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_z_hiz_AGN_xgboost.py"

rule decision_redshift_gbr:
    output:
        "src/tex/figures/SHAP/SHAP_decision_z_base_gbr_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_z_hiz_AGN_gbr.py"

rule decision_redshift_catboost:
    output:
        "src/tex/figures/SHAP/SHAP_decision_z_base_catboost_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_z_hiz_AGN_catboost.py"

rule decision_redshift_et:
    output:
        "src/tex/figures/SHAP/SHAP_decision_z_base_et_HETDEX_highz.pdf"
    script:
        "src/scripts/SHAP/fig_decision_z_hiz_AGN_et.py"