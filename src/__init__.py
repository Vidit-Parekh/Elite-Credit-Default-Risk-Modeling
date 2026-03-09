"""
src — Credit Default Risk Modeling
===================================
Reusable modules for all notebooks and the Streamlit app.

Modules
-------
preprocessing   Column renaming, target encoding, feature group constants
woe_iv          Weight of Evidence / Information Value — from scratch
modeling        Logistic Regression (statsmodels), Random Forest, XGBoost + Optuna
evaluation      KS, Gini, ROC-AUC, Brier, calibration, Hosmer-Lemeshow test
scorecard       FICO-style points system, single-applicant scoring, portfolio scoring
"""

from src.preprocessing import (
    load_data,
    rename_columns,
    encode_target,
    get_feature_groups,
    detect_outliers_iqr,
    COLUMN_MAPPING,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    SELECTED_NUMERIC,
    SELECTED_CATEGORICAL,
    SELECTED_FEATURES,
    LABEL_MAPS,
)

from src.woe_iv import (
    compute_woe_iv,
    compute_all_woe_iv,
    select_features_by_iv,
    build_woe_maps,
    apply_woe_encoding,
    woe_encode_new,
    plot_iv_bar,
    plot_woe_chart,
    iv_label,
)

from src.modeling import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    find_optimal_threshold,
)

from src.evaluation import (
    compute_ks,
    compute_gini,
    compute_all_metrics,
    compare_models,
    ks_benchmark,
    print_scorecard,
    hosmer_lemeshow_test,
    plot_roc_ks,
    plot_calibration,
    plot_model_comparison,
    plot_pr_curve,
)

from src.scorecard import (
    compute_scaling_factors,
    build_scorecard,
    score_single_applicant,
    score_portfolio,
    scorecard_performance,
    get_score_band,
    plot_scorecard_points,
    plot_score_distribution,
    DEFAULT_BASE_SCORE,
    DEFAULT_PDO,
    DEFAULT_BASE_ODDS,
    SCORE_BANDS,
)

__all__ = [
    # preprocessing
    'load_data', 'rename_columns', 'encode_target', 'get_feature_groups',
    'detect_outliers_iqr', 'COLUMN_MAPPING', 'NUMERIC_COLS', 'CATEGORICAL_COLS',
    'SELECTED_NUMERIC', 'SELECTED_CATEGORICAL', 'SELECTED_FEATURES', 'LABEL_MAPS',
    # woe_iv
    'compute_woe_iv', 'compute_all_woe_iv', 'select_features_by_iv',
    'build_woe_maps', 'apply_woe_encoding', 'woe_encode_new',
    'plot_iv_bar', 'plot_woe_chart', 'iv_label',
    # modeling
    'train_logistic_regression', 'train_random_forest', 'train_xgboost',
    'find_optimal_threshold',
    # evaluation
    'compute_ks', 'compute_gini', 'compute_all_metrics', 'compare_models',
    'ks_benchmark', 'print_scorecard', 'hosmer_lemeshow_test',
    'plot_roc_ks', 'plot_calibration', 'plot_model_comparison', 'plot_pr_curve',
    # scorecard
    'compute_scaling_factors', 'build_scorecard', 'score_single_applicant',
    'score_portfolio', 'scorecard_performance', 'get_score_band',
    'plot_scorecard_points', 'plot_score_distribution',
    'DEFAULT_BASE_SCORE', 'DEFAULT_PDO', 'DEFAULT_BASE_ODDS', 'SCORE_BANDS',
]
