"""
src/modeling.py
---------------
Model training utilities for all three models in Phase 3 & 4:
  - Logistic Regression  (statsmodels — statistical inference)
  - Random Forest        (sklearn     — Optuna tuning)
  - XGBoost              (xgboost     — Optuna tuning)

Design principle: each function is self-contained and returns a
(model, y_prob_train, y_prob_test) tuple so evaluation.py can
consume any of them identically.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import StratifiedKFold, cross_val_score
from sklearn.metrics           import roc_auc_score
from xgboost                   import XGBClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("  ⚠  optuna not installed — hyperparameter tuning disabled.")


# ══════════════════════════════════════════════════════════════════════════
# Logistic Regression (statsmodels)
# ══════════════════════════════════════════════════════════════════════════

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    method:  str = 'newton',
    maxiter: int = 200,
    add_constant: bool = True,
) -> dict:
    """
    Fit statsmodels Logit and return full results.

    Why statsmodels (not sklearn)?
    ─────────────────────────────
    In regulated credit risk environments you need p-values, confidence
    intervals, and odds ratios for every coefficient. sklearn's LogisticRegression
    does not expose these. statsmodels fits the model via MLE and provides the
    full statistical summary required by regulators and auditors.

    Parameters
    ----------
    X_train      : WoE-encoded feature matrix (train set)
    y_train      : binary target (train set)
    X_test       : WoE-encoded feature matrix (test set)
    method       : optimisation method ('newton', 'bfgs', 'lbfgs')
    maxiter      : maximum iterations
    add_constant : whether to prepend intercept column

    Returns
    -------
    dict with keys:
        model         → fitted statsmodels Logit result
        X_train_sm    → design matrix with constant (train)
        X_test_sm     → design matrix with constant (test)
        y_prob_train  → predicted probabilities on train
        y_prob_test   → predicted probabilities on test
        summary       → model.summary2() output
        odds_ratios   → DataFrame with OR and 95% CI
    """
    X_tr = sm.add_constant(X_train) if add_constant else X_train.copy()
    X_te = sm.add_constant(X_test)  if add_constant else X_test.copy()

    model = sm.Logit(y_train, X_tr).fit(
        method=method, maxiter=maxiter, disp=False
    )

    y_prob_train = model.predict(X_tr)
    y_prob_test  = model.predict(X_te)

    # ── Odds ratios ────────────────────────────────────────────────────────
    params = model.params
    conf   = model.conf_int()
    conf.columns = ['CI_lower', 'CI_upper']
    or_df  = pd.DataFrame({
        'Coefficient' : params,
        'Odds_Ratio'  : np.exp(params),
        'OR_CI_lower' : np.exp(conf['CI_lower']),
        'OR_CI_upper' : np.exp(conf['CI_upper']),
        'p_value'     : model.pvalues,
        'Significant' : model.pvalues < 0.05,
    }).round(4)

    return {
        'model'        : model,
        'X_train_sm'   : X_tr,
        'X_test_sm'    : X_te,
        'y_prob_train' : y_prob_train.values,
        'y_prob_test'  : y_prob_test.values,
        'summary'      : model.summary2(),
        'odds_ratios'  : or_df,
    }


# ══════════════════════════════════════════════════════════════════════════
# Random Forest (sklearn + Optuna)
# ══════════════════════════════════════════════════════════════════════════

def _rf_objective(trial, X_train, y_train, n_splits=5):
    params = {
        'n_estimators'      : trial.suggest_int('n_estimators',   50, 400),
        'max_depth'         : trial.suggest_int('max_depth',       3,  15),
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf'  : trial.suggest_int('min_samples_leaf',  1, 10),
        'max_features'      : trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight'      : 'balanced',
        'random_state'      : 42,
        'n_jobs'            : -1,
    }
    clf = RandomForestClassifier(**params)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv,
                             scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 5,
    tune: bool = True,
) -> dict:
    """
    Train Random Forest with optional Optuna hyperparameter tuning.

    Parameters
    ----------
    X_train  : feature matrix (raw features, not WoE-encoded)
    y_train  : binary target
    X_test   : feature matrix (test set)
    n_trials : Optuna trials (used when tune=True)
    n_splits : cross-validation folds
    tune     : True  → run Optuna TPE search
               False → use sensible defaults

    Returns
    -------
    dict with keys:
        model         → fitted RandomForestClassifier
        best_params   → dict of hyperparameters used
        y_prob_train  → predicted probabilities on train
        y_prob_test   → predicted probabilities on test
        study         → Optuna study object (None if tune=False)
        cv_auc        → best cross-validation AUC
    """
    if tune and OPTUNA_AVAILABLE:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda t: _rf_objective(t, X_train, y_train, n_splits),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = 42
        best_params['n_jobs']       = -1
        cv_auc = study.best_value
        print(f"  Best CV AUC (RF): {cv_auc:.4f}")
        print(f"  Best params     : {best_params}")
    else:
        study = None
        best_params = {
            'n_estimators'      : 200,
            'max_depth'         : 8,
            'min_samples_split' : 4,
            'min_samples_leaf'  : 2,
            'max_features'      : 'sqrt',
            'class_weight'      : 'balanced',
            'random_state'      : 42,
            'n_jobs'            : -1,
        }
        cv_auc = None

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]

    return {
        'model'        : model,
        'best_params'  : best_params,
        'y_prob_train' : y_prob_train,
        'y_prob_test'  : y_prob_test,
        'study'        : study,
        'cv_auc'       : cv_auc,
    }


# ══════════════════════════════════════════════════════════════════════════
# XGBoost (xgboost + Optuna)
# ══════════════════════════════════════════════════════════════════════════

def _xgb_objective(trial, X_train, y_train, scale_pos_weight, n_splits=5):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators',     50, 500),
        'max_depth'        : trial.suggest_int('max_depth',          3,  10),
        'learning_rate'    : trial.suggest_float('learning_rate',  0.01, 0.30, log=True),
        'subsample'        : trial.suggest_float('subsample',       0.5,  1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',0.5,  1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight',   1,  20),
        'reg_alpha'        : trial.suggest_float('reg_alpha',       0.0,  5.0),
        'reg_lambda'       : trial.suggest_float('reg_lambda',      0.5,  5.0),
        'scale_pos_weight' : scale_pos_weight,
        'use_label_encoder': False,
        'eval_metric'      : 'auc',
        'random_state'     : 42,
        'n_jobs'           : -1,
    }
    clf = XGBClassifier(**params)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv,
                             scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 5,
    tune: bool = True,
) -> dict:
    """
    Train XGBoost with optional Optuna hyperparameter tuning.

    Class imbalance is handled via scale_pos_weight = n_negative / n_positive,
    which upweights the minority (default) class in the XGBoost loss function.

    Parameters
    ----------
    X_train  : feature matrix (raw or WoE-encoded — raw recommended for XGB)
    y_train  : binary target
    X_test   : feature matrix (test set)
    n_trials : Optuna trials
    n_splits : cross-validation folds
    tune     : run Optuna search if True

    Returns
    -------
    dict with keys:
        model              → fitted XGBClassifier
        best_params        → dict of hyperparameters used
        scale_pos_weight   → imbalance weight used
        y_prob_train       → predicted probabilities on train
        y_prob_test        → predicted probabilities on test
        study              → Optuna study (None if tune=False)
        cv_auc             → best cross-validation AUC
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = round(n_neg / n_pos, 4)
    print(f"  scale_pos_weight = {spw:.2f}  (n_neg={n_neg}, n_pos={n_pos})")

    if tune and OPTUNA_AVAILABLE:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda t: _xgb_objective(t, X_train, y_train, spw, n_splits),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        best_params['scale_pos_weight'] = spw
        best_params['use_label_encoder'] = False
        best_params['eval_metric']       = 'auc'
        best_params['random_state']      = 42
        best_params['n_jobs']            = -1
        cv_auc = study.best_value
        print(f"  Best CV AUC (XGB): {cv_auc:.4f}")
        print(f"  Best params      : {best_params}")
    else:
        study = None
        best_params = {
            'n_estimators'     : 200,
            'max_depth'        : 5,
            'learning_rate'    : 0.05,
            'subsample'        : 0.8,
            'colsample_bytree' : 0.8,
            'min_child_weight' : 5,
            'reg_alpha'        : 0.5,
            'reg_lambda'       : 1.5,
            'scale_pos_weight' : spw,
            'use_label_encoder': False,
            'eval_metric'      : 'auc',
            'random_state'     : 42,
            'n_jobs'           : -1,
        }
        cv_auc = None

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]

    return {
        'model'            : model,
        'best_params'      : best_params,
        'scale_pos_weight' : spw,
        'y_prob_train'     : y_prob_train,
        'y_prob_test'      : y_prob_test,
        'study'            : study,
        'cv_auc'           : cv_auc,
    }


# ══════════════════════════════════════════════════════════════════════════
# Threshold optimisation
# ══════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> dict:
    """
    Find the classification threshold that minimises total business cost.

    In credit risk:
      FN (missed default)  → bank loses loan amount   → HIGH cost
      FP (rejected good)   → bank loses interest only → LOW cost

    Default ratio FN:FP = 5:1 reflects that a missed default is ~5×
    more costly than an unnecessary rejection.

    Parameters
    ----------
    y_true   : true binary labels
    y_prob   : predicted probabilities
    fn_cost  : relative cost of a false negative
    fp_cost  : relative cost of a false positive

    Returns
    -------
    dict with keys:
        optimal_threshold → float
        min_cost          → float
        threshold_curve   → DataFrame (threshold, FN, FP, total_cost)
    """
    thresholds   = np.arange(0.05, 0.80, 0.01)
    results      = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn     = int(((y_true == 1) & (y_pred == 0)).sum())
        fp     = int(((y_true == 0) & (y_pred == 1)).sum())
        cost   = fn * fn_cost + fp * fp_cost
        results.append({'threshold': round(t, 2), 'FN': fn, 'FP': fp, 'total_cost': cost})

    curve_df = pd.DataFrame(results)
    best_row = curve_df.loc[curve_df['total_cost'].idxmin()]

    return {
        'optimal_threshold' : float(best_row['threshold']),
        'min_cost'          : float(best_row['total_cost']),
        'threshold_curve'   : curve_df,
    }


# ══════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from sklearn.model_selection import train_test_split
    from src.preprocessing import load_data, SELECTED_FEATURES, SELECTED_NUMERIC
    from src.woe_iv import (compute_all_woe_iv, build_woe_maps,
                            apply_woe_encoding, select_features_by_iv)

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/german_credit_data.csv'
    df   = load_data(path)

    woe_tables, iv_summary = compute_all_woe_iv(
        df, SELECTED_FEATURES, 'target', SELECTED_NUMERIC)
    selected, _   = select_features_by_iv(iv_summary)
    woe_maps      = build_woe_maps(woe_tables)
    df_enc        = apply_woe_encoding(df, selected, SELECTED_NUMERIC, woe_maps)

    woe_cols = [f'{f}_woe' for f in selected if f'{f}_woe' in df_enc.columns]
    X = df_enc[woe_cols].dropna()
    y = df_enc.loc[X.index, 'target']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n── Logistic Regression ──")
    lr_res = train_logistic_regression(X_tr, y_tr, X_te)
    print(f"Test AUC: {roc_auc_score(y_te, lr_res['y_prob_test']):.4f}")

    print("\n── Random Forest (no tuning) ──")
    rf_res = train_random_forest(X_tr, y_tr, X_te, tune=False)
    print(f"Test AUC: {roc_auc_score(y_te, rf_res['y_prob_test']):.4f}")

    print("\n── XGBoost (no tuning) ──")
    xgb_res = train_xgboost(X_tr, y_tr, X_te, tune=False)
    print(f"Test AUC: {roc_auc_score(y_te, xgb_res['y_prob_test']):.4f}")
