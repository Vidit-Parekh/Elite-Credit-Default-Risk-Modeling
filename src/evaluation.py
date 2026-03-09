"""
src/evaluation.py
-----------------
Model evaluation — finance-specific metrics used in credit risk.

Metrics covered
---------------
  ROC-AUC          → area under receiver operating characteristic curve
  KS Statistic     → maximum separation between Good and Default CDF (bank standard)
  Gini Coefficient → 2 × AUC − 1
  PR-AUC           → area under precision-recall curve (imbalanced data)
  Brier Score      → mean squared error of probability calibration
  Log Loss         → cross-entropy
  F1 @ threshold   → harmonic mean of precision and recall
  Calibration      → reliability diagram + Hosmer-Lemeshow test

KS Benchmarks (industry)
-------------------------
< 0.20  → Poor
0.20–0.40 → Average
0.40–0.60 → Good
> 0.60  → Excellent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss, log_loss,
    f1_score, confusion_matrix, classification_report,
    calibration_curve,
)
from typing import Dict, Optional


# ══════════════════════════════════════════════════════════════════════════
# Single-model metrics
# ══════════════════════════════════════════════════════════════════════════

def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the KS (Kolmogorov-Smirnov) Statistic.

    KS = max | CDF_Good(score) − CDF_Default(score) |

    This is the primary discrimination metric used by banks.
    It measures how well the model separates goods from bads.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_gini(auc: float) -> float:
    """Gini Coefficient = 2 × AUC − 1."""
    return round(2 * auc - 1, 4)


def compute_all_metrics(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.5,
    model_name: str = 'Model',
) -> dict:
    """
    Compute the full suite of credit risk evaluation metrics.

    Parameters
    ----------
    y_true     : true binary labels (0/1)
    y_prob     : predicted probabilities of class 1 (default)
    threshold  : decision threshold for F1/confusion matrix
    model_name : label used in the returned dict

    Returns
    -------
    dict with all metric keys (suitable for building comparison tables)
    """
    auc     = roc_auc_score(y_true, y_prob)
    ks      = compute_ks(y_true, y_prob)
    gini    = compute_gini(auc)
    pr_auc  = average_precision_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)
    ll      = log_loss(y_true, y_prob)

    y_pred  = (y_prob >= threshold).astype(int)
    f1      = f1_score(y_true, y_pred)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'Model'         : model_name,
        'ROC_AUC'       : round(auc,   4),
        'KS_Statistic'  : round(ks,    4),
        'Gini'          : round(gini,  4),
        'PR_AUC'        : round(pr_auc,4),
        'Brier_Score'   : round(brier, 4),
        'Log_Loss'      : round(ll,    4),
        'F1_Score'      : round(f1,    4),
        'Threshold'     : threshold,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'Precision'     : round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        'Recall'        : round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
    }


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Build a comparison table from a list of compute_all_metrics() dicts.

    Parameters
    ----------
    results_list : list of dicts, each from compute_all_metrics()

    Returns
    -------
    DataFrame sorted by ROC_AUC descending, key metrics highlighted
    """
    df = pd.DataFrame(results_list)
    cols = ['Model', 'ROC_AUC', 'KS_Statistic', 'Gini',
            'PR_AUC', 'Brier_Score', 'F1_Score']
    return df[cols].sort_values('ROC_AUC', ascending=False).reset_index(drop=True)


def ks_benchmark(ks: float) -> str:
    """Return KS performance label."""
    if ks < 0.20:   return 'Poor ❌'
    elif ks < 0.40: return 'Average 🟡'
    elif ks < 0.60: return 'Good ✅'
    else:           return 'Excellent ✅✅'


def print_scorecard(metrics: dict) -> None:
    """Pretty-print a single model's metrics."""
    print('=' * 55)
    print(f"  {metrics['Model']}")
    print('=' * 55)
    print(f"  ROC-AUC       : {metrics['ROC_AUC']}")
    print(f"  KS Statistic  : {metrics['KS_Statistic']}  → {ks_benchmark(metrics['KS_Statistic'])}")
    print(f"  Gini Coeff    : {metrics['Gini']}")
    print(f"  PR-AUC        : {metrics['PR_AUC']}")
    print(f"  Brier Score   : {metrics['Brier_Score']}")
    print(f"  F1 Score      : {metrics['F1_Score']}")
    print(f"  Precision     : {metrics['Precision']}")
    print(f"  Recall        : {metrics['Recall']}")
    print(f"  Threshold     : {metrics['Threshold']}")
    print(f"  Confusion Matrix:")
    print(f"      TP={metrics['TP']}  FP={metrics['FP']}")
    print(f"      FN={metrics['FN']}  TN={metrics['TN']}")


# ══════════════════════════════════════════════════════════════════════════
# Calibration — Hosmer-Lemeshow test
# ══════════════════════════════════════════════════════════════════════════

def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Hosmer-Lemeshow goodness-of-fit test for logistic regression calibration.

    The H-L test partitions predictions into g groups (deciles) and tests
    whether observed default rates match predicted probabilities.

    H0 : model is well-calibrated (p > 0.05 → fail to reject → good calibration)
    H1 : model is poorly calibrated

    Returns
    -------
    dict with:
        hl_statistic → chi-squared test statistic
        p_value      → p-value (want > 0.05)
        df           → degrees of freedom (n_bins - 2)
        result       → 'Well calibrated' or 'Poorly calibrated'
        table        → DataFrame with observed vs expected per bin
    """
    from scipy.stats import chi2

    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df['decile'] = pd.qcut(df['y_prob'], q=n_bins, duplicates='drop')

    grouped = df.groupby('decile', observed=True).agg(
        n      = ('y_true', 'count'),
        obs_1  = ('y_true', 'sum'),
        exp_1  = ('y_prob', 'sum'),
    ).reset_index()
    grouped['obs_0'] = grouped['n'] - grouped['obs_1']
    grouped['exp_0'] = grouped['n'] - grouped['exp_1']

    hl_stat = float(np.sum(
        (grouped['obs_1'] - grouped['exp_1'])**2 / grouped['exp_1'].clip(lower=0.001) +
        (grouped['obs_0'] - grouped['exp_0'])**2 / grouped['exp_0'].clip(lower=0.001)
    ))
    dof    = n_bins - 2
    pvalue = float(1 - chi2.cdf(hl_stat, dof))

    return {
        'hl_statistic' : round(hl_stat, 4),
        'p_value'      : round(pvalue,  4),
        'df'           : dof,
        'result'       : 'Well calibrated ✅' if pvalue > 0.05 else 'Poorly calibrated ❌',
        'table'        : grouped,
    }


# ══════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════

def plot_roc_ks(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = 'Model',
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """ROC curve + KS separation plot side by side."""
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ks  = float(np.max(tpr - fpr))
    ks_idx = int(np.argmax(tpr - fpr))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC
    axes[0].plot(fpr, tpr, color='steelblue', linewidth=2.5,
                 label=f'{model_name} (AUC={auc:.4f})')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve', fontweight='bold')
    axes[0].legend()

    # KS separation
    sorted_df = pd.DataFrame({'score': y_prob, 'target': y_true})\
                  .sort_values('score', ascending=False).reset_index(drop=True)
    n  = len(sorted_df)
    n1 = int(y_true.sum())
    n0 = n - n1

    cdf_bad  = sorted_df['target'].cumsum() / n1
    cdf_good = (1 - sorted_df['target']).cumsum() / n0
    pct      = np.arange(1, n + 1) / n * 100

    axes[1].plot(pct, cdf_bad  * 100, color='crimson',   linewidth=2, label='Default CDF')
    axes[1].plot(pct, cdf_good * 100, color='steelblue', linewidth=2, label='Good CDF')
    ks_pct = pct[int(np.argmax(np.abs(cdf_bad - cdf_good)))]
    axes[1].axvline(ks_pct, color='black', linestyle='--', linewidth=1.5,
                   label=f'KS = {ks:.4f}  ({ks_benchmark(ks)})')
    axes[1].set_xlabel('% of Population (ranked by score)')
    axes[1].set_ylabel('Cumulative %')
    axes[1].set_title('KS Separation Plot', fontweight='bold')
    axes[1].legend()

    plt.suptitle(f'{model_name} — Discrimination Metrics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = 'Model',
    n_bins: int = 10,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Reliability diagram (calibration curve)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(prob_pred, prob_true, 's-', color='steelblue', linewidth=2,
            markersize=8, label=f'{model_name} (Brier={brier:.4f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Defaults (Observed)')
    ax.set_title('Calibration Curve (Reliability Diagram)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Grouped bar chart comparing ROC-AUC, KS, and Gini across models."""
    metrics   = ['ROC_AUC', 'KS_Statistic', 'Gini']
    x         = np.arange(len(comparison_df))
    width     = 0.25
    colors    = ['steelblue', 'darkorange', 'seagreen']

    fig, ax = plt.subplots(figsize=figsize)
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, comparison_df[metric],
                      width, label=metric, color=color, alpha=0.85, edgecolor='black')
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(comparison_df['Model'], fontsize=11)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.7, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_title('Model Comparison — AUC / KS / Gini', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def plot_pr_curve(
    models_dict: Dict[str, tuple],
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Precision-Recall curves for multiple models.

    Parameters
    ----------
    models_dict : { model_name → (y_true, y_prob) }
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors  = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    for (name, (y_true, y_prob)), color in zip(models_dict.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc       = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, linewidth=2, color=color,
                label=f'{name} (PR-AUC={pr_auc:.4f})')

    baseline = y_true.mean() if hasattr(y_true, 'mean') else 0.30
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1,
               label=f'Baseline (prevalence={baseline:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


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
    from src.modeling import train_logistic_regression

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/german_credit_data.csv'
    df   = load_data(path)

    woe_tables, iv_summary = compute_all_woe_iv(
        df, SELECTED_FEATURES, 'target', SELECTED_NUMERIC)
    selected, _ = select_features_by_iv(iv_summary)
    woe_maps    = build_woe_maps(woe_tables)
    df_enc      = apply_woe_encoding(df, selected, SELECTED_NUMERIC, woe_maps)
    woe_cols    = [f'{f}_woe' for f in selected if f'{f}_woe' in df_enc.columns]

    X = df_enc[woe_cols].dropna()
    y = df_enc.loc[X.index, 'target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)

    lr  = train_logistic_regression(X_tr, y_tr, X_te)
    m   = compute_all_metrics(y_te.values, lr['y_prob_test'], model_name='Logistic Regression')
    print_scorecard(m)

    hl = hosmer_lemeshow_test(y_te.values, lr['y_prob_test'])
    print(f"\nHosmer-Lemeshow: χ²={hl['hl_statistic']}, p={hl['p_value']} → {hl['result']}")
