"""
src/scorecard.py
----------------
FICO-style credit scorecard — points per bin, full score calculation,
portfolio scoring, and band assignment.

Theory
------
A scorecard converts logistic regression log-odds to a points system.
Each feature bin contributes a fixed number of points.

Key formulas:
    Factor = PDO / ln(2)
    Offset = Base_Score − Factor × ln(Base_Odds)

    Points_ij = −(β_i × WoE_ij + β_0/k) × Factor

    Score = Offset + Σ Points_ij
          = Offset − Factor × (β_0 + β_1×WoE_1 + ... + β_k×WoE_k)
          = Offset − Factor × log_odds

Score ↑ → log_odds ↓ → P(Default) ↓   (higher score = safer applicant)

PDO (Points to Double Odds): every PDO points added to the score
doubles the odds of being a Good borrower.

References
----------
Anderson, R. (2007). The Credit Scoring Toolkit. Oxford University Press.
Basel Committee on Banking Supervision (2006). Basel II: IRB Approach.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════
# Scorecard constants
# ══════════════════════════════════════════════════════════════════════════

# Industry standard: Base Score 600, PDO 20, Base Odds 1:1
DEFAULT_BASE_SCORE = 600
DEFAULT_PDO        = 20
DEFAULT_BASE_ODDS  = 1       # 1 Good : 1 Bad at the base score


# ══════════════════════════════════════════════════════════════════════════
# Score bands
# ══════════════════════════════════════════════════════════════════════════

SCORE_BANDS = [
    # (min_score_inclusive, label, decision, hex_color)
    (750, 'Very Low Risk',  'Auto Approve',  '#2ecc71'),
    (650, 'Low Risk',       'Approve',       '#82e0aa'),
    (550, 'Medium Risk',    'Manual Review', '#f39c12'),
    (450, 'High Risk',      'Decline',       '#e74c3c'),
    (  0, 'Very High Risk', 'Auto Decline',  '#922b21'),
]


def get_score_band(score: int) -> Tuple[str, str, str]:
    """
    Map a credit score to (risk_band, decision, hex_color).

    Parameters
    ----------
    score : integer credit score

    Returns
    -------
    (risk_band, decision, hex_color)
    """
    for threshold, band, decision, color in SCORE_BANDS:
        if score >= threshold:
            return band, decision, color
    return 'Very High Risk', 'Auto Decline', '#922b21'


# ══════════════════════════════════════════════════════════════════════════
# Scaling factors
# ══════════════════════════════════════════════════════════════════════════

def compute_scaling_factors(
    pdo:        int   = DEFAULT_PDO,
    base_score: int   = DEFAULT_BASE_SCORE,
    base_odds:  float = DEFAULT_BASE_ODDS,
) -> Tuple[float, float]:
    """
    Compute the Factor and Offset used in the points formula.

    Factor = PDO / ln(2)
    Offset = Base_Score − Factor × ln(Base_Odds)

    Returns
    -------
    (factor, offset)
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds + 1e-9)
    return factor, offset


# ══════════════════════════════════════════════════════════════════════════
# Build scorecard table
# ══════════════════════════════════════════════════════════════════════════

def build_scorecard(
    lr_model,                          # statsmodels Logit result
    woe_tables:  Dict[str, pd.DataFrame],
    woe_cols:    List[str],
    features:    List[str],
    factor:      float,
    offset:      float,
) -> pd.DataFrame:
    """
    Build the full scorecard — one row per (feature, bin).

    Each row shows how many points that bin adds to or subtracts
    from the applicant's total credit score.

    Parameters
    ----------
    lr_model    : fitted statsmodels Logit result
    woe_tables  : dict { feature → WoE table DataFrame }
    woe_cols    : list of WoE column names used in the model
    features    : list of raw feature names
    factor      : from compute_scaling_factors()
    offset      : from compute_scaling_factors()

    Returns
    -------
    scorecard_df : DataFrame with columns
        [Feature, Bin, Count, Default_Rate, WoE, Beta, Points]
    """
    intercept     = lr_model.params['const']
    n_features    = len(woe_cols)
    ipt_per_feat  = intercept / n_features

    rows = []
    for feat in features:
        woe_col = f'{feat}_woe'
        if woe_col not in woe_cols:
            continue
        beta = lr_model.params.get(woe_col)
        if beta is None:
            continue
        for _, row in woe_tables[feat].iterrows():
            woe_val = float(row['WoE'])
            points  = -(float(beta) * woe_val + ipt_per_feat) * factor
            rows.append({
                'Feature'     : feat,
                'Bin'         : str(row['bin']),
                'Count'       : int(row['Total']),
                'Default_Rate': round(float(row['Default_Rate']), 4),
                'WoE'         : round(woe_val, 4),
                'Beta'        : round(float(beta), 4),
                'Points'      : round(points, 1),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# Score a single applicant
# ══════════════════════════════════════════════════════════════════════════

def score_single_applicant(
    applicant:    dict,
    lr_model,
    woe_maps:     Dict[str, dict],
    features:     List[str],
    numeric_cols: List[str],
    woe_cols:     List[str],
    factor:       float,
    offset:       float,
) -> Tuple[int, float, pd.DataFrame]:
    """
    Compute credit score, default probability, and feature breakdown
    for a single applicant.

    Parameters
    ----------
    applicant    : dict { feature_name → raw_value }
    lr_model     : fitted statsmodels Logit result
    woe_maps     : dict { feature → { bin → WoE } }
    features     : list of features in model
    numeric_cols : which features are numeric (need interval matching)
    woe_cols     : WoE column names in model
    factor       : scaling factor
    offset       : scaling offset

    Returns
    -------
    (score, default_probability, breakdown_df)
        score              : integer credit score
        default_probability: float PD (0–1)
        breakdown_df       : DataFrame with per-feature WoE and Points
    """
    intercept    = lr_model.params['const']
    n_feat       = len(woe_cols)
    ipt_per_feat = intercept / n_feat

    woe_row = {}

    for feat in features:
        woe_col = f'{feat}_woe'
        if woe_col not in woe_cols:
            continue

        raw  = applicant.get(feat)
        wmap = woe_maps.get(feat, {})
        wv   = None

        if feat in numeric_cols:
            for interval, w in wmap.items():
                try:
                    if raw in interval:
                        wv = w
                        break
                except TypeError:
                    pass
            if wv is None:
                lefts = [iv.left for iv in wmap.keys()]
                wv = (list(wmap.values())[0]
                      if raw <= min(lefts)
                      else list(wmap.values())[-1])
        else:
            wv = wmap.get(raw, 0.0)

        woe_row[woe_col] = wv if wv is not None else 0.0

    # ── Predicted probability ──────────────────────────────────────────────
    woe_series       = pd.Series(woe_row)
    woe_with_const   = pd.concat([pd.Series({'const': 1.0}), woe_series])
    log_odds         = lr_model.params.reindex(woe_with_const.index).fillna(0).dot(woe_with_const)
    default_prob     = float(1 / (1 + np.exp(-log_odds)))
    credit_score     = int(round(offset - factor * log_odds))

    # ── Per-feature breakdown ──────────────────────────────────────────────
    breakdown = []
    for feat in features:
        woe_col = f'{feat}_woe'
        if woe_col not in woe_cols:
            continue
        beta = float(lr_model.params.get(woe_col, 0))
        wv   = woe_row.get(woe_col, 0.0)
        pts  = -(beta * wv + ipt_per_feat) * factor
        breakdown.append({
            'Feature' : feat,
            'Value'   : applicant.get(feat),
            'WoE'     : round(wv, 4),
            'Points'  : round(pts, 1),
        })

    breakdown_df = pd.DataFrame(breakdown)
    return credit_score, round(default_prob, 4), breakdown_df


# ══════════════════════════════════════════════════════════════════════════
# Score full portfolio (vectorised)
# ══════════════════════════════════════════════════════════════════════════

def score_portfolio(
    df_woe_encoded: pd.DataFrame,
    lr_model,
    woe_cols:       List[str],
    factor:         float,
    offset:         float,
    target_col:     str = 'target',
) -> pd.DataFrame:
    """
    Score all applicants in a WoE-encoded DataFrame.

    Parameters
    ----------
    df_woe_encoded : DataFrame with '{feature}_woe' columns
    lr_model       : fitted statsmodels Logit result
    woe_cols       : WoE column names
    factor, offset : scaling constants

    Returns
    -------
    DataFrame with columns:
        credit_score, default_prob, score_band, decision
        (plus target if present)
    """
    X  = df_woe_encoded[woe_cols].dropna()
    Xs = sm.add_constant(X)

    log_odds      = Xs @ lr_model.params
    default_probs = 1 / (1 + np.exp(-log_odds))
    scores        = (offset - factor * log_odds).round().astype(int)

    result = df_woe_encoded.loc[X.index].copy()
    result['credit_score'] = scores.values
    result['default_prob'] = default_probs.values.round(4)

    bands      = [get_score_band(s) for s in result['credit_score']]
    result['score_band'] = [b[0] for b in bands]
    result['decision']   = [b[1] for b in bands]

    return result


# ══════════════════════════════════════════════════════════════════════════
# Scorecard validation
# ══════════════════════════════════════════════════════════════════════════

def scorecard_performance(
    scored_df: pd.DataFrame,
    target_col: str = 'target',
) -> pd.DataFrame:
    """
    Validate scorecard monotonicity: default rate should decrease
    as score band improves.

    Returns band-level performance table.
    """
    band_order = ['Very High Risk', 'High Risk', 'Medium Risk',
                  'Low Risk', 'Very Low Risk']
    existing   = [b for b in band_order if b in scored_df['score_band'].unique()]

    perf = scored_df.groupby('score_band').agg(
        Count        = (target_col, 'count'),
        Defaults     = (target_col, 'sum'),
        Default_Rate = (target_col, 'mean'),
        Avg_Score    = ('credit_score', 'mean'),
    ).reindex(existing).round(3)

    perf['Approval_Rate'] = (1 - perf['Default_Rate']).round(3)
    return perf


# ══════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════

def plot_scorecard_points(
    scorecard_df: pd.DataFrame,
    n_features:   int = 6,
    figsize:      tuple = (18, 10),
) -> plt.Figure:
    """
    Horizontal bar charts of Points per Bin for the top-N features
    (ranked by points range = max − min).
    """
    top_feats = (
        scorecard_df.groupby('Feature')['Points']
        .apply(lambda x: x.max() - x.min())
        .nlargest(n_features)
        .index.tolist()
    )
    rows = int(np.ceil(n_features / 3))
    fig, axes = plt.subplots(rows, 3, figsize=figsize)
    axes = axes.flatten()

    for i, feat in enumerate(top_feats):
        feat_df = scorecard_df[scorecard_df['Feature'] == feat].sort_values('Points')
        colors  = ['#e74c3c' if p < 0 else '#2ecc71' for p in feat_df['Points']]
        labels  = [str(b)[:28] for b in feat_df['Bin']]

        axes[i].barh(labels, feat_df['Points'],
                     color=colors, edgecolor='black', alpha=0.85)
        axes[i].axvline(0, color='black', linewidth=1)
        for bar, val in zip(axes[i].patches, feat_df['Points']):
            xpos = bar.get_width() + (0.2 if val >= 0 else -0.2)
            axes[i].text(xpos, bar.get_y() + bar.get_height() / 2,
                         f'{val:.1f}', va='center', fontsize=8)
        axes[i].set_title(feat, fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Points')

    # Hide unused axes
    for j in range(len(top_feats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f'Scorecard Points per Bin — Top {n_features} Features\n'
        '(Green = positive points = lower risk  |  Red = negative points = higher risk)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    return fig


def plot_score_distribution(
    scored_df: pd.DataFrame,
    target_col: str = 'target',
    figsize:    tuple = (16, 5),
) -> plt.Figure:
    """Score distribution histogram split by Good / Default."""
    good    = scored_df[scored_df[target_col] == 0]['credit_score']
    default = scored_df[scored_df[target_col] == 1]['credit_score']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Distribution overlay
    axes[0].hist(good,    bins=30, alpha=0.65, color='steelblue',
                 density=True, label=f'Good (n={len(good)})')
    axes[0].hist(default, bins=30, alpha=0.65, color='crimson',
                 density=True, label=f'Default (n={len(default)})')
    axes[0].axvline(good.mean(),    color='steelblue', linestyle='--', linewidth=2,
                    label=f'Good mean: {good.mean():.0f}')
    axes[0].axvline(default.mean(), color='crimson',   linestyle='--', linewidth=2,
                    label=f'Default mean: {default.mean():.0f}')
    axes[0].set_title('Score Distribution: Good vs Default', fontweight='bold')
    axes[0].set_xlabel('Credit Score')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=9)

    # Default rate by band
    band_order = ['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']
    existing   = [b for b in band_order if b in scored_df['score_band'].unique()]
    bstats     = scored_df.groupby('score_band').agg(
        Count        = (target_col, 'count'),
        Default_Rate = (target_col, 'mean'),
    ).reindex(existing)

    colors_all = ['#922b21', '#e74c3c', '#f39c12', '#82e0aa', '#2ecc71']
    bc = colors_all[-len(bstats):]
    bars = axes[1].bar(range(len(bstats)), bstats['Default_Rate'] * 100,
                       color=bc, edgecolor='black', alpha=0.85)
    axes[1].set_xticks(range(len(bstats)))
    axes[1].set_xticklabels(bstats.index, rotation=25, ha='right', fontsize=9)
    for bar, val in zip(bars, bstats['Default_Rate'] * 100):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    axes[1].axhline(scored_df[target_col].mean() * 100, color='navy',
                    linestyle='--', linewidth=1.5,
                    label=f'Avg {scored_df[target_col].mean()*100:.1f}%')
    axes[1].set_title('Default Rate by Score Band', fontweight='bold')
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].legend(fontsize=9)

    # Count by band
    axes[2].bar(range(len(bstats)), bstats['Count'],
                color=bc, edgecolor='black', alpha=0.85)
    axes[2].set_xticks(range(len(bstats)))
    axes[2].set_xticklabels(bstats.index, rotation=25, ha='right', fontsize=9)
    for i, cnt in enumerate(bstats['Count']):
        axes[2].text(i, cnt + 3, str(int(cnt)), ha='center', fontsize=9, fontweight='bold')
    axes[2].set_title('Applicant Count by Score Band', fontweight='bold')
    axes[2].set_ylabel('Count')

    plt.suptitle('Credit Scorecard Analysis', fontsize=14, fontweight='bold')
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
    from src.woe_iv        import (compute_all_woe_iv, build_woe_maps,
                                   apply_woe_encoding, select_features_by_iv)
    from src.modeling      import train_logistic_regression

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
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    lr_res  = train_logistic_regression(X_tr, y_tr, X_te)
    lr      = lr_res['model']

    factor, offset = compute_scaling_factors()
    print(f'Factor={factor:.4f}  Offset={offset:.4f}')

    sc_df = build_scorecard(lr, woe_tables, woe_cols, selected, factor, offset)
    print(f'\nScorecard: {sc_df["Feature"].nunique()} features, {len(sc_df)} bins')
    print(sc_df.head(10).to_string(index=False))

    # Score 3 example applicants
    example = {
        'duration_months':12, 'credit_amount':1500, 'age':45,
        'installment_rate':1, 'checking_account':4, 'credit_history':4,
        'purpose':0, 'savings_account':5, 'employment_years':5,
        'personal_status':3, 'other_debtors':1, 'property':1,
        'other_installments':3, 'housing':2, 'foreign_worker':2,
    }
    score, prob, bdown = score_single_applicant(
        example, lr, woe_maps, selected, SELECTED_NUMERIC, woe_cols, factor, offset)
    band, decision, _ = get_score_band(score)
    print(f'\nExample Applicant → Score: {score}  PD: {prob*100:.1f}%  Band: {band}  Decision: {decision}')
    print(bdown.to_string(index=False))
