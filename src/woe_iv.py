"""
src/woe_iv.py
-------------
Weight of Evidence (WoE) and Information Value (IV) computation from scratch.

Theory
------
WoE_i = ln( P(Events in bin i) / P(Non-Events in bin i) )
IV     = Σ [ (P(Events_i) - P(NonEvents_i)) × WoE_i ]

WoE linearises the log-odds relationship between each feature and the
binary target, making it ideal as input to logistic regression.

IV Interpretation
-----------------
< 0.02   → Useless  (drop)
0.02–0.1 → Weak
0.1–0.3  → Medium
0.3–0.5  → Strong
> 0.5    → Suspicious (check for leakage)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Tuple


# ── IV strength labels ─────────────────────────────────────────────────────
IV_LABELS = {
    (0.00, 0.02): 'Useless',
    (0.02, 0.10): 'Weak',
    (0.10, 0.30): 'Medium',
    (0.30, 0.50): 'Strong',
    (0.50, 9.99): 'Suspicious',
}


def iv_label(iv: float) -> str:
    for (lo, hi), label in IV_LABELS.items():
        if lo <= iv < hi:
            return label
    return 'Unknown'


# ══════════════════════════════════════════════════════════════════════════
# Core WoE / IV computation
# ══════════════════════════════════════════════════════════════════════════

def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
    is_numeric: bool = True,
    epsilon: float = 0.5,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute WoE and IV for a single feature.

    Parameters
    ----------
    df         : DataFrame containing feature and target columns
    feature    : column name to analyse
    target     : binary target column (0 = Good, 1 = Default)
    bins       : number of quantile bins for numeric features
    is_numeric : True → qcut binning; False → use raw category values
    epsilon    : small constant added to avoid log(0); default 0.5

    Returns
    -------
    (woe_table, iv_total)
        woe_table : DataFrame with columns
                    [bin, Total, Events, NonEvents, Default_Rate,
                     pct_Events, pct_NonEvents, WoE, IV]
        iv_total  : float, sum of IV across all bins
    """
    df = df[[feature, target]].copy()

    total_events     = int(df[target].sum())
    total_non_events = len(df) - total_events

    if total_events == 0 or total_non_events == 0:
        raise ValueError(
            f"Feature '{feature}': target has only one class — cannot compute WoE."
        )

    # ── Bin the feature ────────────────────────────────────────────────────
    if is_numeric:
        df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    else:
        df['bin'] = df[feature]

    # ── Aggregate ─────────────────────────────────────────────────────────
    grouped = (
        df.groupby('bin', observed=True)[target]
        .agg(Events='sum', Total='count')
        .reset_index()
    )
    grouped['NonEvents'] = grouped['Total'] - grouped['Events']

    # ── WoE & IV with Laplace smoothing ───────────────────────────────────
    grouped['pct_Events']    = (grouped['Events']    + epsilon) / (total_events     + epsilon)
    grouped['pct_NonEvents'] = (grouped['NonEvents'] + epsilon) / (total_non_events + epsilon)
    grouped['WoE']           = np.log(grouped['pct_Events'] / grouped['pct_NonEvents'])
    grouped['IV']            = (grouped['pct_Events'] - grouped['pct_NonEvents']) * grouped['WoE']
    grouped['Default_Rate']  = grouped['Events'] / grouped['Total']

    woe_table = grouped[[
        'bin', 'Total', 'Events', 'NonEvents',
        'Default_Rate', 'pct_Events', 'pct_NonEvents', 'WoE', 'IV'
    ]].round(4)

    iv_total = round(float(grouped['IV'].sum()), 4)
    return woe_table, iv_total


def compute_all_woe_iv(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    numeric_cols: List[str],
    bins: int = 10,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Compute WoE and IV for every feature in `features`.

    Returns
    -------
    woe_tables : dict  { feature_name → woe_table DataFrame }
    iv_summary : DataFrame with [Feature, IV, Strength] sorted descending
    """
    woe_tables = {}
    iv_rows    = []

    for feat in features:
        is_num = feat in numeric_cols
        try:
            wt, iv = compute_woe_iv(df, feat, target, bins=bins, is_numeric=is_num)
            woe_tables[feat] = wt
            iv_rows.append({
                'Feature' : feat,
                'IV'      : iv,
                'Strength': iv_label(iv),
            })
        except Exception as e:
            print(f"  ⚠  Skipped '{feat}': {e}")

    iv_summary = (
        pd.DataFrame(iv_rows)
        .sort_values('IV', ascending=False)
        .reset_index(drop=True)
    )
    return woe_tables, iv_summary


# ══════════════════════════════════════════════════════════════════════════
# Feature selection
# ══════════════════════════════════════════════════════════════════════════

def select_features_by_iv(
    iv_summary: pd.DataFrame,
    threshold: float = 0.02,
) -> Tuple[List[str], List[str]]:
    """
    Split features into selected (IV >= threshold) and dropped.

    Returns
    -------
    (selected_features, dropped_features)
    """
    selected = iv_summary[iv_summary['IV'] >= threshold]['Feature'].tolist()
    dropped  = iv_summary[iv_summary['IV'] <  threshold]['Feature'].tolist()
    return selected, dropped


# ══════════════════════════════════════════════════════════════════════════
# WoE encoding
# ══════════════════════════════════════════════════════════════════════════

def build_woe_maps(
    woe_tables: Dict[str, pd.DataFrame],
) -> Dict[str, dict]:
    """
    Build a bin→WoE lookup dict for every feature.

    Returns
    -------
    woe_maps : { feature_name → { bin_value/interval → woe_float } }
    """
    return {
        feat: wt.set_index('bin')['WoE'].to_dict()
        for feat, wt in woe_tables.items()
    }


def apply_woe_encoding(
    df: pd.DataFrame,
    features: List[str],
    numeric_cols: List[str],
    woe_maps: Dict[str, dict],
    bins: int = 10,
    suffix: str = '_woe',
) -> pd.DataFrame:
    """
    Apply WoE encoding to a DataFrame using pre-computed woe_maps.
    Numeric features are binned with qcut; categorical mapped directly.

    Parameters
    ----------
    df           : input DataFrame
    features     : list of features to encode
    numeric_cols : which features are numeric (need binning)
    woe_maps     : output of build_woe_maps()
    bins         : number of quantile bins (must match what was used to build maps)
    suffix       : column name suffix for WoE columns

    Returns
    -------
    df_encoded : copy of df with new '{feature}_woe' columns appended
    """
    df_enc = df.copy()

    for feat in features:
        if feat not in woe_maps:
            continue
        wmap = woe_maps[feat]

        if feat in numeric_cols:
            bin_col = pd.qcut(df[feat], q=bins, duplicates='drop')
            df_enc[f'{feat}{suffix}'] = bin_col.map(wmap)
        else:
            df_enc[f'{feat}{suffix}'] = df[feat].map(wmap)

    return df_enc


def woe_encode_new(
    row_dict: dict,
    features: List[str],
    numeric_cols: List[str],
    woe_maps: Dict[str, dict],
    suffix: str = '_woe',
) -> dict:
    """
    WoE-encode a single applicant dict (used in scoring / Streamlit app).

    For numeric features, matches the raw value against the pandas Interval
    keys in woe_map. Falls back to nearest bin for edge values.

    Returns
    -------
    dict { '{feature}_woe' → float }
    """
    encoded = {}
    for feat in features:
        if feat not in woe_maps:
            continue
        raw  = row_dict.get(feat)
        wmap = woe_maps[feat]
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
                # edge case: clamp to nearest bin
                lefts = [iv.left for iv in wmap.keys()]
                wv = list(wmap.values())[0] if raw <= min(lefts) else list(wmap.values())[-1]
        else:
            wv = wmap.get(raw, 0.0)

        encoded[f'{feat}{suffix}'] = wv if wv is not None else 0.0

    return encoded


# ══════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════

def plot_iv_bar(
    iv_summary: pd.DataFrame,
    threshold: float = 0.02,
    figsize: tuple = (12, 6),
    title: str = 'Information Value by Feature',
) -> plt.Figure:
    """Horizontal bar chart of IV values with strength colour coding."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = []
    for iv in iv_summary['IV']:
        if iv < 0.02:   colors.append('#e74c3c')   # red  — useless
        elif iv < 0.10: colors.append('#e67e22')   # orange — weak
        elif iv < 0.30: colors.append('#f1c40f')   # yellow — medium
        elif iv < 0.50: colors.append('#2ecc71')   # green — strong
        else:           colors.append('#3498db')   # blue — suspicious

    bars = ax.barh(iv_summary['Feature'], iv_summary['IV'],
                   color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'IV threshold = {threshold}')

    for bar, val, label in zip(bars, iv_summary['IV'], iv_summary['Strength']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}  ({label})', va='center', fontsize=9)

    ax.set_xlabel('Information Value (IV)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def plot_woe_chart(
    woe_table: pd.DataFrame,
    feature: str,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Bar chart of WoE values per bin for one feature."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    bins_str = [str(b)[:30] for b in woe_table['bin']]
    colors   = ['#e74c3c' if w < 0 else '#2ecc71' for w in woe_table['WoE']]

    # WoE
    axes[0].bar(bins_str, woe_table['WoE'], color=colors, edgecolor='black', alpha=0.8)
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_title(f'{feature} — WoE per Bin', fontweight='bold')
    axes[0].set_ylabel('WoE')
    axes[0].tick_params(axis='x', rotation=40)

    # Default rate
    axes[1].bar(bins_str, woe_table['Default_Rate'] * 100,
                color='steelblue', edgecolor='black', alpha=0.8)
    axes[1].set_title(f'{feature} — Default Rate per Bin', fontweight='bold')
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].tick_params(axis='x', rotation=40)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# CLI usage
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from src.preprocessing import load_data, SELECTED_FEATURES, SELECTED_NUMERIC

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/german_credit_data.csv'
    df   = load_data(path)

    woe_tables, iv_summary = compute_all_woe_iv(
        df, SELECTED_FEATURES, 'target', SELECTED_NUMERIC
    )

    print("\n=== IV Summary ===")
    print(iv_summary.to_string(index=False))

    selected, dropped = select_features_by_iv(iv_summary, threshold=0.02)
    print(f"\nSelected ({len(selected)}): {selected}")
    print(f"Dropped  ({len(dropped)}):  {dropped}")
