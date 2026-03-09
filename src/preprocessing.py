"""
src/preprocessing.py
--------------------
Column renaming, target encoding, feature group definitions.
Used by all notebooks and app.py.
"""

import pandas as pd
import numpy as np


# ── Column mapping (German → English) ─────────────────────────────────────
COLUMN_MAPPING = {
    'laufkont' : 'checking_account',
    'laufzeit' : 'duration_months',
    'moral'    : 'credit_history',
    'verw'     : 'purpose',
    'hoehe'    : 'credit_amount',
    'sparkont' : 'savings_account',
    'beszeit'  : 'employment_years',
    'rate'     : 'installment_rate',
    'famges'   : 'personal_status',
    'buerge'   : 'other_debtors',
    'wohnzeit' : 'residence_years',
    'verm'     : 'property',
    'alter'    : 'age',
    'weitkred' : 'other_installments',
    'wohn'     : 'housing',
    'bishkred' : 'existing_credits',
    'beruf'    : 'job',
    'pers'     : 'dependents',
    'telef'    : 'telephone',
    'gastarb'  : 'foreign_worker',
    'kredit'   : 'target',
}

# ── Feature groups ─────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'duration_months',
    'credit_amount',
    'age',
    'installment_rate',
    'residence_years',
    'existing_credits',
    'dependents',
]

CATEGORICAL_COLS = [
    'checking_account',
    'credit_history',
    'purpose',
    'savings_account',
    'employment_years',
    'personal_status',
    'other_debtors',
    'property',
    'other_installments',
    'housing',
    'job',
    'telephone',
    'foreign_worker',
]

# Confirmed significant features from Phase 2 hypothesis testing (IV >= 0.02)
SELECTED_NUMERIC = [
    'duration_months',
    'credit_amount',
    'age',
    'installment_rate',
]

SELECTED_CATEGORICAL = [
    'checking_account',
    'credit_history',
    'purpose',
    'savings_account',
    'employment_years',
    'personal_status',
    'other_debtors',
    'property',
    'other_installments',
    'housing',
    'foreign_worker',
]

SELECTED_FEATURES = SELECTED_NUMERIC + SELECTED_CATEGORICAL


# ── Readable label maps ────────────────────────────────────────────────────
LABEL_MAPS = {
    'checking_account': {
        1: '< 0 DM (negative)',
        2: '0–200 DM',
        3: '> 200 DM',
        4: 'No account',
    },
    'credit_history': {
        0: 'No credits taken',
        1: 'All credits paid duly',
        2: 'Existing credits paid duly',
        3: 'Delay in past',
        4: 'Critical account',
    },
    'purpose': {
        0: 'New car',      1: 'Used car',   2: 'Furniture/equipment',
        3: 'Radio/TV',     4: 'Appliances', 5: 'Repairs',
        6: 'Education',    7: 'Vacation',   8: 'Retraining',
        9: 'Business',    10: 'Other',
    },
    'savings_account': {
        1: '< 100 DM',
        2: '100–500 DM',
        3: '500–1000 DM',
        4: '> 1000 DM',
        5: 'Unknown/None',
    },
    'employment_years': {
        1: 'Unemployed',
        2: '< 1 year',
        3: '1–4 years',
        4: '4–7 years',
        5: '> 7 years',
    },
    'personal_status': {
        1: 'Male: divorced/separated',
        2: 'Female: divorced/separated/married',
        3: 'Male: single',
        4: 'Male: married/widowed',
    },
    'other_debtors': {
        1: 'None',
        2: 'Co-applicant',
        3: 'Guarantor',
    },
    'property': {
        1: 'Real estate',
        2: 'Savings/insurance',
        3: 'Car or other',
        4: 'Unknown/None',
    },
    'other_installments': {
        1: 'Bank',
        2: 'Stores',
        3: 'None',
    },
    'housing': {
        1: 'Rent',
        2: 'Own',
        3: 'Free',
    },
    'job': {
        1: 'Unemployed/unskilled (non-resident)',
        2: 'Unskilled (resident)',
        3: 'Skilled employee',
        4: 'Management/self-employed',
    },
    'telephone': {
        1: 'None',
        2: 'Yes (registered)',
    },
    'foreign_worker': {
        1: 'Yes',
        2: 'No',
    },
}


# ══════════════════════════════════════════════════════════════════════════
# Functions
# ══════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw German Credit CSV, rename columns to English,
    encode target correctly.

    Parameters
    ----------
    filepath : str
        Path to german_credit_data.csv

    Returns
    -------
    pd.DataFrame with English column names and binary target
        (0 = Good, 1 = Default)
    """
    df = pd.read_csv(filepath)
    df = rename_columns(df)
    df = encode_target(df)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename German column names to English."""
    existing = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    return df.rename(columns=existing)


def encode_target(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Ensure target is binary: 0 = Good (no default), 1 = Default.

    Handles two versions of the dataset:
      - UCI original : 1 = Good, 2 = Bad  → map {1:0, 2:1}
      - Kaggle version: 1 = Good, 0 = Bad  → map {1:0, 0:1}
    """
    df = df.copy()
    raw_vals = set(df[target_col].unique())

    if raw_vals == {1, 2}:
        df[target_col] = df[target_col].map({1: 0, 2: 1})
    elif raw_vals == {0, 1}:
        df[target_col] = df[target_col].map({1: 0, 0: 1})
    else:
        raise ValueError(
            f"Unexpected target values {raw_vals}. "
            "Expected {{0,1}} or {{1,2}}."
        )

    assert df[target_col].isin([0, 1]).all(), "Target encoding failed."
    return df


def get_feature_groups(df: pd.DataFrame):
    """
    Return (numeric_cols, categorical_cols) that are present in df,
    using the confirmed selected features from Phase 2.
    """
    num  = [c for c in SELECTED_NUMERIC     if c in df.columns]
    cat  = [c for c in SELECTED_CATEGORICAL if c in df.columns]
    return num, cat


def basic_info(df: pd.DataFrame) -> None:
    """Print basic dataset info: shape, dtypes, missing values, default rate."""
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Shape            : {df.shape}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    if 'target' in df.columns:
        print(f"Default rate     : {df['target'].mean()*100:.1f}%")
        print(f"  Good  (0)      : {(df['target']==0).sum()}")
        print(f"  Default (1)    : {(df['target']==1).sum()}")
    print()
    print(df.dtypes.to_string())


def detect_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Detect outliers using the IQR method for a list of numeric columns.

    Returns a summary DataFrame with outlier counts and bounds.
    """
    rows = []
    for col in cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 1.5 * IQR
        hi  = Q3 + 1.5 * IQR
        n   = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append({
            'Feature'    : col,
            'Lower Bound': round(lo, 2),
            'Upper Bound': round(hi, 2),
            'Outliers'   : n,
            'Outlier %'  : round(n / len(df) * 100, 2),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/german_credit_data.csv'
    df   = load_data(path)
    basic_info(df)
    print("\nOutlier summary:")
    print(detect_outliers_iqr(df, SELECTED_NUMERIC).to_string(index=False))
