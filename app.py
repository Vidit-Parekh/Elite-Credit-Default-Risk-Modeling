"""
Credit Default Risk — Streamlit App
Phases covered: Scorecard, Model Comparison, Portfolio Simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }

    /* Score card */
    .score-display {
        background: linear-gradient(135deg, #1c2128, #21262d);
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        border: 1px solid #30363d;
    }
    .score-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 5rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-band {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }

    /* Risk factor rows */
    .factor-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 0;
        background: #161b22;
        border-left: 3px solid;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        color: #8b949e;
        text-transform: uppercase;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Decision banner */
    .decision-approve  { background:#1a3a2a; border:1px solid #2ea043; color:#3fb950; border-radius:8px; padding:12px 20px; text-align:center; font-weight:700; font-size:1.1rem; }
    .decision-review   { background:#3a2e0a; border:1px solid #9e6a03; color:#d29922; border-radius:8px; padding:12px 20px; text-align:center; font-weight:700; font-size:1.1rem; }
    .decision-decline  { background:#3a1a1a; border:1px solid #da3633; color:#f85149; border-radius:8px; padding:12px 20px; text-align:center; font-weight:700; font-size:1.1rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        border-bottom: 1px solid #30363d;
        padding: 0 16px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e;
        border: none;
        padding: 12px 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }

    /* Slider labels */
    .stSlider label { color: #8b949e; font-size: 0.85rem; }

    /* Selectbox */
    .stSelectbox label { color: #8b949e; font-size: 0.85rem; }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }
    div[data-testid="stMetric"] label { color: #8b949e; }
    div[data-testid="stMetric"] div   { color: #e6edf3; font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL — cached so they only build once
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_prepare():
    """Load data, WoE-encode, train LR model. Returns everything needed."""
    try:
        data = pd.read_csv('/home/vidit/Public/Project/Customer Churn Prediction/Data/german_credit_data.csv')
    except FileNotFoundError:
        # Fallback: try common paths
        import os
        for path in ['data/german_credit_data.csv', '../data/german_credit_data.csv']:
            if os.path.exists(path):
                data = pd.read_csv(path)
                break
        else:
            st.error("❌ Could not find german_credit_data.csv — place it in the same folder as app.py")
            st.stop()

    column_mapping = {
        'laufkont':'checking_account', 'laufzeit':'duration_months',
        'moral':'credit_history',      'verw':'purpose',
        'hoehe':'credit_amount',       'sparkont':'savings_account',
        'beszeit':'employment_years',  'rate':'installment_rate',
        'famges':'personal_status',    'buerge':'other_debtors',
        'wohnzeit':'residence_years',  'verm':'property',
        'alter':'age',                 'weitkred':'other_installments',
        'wohn':'housing',              'bishkred':'existing_credits',
        'beruf':'job',                 'pers':'dependents',
        'telef':'telephone',           'gastarb':'foreign_worker',
        'kredit':'target'
    }
    data.rename(columns=column_mapping, inplace=True)

    raw_vals = sorted(data['target'].unique())
    if set(raw_vals) == {1, 2}:
        data['target'] = data['target'].map({1: 0, 2: 1})
    elif set(raw_vals) == {0, 1}:
        data['target'] = data['target'].map({1: 0, 0: 1})

    numeric_cols = ['duration_months', 'credit_amount', 'age', 'installment_rate']
    cat_cols     = [
        'checking_account', 'credit_history', 'purpose', 'savings_account',
        'employment_years', 'personal_status', 'other_debtors', 'property',
        'other_installments', 'housing', 'foreign_worker'
    ]
    features = numeric_cols + cat_cols

    # WoE encoding
    def compute_woe(df, feat, target, bins=10, is_num=True):
        d = df[[feat, target]].copy()
        te = d[target].sum(); tne = len(d) - te
        d['bin'] = pd.qcut(d[feat], q=bins, duplicates='drop') if is_num else d[feat]
        g = d.groupby('bin', observed=True)[target].agg(E='sum', T='count').reset_index()
        g['NE'] = g['T'] - g['E']; eps = 0.5
        g['pE']  = (g['E']  + eps) / (te  + eps)
        g['pNE'] = (g['NE'] + eps) / (tne + eps)
        g['WoE'] = np.log(g['pE'] / g['pNE'])
        g['DR']  = g['E'] / g['T']
        return g[['bin','T','E','NE','DR','WoE']].round(4)

    woe_maps   = {}
    woe_tables = {}
    df_enc     = data.copy()

    for feat in features:
        is_n = feat in numeric_cols
        try:
            wt = compute_woe(data, feat, 'target', bins=10, is_num=is_n)
            woe_tables[feat] = wt
            if is_n:
                df_enc[f'{feat}_bin'] = pd.qcut(data[feat], q=10, duplicates='drop')
                wmap = wt.set_index('bin')['WoE'].to_dict()
                df_enc[f'{feat}_woe'] = df_enc[f'{feat}_bin'].map(wmap)
                df_enc.drop(columns=[f'{feat}_bin'], inplace=True)
            else:
                wmap = wt.set_index('bin')['WoE'].to_dict()
                df_enc[f'{feat}_woe'] = data[feat].map(wmap)
            woe_maps[feat] = wmap
        except:
            pass

    woe_cols = [f'{f}_woe' for f in features if f'{f}_woe' in df_enc.columns]
    X = df_enc[woe_cols].dropna()
    y = df_enc.loc[X.index, 'target']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_sm = sm.add_constant(X_tr); X_te_sm = sm.add_constant(X_te)
    lr = sm.Logit(y_tr, X_tr_sm).fit(method='newton', maxiter=100, disp=False)

    # Scorecard constants
    PDO = 20; BASE_SCORE = 600; BASE_ODDS = 1
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS + 1e-9)
    intercept = lr.params['const']
    n_feat    = len(woe_cols)
    ipt_feat  = intercept / n_feat

    # Scorecard table
    sc_rows = []
    for feat in features:
        wc = f'{feat}_woe'
        if wc not in woe_cols: continue
        beta = lr.params.get(wc)
        if beta is None: continue
        for _, row in woe_tables[feat].iterrows():
            pts = -(float(beta) * float(row['WoE']) + ipt_feat) * factor
            sc_rows.append({'Feature': feat, 'Bin': str(row['bin']),
                            'WoE': round(float(row['WoE']),4), 'Points': round(pts,1),
                            'Beta': round(float(beta),4)})
    scorecard_df = pd.DataFrame(sc_rows)

    # Score full portfolio
    X_all_sm   = sm.add_constant(X)
    log_odds   = X_all_sm @ lr.params
    prob_all   = 1 / (1 + np.exp(-log_odds))
    scores_all = offset - factor * log_odds

    portfolio = data.loc[X.index, ['credit_amount','target']].copy()
    portfolio['PD']    = prob_all.values
    portfolio['score'] = scores_all.round().astype(int).values
    portfolio['EAD']   = portfolio['credit_amount']
    portfolio['EL']    = portfolio['PD'] * 0.45 * portfolio['EAD']

    # AUC
    y_prob_te = lr.predict(X_te_sm)
    auc = roc_auc_score(y_te, y_prob_te)
    fpr, tpr, _ = roc_curve(y_te, y_prob_te)
    ks  = float(np.max(tpr - fpr))

    return {
        'data': data, 'lr': lr, 'woe_maps': woe_maps, 'woe_tables': woe_tables,
        'woe_cols': woe_cols, 'features': features, 'numeric_cols': numeric_cols,
        'scorecard_df': scorecard_df, 'portfolio': portfolio,
        'factor': factor, 'offset': offset, 'intercept': intercept,
        'n_feat': n_feat, 'ipt_feat': ipt_feat,
        'auc': auc, 'ks': ks, 'fpr': fpr, 'tpr': tpr,
        'y_te': y_te, 'y_prob_te': y_prob_te,
        'X_te': X_te_sm
    }


def score_applicant(app_dict, m):
    """Score a single applicant dict → (score, prob, breakdown_df)."""
    row = {}
    for feat in m['features']:
        wc = f'{feat}_woe'
        if wc not in m['woe_cols']: continue
        raw = app_dict.get(feat)
        wmap = m['woe_maps'].get(feat, {})
        wv   = None
        if feat in m['numeric_cols']:
            for interval, w in wmap.items():
                try:
                    if raw in interval: wv = w; break
                except: pass
            if wv is None:
                lefts = [iv.left for iv in wmap.keys()]
                wv = list(wmap.values())[0] if raw <= min(lefts) else list(wmap.values())[-1]
        else:
            wv = wmap.get(raw, 0.0)
        row[wc] = wv if wv is not None else 0.0

    ws = pd.Series(row)
    ws_c = pd.concat([pd.Series({'const': 1.0}), ws])
    log_odds = m['lr'].params.reindex(ws_c.index).fillna(0).dot(ws_c)
    prob  = float(1 / (1 + np.exp(-log_odds)))
    score = int(round(m['offset'] - m['factor'] * log_odds))

    breakdown = []
    for feat in m['features']:
        wc = f'{feat}_woe'
        if wc not in m['woe_cols']: continue
        beta = float(m['lr'].params.get(wc, 0))
        wv   = row.get(wc, 0.0)
        pts  = -(beta * wv + m['ipt_feat']) * m['factor']
        breakdown.append({'Feature': feat, 'Value': app_dict.get(feat),
                          'WoE': round(wv,4), 'Points': round(pts,1)})
    return score, round(prob, 4), pd.DataFrame(breakdown)


def get_band(score):
    if score >= 750:   return 'VERY LOW RISK',  '#3fb950', 'Auto Approve',    'decision-approve'
    elif score >= 650: return 'LOW RISK',        '#3fb950', 'Approve',         'decision-approve'
    elif score >= 550: return 'MEDIUM RISK',     '#d29922', 'Manual Review',   'decision-review'
    elif score >= 450: return 'HIGH RISK',       '#f85149', 'Decline',         'decision-decline'
    else:              return 'VERY HIGH RISK',  '#f85149', 'Auto Decline',    'decision-decline'


def run_monte_carlo(pd_arr, ead_arr, n_sim=5000, lgd_mean=0.45, lgd_std=0.10):
    la = lgd_mean * ((lgd_mean*(1-lgd_mean))/lgd_std**2 - 1)
    lb = (1-lgd_mean) * ((lgd_mean*(1-lgd_mean))/lgd_std**2 - 1)
    u  = np.random.uniform(0, 1, size=(n_sim, len(pd_arr)))
    dm = (u < pd_arr).astype(float)
    lm = np.random.beta(abs(la)+0.01, abs(lb)+0.01, size=(n_sim, len(pd_arr)))
    return (dm * lm * ead_arr).sum(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner('🔧 Building models...'):
    m = load_and_prepare()

portfolio  = m['portfolio']
data       = m['data']


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Credit Risk Engine")
    st.markdown("<p style='color:#8b949e;font-size:0.8rem;'>German Credit Dataset · Logistic Regression · Monte Carlo</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p class='section-header'>Navigation</p>", unsafe_allow_html=True)
    page = st.radio("", [
        "🎯 Score an Applicant",
        "📊 Model Performance",
        "📉 Portfolio Risk",
        "🃏 Scorecard Table"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#8b949e; line-height:1.8;'>
    <b style='color:#58a6ff;'>Model AUC</b><br>
    """ + f"{m['auc']:.4f}" + """<br><br>
    <b style='color:#58a6ff;'>KS Statistic</b><br>
    """ + f"{m['ks']:.4f}" + """<br><br>
    <b style='color:#58a6ff;'>Portfolio Size</b><br>
    """ + f"{len(portfolio):,} loans" + """<br><br>
    <b style='color:#58a6ff;'>Total Exposure</b><br>
    DM """ + f"{portfolio['EAD'].sum():,.0f}" + """
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SCORE AN APPLICANT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Score an Applicant":
    st.markdown("## 🎯 Applicant Credit Scoring")
    st.markdown("<p style='color:#8b949e;'>Enter applicant details to compute credit score, default probability, and decision.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<p class='section-header'>Applicant Details</p>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            duration    = st.slider("Loan Duration (months)", 4, 72, 18, 2)
            credit_amt  = st.slider("Loan Amount (DM)", 250, 20000, 3000, 250)
            age         = st.slider("Age (years)", 18, 75, 35, 1)
            inst_rate   = st.slider("Installment Rate (% of income)", 1, 4, 2, 1)
        with c2:
            checking_ac = st.selectbox("Checking Account", [1,2,3,4],
                format_func=lambda x: {1:"< 0 DM (negative)", 2:"0–200 DM", 3:"> 200 DM", 4:"No account"}[x])
            savings_ac  = st.selectbox("Savings Account", [1,2,3,4,5],
                format_func=lambda x: {1:"< 100 DM", 2:"100–500 DM", 3:"500–1000 DM", 4:"> 1000 DM", 5:"Unknown/None"}[x])
            credit_hist = st.selectbox("Credit History", [0,1,2,3,4],
                format_func=lambda x: {0:"No credits", 1:"All paid", 2:"Existing paid", 3:"Delay in past", 4:"Critical"}[x])
            employment  = st.selectbox("Employment (years)", [1,2,3,4,5],
                format_func=lambda x: {1:"Unemployed", 2:"< 1yr", 3:"1–4yr", 4:"4–7yr", 5:"> 7yr"}[x])

        c3, c4 = st.columns(2)
        with c3:
            purpose      = st.selectbox("Loan Purpose", [0,1,2,3,4,5,6,7,8,9,10],
                format_func=lambda x: {0:"New car",1:"Used car",2:"Furniture",3:"Radio/TV",
                    4:"Appliances",5:"Repairs",6:"Education",7:"Vacation",8:"Retraining",9:"Business",10:"Other"}[x])
            housing      = st.selectbox("Housing", [1,2,3],
                format_func=lambda x: {1:"Rent",2:"Own",3:"Free"}[x])
        with c4:
            personal_st  = st.selectbox("Personal Status", [1,2,3,4],
                format_func=lambda x: {1:"Male divorced",2:"Female",3:"Male single",4:"Male married"}[x])
            other_inst   = st.selectbox("Other Installments", [1,2,3],
                format_func=lambda x: {1:"Bank",2:"Stores",3:"None"}[x])

        # Hidden features with sensible defaults
        app = {
            'duration_months': duration, 'credit_amount': credit_amt,
            'age': age, 'installment_rate': inst_rate,
            'checking_account': checking_ac, 'credit_history': credit_hist,
            'purpose': purpose, 'savings_account': savings_ac,
            'employment_years': employment, 'personal_status': personal_st,
            'other_debtors': 1, 'property': 2,
            'other_installments': other_inst, 'housing': housing,
            'foreign_worker': 2
        }

        score_btn = st.button("⚡ Calculate Credit Score", type="primary", use_container_width=True)

    with col_right:
        if score_btn or True:   # always show (updates live)
            score, prob, breakdown = score_applicant(app, m)
            band, color, decision, dec_class = get_band(score)

            # Score display
            gauge_color = color
            st.markdown(f"""
            <div class='score-display'>
                <div style='color:#8b949e; font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;'>Credit Score</div>
                <div class='score-number' style='color:{gauge_color};'>{score}</div>
                <div class='score-band' style='color:{gauge_color};'>{band}</div>
                <div style='color:#8b949e; font-size:0.85rem; margin-top:12px;'>Default Probability: <span style='color:{gauge_color}; font-family:IBM Plex Mono;'>{prob*100:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='{dec_class}'>Decision: {decision}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Risk Factor Breakdown</p>", unsafe_allow_html=True)

            bd_sorted = breakdown.sort_values('Points')
            for _, row in bd_sorted.iterrows():
                pts   = row['Points']
                bcolor = '#f85149' if pts < 0 else '#3fb950'
                bar_w  = min(abs(pts) / (breakdown['Points'].abs().max() + 1) * 100, 100)
                st.markdown(f"""
                <div class='factor-row' style='border-left-color:{bcolor};'>
                    <span style='color:#e6edf3;'>{row['Feature']}</span>
                    <span style='color:{bcolor}; font-weight:600;'>{pts:+.1f} pts</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            # Mini bar chart
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#161b22')
            bd_plot = breakdown.sort_values('Points')
            colors  = ['#f85149' if p < 0 else '#3fb950' for p in bd_plot['Points']]
            ax.barh(bd_plot['Feature'], bd_plot['Points'], color=colors, edgecolor='none', alpha=0.9)
            ax.axvline(0, color='#8b949e', linewidth=1)
            ax.tick_params(colors='#8b949e', labelsize=8)
            for spine in ax.spines.values(): spine.set_color('#30363d')
            ax.set_title('Points Contribution per Feature', color='#e6edf3', fontsize=9, pad=8)
            ax.set_xlabel('Points', color='#8b949e', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")
    st.markdown("<p style='color:#8b949e;'>Logistic Regression with WoE encoding — full statistical evaluation.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Top metrics
    auc  = m['auc']
    ks   = m['ks']
    gini = 2*auc - 1
    fpr_arr, tpr_arr = m['fpr'], m['tpr']

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ROC-AUC", f"{auc:.4f}", delta="Good (>0.70)")
    with c2:
        st.metric("KS Statistic", f"{ks:.4f}", delta="Good (>0.40)")
    with c3:
        st.metric("Gini Coefficient", f"{gini:.4f}", delta="Good (>0.40)")
    with c4:
        default_rate = data['target'].mean()
        st.metric("Portfolio Default Rate", f"{default_rate*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p class='section-header'>ROC Curve</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.plot(fpr_arr, tpr_arr, color='#58a6ff', linewidth=2.5,
                label=f'Logistic Regression (AUC={auc:.4f})')
        ax.plot([0,1],[0,1],'--', color='#8b949e', linewidth=1, label='Random (AUC=0.50)')
        ax.fill_between(fpr_arr, tpr_arr, alpha=0.1, color='#58a6ff')
        ax.set_xlabel('False Positive Rate', color='#8b949e')
        ax.set_ylabel('True Positive Rate', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        ax.legend(fontsize=9, facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
        for spine in ax.spines.values(): spine.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<p class='section-header'>Score Distribution by Class</p>", unsafe_allow_html=True)
        good_sc = portfolio[portfolio['target']==0]['score']
        bad_sc  = portfolio[portfolio['target']==1]['score']
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.hist(good_sc, bins=30, alpha=0.7, color='#3fb950',
                density=True, label=f'Good (n={len(good_sc)})')
        ax.hist(bad_sc,  bins=30, alpha=0.7, color='#f85149',
                density=True, label=f'Default (n={len(bad_sc)})')
        ax.axvline(good_sc.mean(), color='#3fb950', linestyle='--', linewidth=2,
                   label=f'Good mean: {good_sc.mean():.0f}')
        ax.axvline(bad_sc.mean(),  color='#f85149', linestyle='--', linewidth=2,
                   label=f'Default mean: {bad_sc.mean():.0f}')
        ax.set_xlabel('Credit Score', color='#8b949e')
        ax.set_ylabel('Density', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        ax.legend(fontsize=9, facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
        for spine in ax.spines.values(): spine.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Default rate by score band
    st.markdown("<br>")
    st.markdown("<p class='section-header'>Default Rate by Score Band</p>", unsafe_allow_html=True)

    def get_band_name(s):
        if s >= 750: return 'Very Low'
        elif s >= 650: return 'Low'
        elif s >= 550: return 'Medium'
        elif s >= 450: return 'High'
        else: return 'Very High'

    portfolio['band'] = portfolio['score'].apply(get_band_name)
    band_order = ['Very High','High','Medium','Low','Very Low']
    bstats = portfolio.groupby('band').agg(
        Count=('target','count'), DefaultRate=('target','mean')
    ).reindex([b for b in band_order if b in portfolio['band'].unique()])

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    band_colors = ['#f85149','#ff7b72','#d29922','#3fb950','#2ea043']
    existing = band_colors[-len(bstats):]
    bars = ax.bar(range(len(bstats)), bstats['DefaultRate']*100,
                  color=existing, edgecolor='none', alpha=0.9, width=0.6)
    ax.axhline(data['target'].mean()*100, color='#58a6ff', linestyle='--',
               linewidth=1.5, label=f'Portfolio avg {data["target"].mean()*100:.1f}%')
    for bar, val in zip(bars, bstats['DefaultRate']*100):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}%', ha='center', color='#e6edf3', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(bstats)))
    ax.set_xticklabels(bstats.index, color='#8b949e')
    ax.set_ylabel('Default Rate (%)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.legend(fontsize=9, facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    for spine in ax.spines.values(): spine.set_color('#30363d')
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PORTFOLIO RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Portfolio Risk":
    st.markdown("## 📉 Portfolio Risk Simulation")
    st.markdown("<p style='color:#8b949e;'>Monte Carlo simulation — Expected Loss, VaR, Stress Testing (Basel II framework).</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_ctrl, col_main = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown("<p class='section-header'>Simulation Parameters</p>", unsafe_allow_html=True)
        n_sim   = st.select_slider("Simulations", [1000, 2000, 5000, 10000], value=5000)
        lgd_m   = st.slider("LGD Mean (%)", 20, 80, 45) / 100
        lgd_s   = st.slider("LGD Std Dev (%)", 5, 20, 10) / 100
        pd_mult = st.slider("PD Stress Multiplier", 1.0, 3.0, 1.0, 0.25,
                            help="1.0 = base case, 2.0 = recession scenario")
        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    with col_main:
        if run_btn or 'port_losses' not in st.session_state:
            with st.spinner('Running Monte Carlo...'):
                pd_arr  = np.clip(portfolio['PD'].values * pd_mult, 0, 1)
                ead_arr = portfolio['EAD'].values
                np.random.seed(42)
                losses = run_monte_carlo(pd_arr, ead_arr, n_sim, lgd_m, lgd_s)
                st.session_state['port_losses'] = losses
                st.session_state['pd_arr']  = pd_arr
                st.session_state['ead_arr'] = ead_arr

        losses = st.session_state.get('port_losses',
                 run_monte_carlo(portfolio['PD'].values, portfolio['EAD'].values, 2000))

        el    = losses.mean()
        v95   = np.percentile(losses, 95)
        v99   = np.percentile(losses, 99)
        cv99  = losses[losses >= v99].mean()
        ec99  = v99 - el
        total = portfolio['EAD'].sum()

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Expected Loss",   f"DM {el:,.0f}",  f"{el/total*100:.2f}% of portfolio")
        with c2: st.metric("VaR @ 95%",        f"DM {v95:,.0f}", f"{v95/total*100:.2f}% of portfolio")
        with c3: st.metric("VaR @ 99%",        f"DM {v99:,.0f}", f"{v99/total*100:.2f}% of portfolio")
        with c4: st.metric("Economic Capital", f"DM {ec99:,.0f}", f"(VaR99 − EL)")

        st.markdown("<br>", unsafe_allow_html=True)

        # Loss distribution chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax in axes:
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#161b22')
            for spine in ax.spines.values(): spine.set_color('#30363d')
            ax.tick_params(colors='#8b949e')

        axes[0].hist(losses, bins=70, color='#58a6ff', alpha=0.7, density=True, edgecolor='none')
        axes[0].hist(losses[losses >= v99], bins=30, color='#f85149', alpha=0.8,
                     density=True, edgecolor='none', label='99% tail')
        axes[0].axvline(el,  color='#3fb950', linewidth=2, linestyle='-',  label=f'EL')
        axes[0].axvline(v95, color='#d29922', linewidth=2, linestyle='--', label=f'VaR 95%')
        axes[0].axvline(v99, color='#f85149', linewidth=2, linestyle='--', label=f'VaR 99%')
        axes[0].set_xlabel('Portfolio Loss (DM)', color='#8b949e')
        axes[0].set_ylabel('Density', color='#8b949e')
        axes[0].set_title('Loss Distribution', color='#e6edf3', fontsize=11)
        axes[0].legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))

        cdf = np.arange(1, len(losses)+1) / len(losses) * 100
        axes[1].plot(np.sort(losses), cdf, color='#58a6ff', linewidth=2)
        axes[1].axhline(95, color='#d29922', linestyle='--', linewidth=1.5, label=f'95th pct')
        axes[1].axhline(99, color='#f85149', linestyle='--', linewidth=1.5, label=f'99th pct')
        axes[1].axvline(v95, color='#d29922', linestyle=':', linewidth=1)
        axes[1].axvline(v99, color='#f85149', linestyle=':', linewidth=1)
        axes[1].fill_betweenx([99,100], v99, np.sort(losses).max(), alpha=0.15, color='#f85149')
        axes[1].set_xlabel('Portfolio Loss (DM)', color='#8b949e')
        axes[1].set_ylabel('Cumulative %', color='#8b949e')
        axes[1].set_title('Cumulative Loss Distribution', color='#e6edf3', fontsize=11)
        axes[1].legend(fontsize=8, facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
        axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("<br>")
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; font-size:0.9rem; color:#8b949e; line-height:2;'>
        <b style='color:#58a6ff;'>Interpretation</b><br>
        The bank should expect to lose <b style='color:#e6edf3;'>DM {el:,.0f}</b> on average from this portfolio.<br>
        In 1 out of 100 scenarios, losses will exceed <b style='color:#f85149;'>DM {v99:,.0f}</b>.<br>
        To be 99% safe, the bank should hold <b style='color:#d29922;'>DM {ec99:,.0f}</b> as capital buffer ({ec99/total*100:.2f}% of total exposure).
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SCORECARD TABLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🃏 Scorecard Table":
    st.markdown("## 🃏 Credit Scorecard")
    st.markdown("<p style='color:#8b949e;'>FICO-style points system. Each bin contributes fixed points to the total credit score.</p>", unsafe_allow_html=True)
    st.markdown("---")

    sc = m['scorecard_df']

    feat_filter = st.selectbox(
        "Filter by Feature",
        ["All Features"] + sorted(sc['Feature'].unique().tolist())
    )

    if feat_filter != "All Features":
        sc_show = sc[sc['Feature'] == feat_filter].copy()
    else:
        sc_show = sc.copy()

    # Color the Points column
    def color_points(val):
        if val > 0:   return 'color: #3fb950; font-weight: 600; font-family: IBM Plex Mono;'
        elif val < 0: return 'color: #f85149; font-weight: 600; font-family: IBM Plex Mono;'
        else:         return 'color: #8b949e;'

    styled = sc_show[['Feature','Bin','WoE','Points']].style\
        .applymap(color_points, subset=['Points'])\
        .set_properties(**{
            'background-color': '#161b22',
            'color': '#e6edf3',
            'border-color': '#30363d',
            'font-size': '13px'
        })\
        .format({'WoE': '{:.4f}', 'Points': '{:+.1f}'})

    st.dataframe(styled, use_container_width=True, height=500)

    st.markdown("<br>")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Features", sc['Feature'].nunique())
    with c2: st.metric("Total Bins",     len(sc))
    with c3: st.metric("Base Score",     "600 (PDO=20)")

    # Points range per feature
    st.markdown("<br>")
    st.markdown("<p class='section-header'>Points Range per Feature (Max−Min)</p>", unsafe_allow_html=True)
    pts_range = sc.groupby('Feature')['Points'].apply(lambda x: x.max()-x.min()).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.barh(pts_range.index, pts_range.values, color='#58a6ff', alpha=0.8, edgecolor='none')
    ax.set_xlabel('Points Range (Max − Min)', color='#8b949e')
    ax.tick_params(colors='#8b949e')
    ax.set_title('Feature Contribution Range', color='#e6edf3', fontsize=11)
    for spine in ax.spines.values(): spine.set_color('#30363d')
    plt.tight_layout()
    st.pyplot(fig); plt.close()
