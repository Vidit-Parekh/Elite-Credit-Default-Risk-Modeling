# 🏦 Credit Default Risk Modeling
### Statistical Inference · Predictive Modeling · Credit Scorecard · Portfolio Risk Simulation

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4051B5?style=flat)](https://www.statsmodels.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-006400?style=flat)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> **End-to-end credit risk system** — from raw data to a deployed FICO-style scorecard and Monte Carlo portfolio simulator.  
> Built the way banks actually do it: WoE encoding, statistical logistic regression, Basel II expected loss framework.

🔗 **Live Demo →[LINK](https://elite-credit-default-risk-modeling-cgcud3q8eea8hvdvxrlqv9.streamlit.app/))**

---

## 📌 Table of Contents

- [Business Problem](#-business-problem)
- [What Makes This Different](#-what-makes-this-different)
- [Project Architecture](#-project-architecture)
- [Pipeline Overview](#-pipeline-overview)
- [Phase Breakdown](#-phase-breakdown)
- [Key Results](#-key-results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [The Streamlit App](#-the-streamlit-app)
- [Interview Talking Points](#-interview-talking-points)
- [Future Work](#-future-work)

---

## 🎯 Business Problem

> **Can we predict whether a loan applicant will default — and quantify how much risk they add to a bank's portfolio?**

This is one of the most important problems in finance. Every major bank (JPMorgan, Goldman Sachs, American Express, Capital One) runs a version of this pipeline internally.

The system answers three questions:

| Question | Output |
|---|---|
| Will this applicant default? | Probability of Default (PD) |
| What is their credit score? | FICO-style score (300–850 range) |
| How much portfolio loss should we expect? | Expected Loss, VaR @ 95% & 99% |

---

## 🔥 What Makes This Different

Most ML students build a credit model like this:

```
Load data → Train XGBoost → Print accuracy → Done
```

This project does what **senior data scientists and risk analysts** actually do:

| What Most Do | What This Project Does |
|---|---|
| Train XGBoost, show accuracy | Statistical logistic regression with p-values, odds ratios, confidence intervals |
| Skip feature selection | Weight of Evidence (WoE) + Information Value (IV) — the finance standard |
| Ignore class imbalance | Hypothesis testing to validate features, cost-sensitive threshold optimization |
| Use default 0.5 threshold | Business cost analysis — FN ≠ FP in credit risk |
| Call it done at AUC | KS Statistic, Gini Coefficient, Calibration Curve |
| No business output | FICO-style scorecard with points per bin |
| No portfolio thinking | Monte Carlo simulation, VaR, Economic Capital, Stress Testing |

---

## 🏗️ Project Architecture

```
Raw Data
    │
    ▼
┌─────────────────────────────────┐
│  Phase 1: EDA + WoE/IV Analysis │  ← Feature selection (finance method)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 2: Hypothesis Testing    │  ← Statistical validation of features
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 3: Logistic Regression   │  ← statsmodels: p-values, odds ratios
│           (Statistical Way)     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 4: Advanced Modeling     │  ← Random Forest + XGBoost + SHAP
│           + Model Comparison    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 5: Credit Scorecard      │  ← FICO-style points system (Basel II)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 6: Portfolio Simulation  │  ← Monte Carlo, VaR, Stress Testing
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Phase 7: Streamlit App         │  ← Live credit risk calculator
└─────────────────────────────────┘
```

---

## 🔬 Pipeline Overview

### 1. Weight of Evidence (WoE) + Information Value (IV)

WoE measures how strongly each feature category separates defaulters from non-defaulters:

$$WoE_i = \ln\left(\frac{\%\ Events_i}{\%\ NonEvents_i}\right)$$

IV quantifies total predictive power:

$$IV = \sum_{i=1}^{n} (\%\ Events_i - \%\ NonEvents_i) \times WoE_i$$

| IV Range | Predictive Power |
|---|---|
| < 0.02 | Useless — dropped |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| 0.3 – 0.5 | Strong ✅ |
| > 0.5 | Suspicious |

**Top features by IV:**
| Feature | IV | Strength |
|---|---|---|
| checking_account | 0.6613 | Strong* |
| credit_history | 0.2923 | Medium |
| duration_months | 0.2439 | Medium |
| savings_account | 0.1888 | Medium |
| purpose | 0.1654 | Medium |

*`checking_account` IV > 0.5 is a known legitimate result in credit risk literature — not data leakage.

---

### 2. Hypothesis Testing

Every feature was statistically validated before modeling:

| Feature Type | Test Used | Why |
|---|---|---|
| Numeric | Mann-Whitney U | Non-parametric; financial data is non-normal |
| Categorical | Chi-Square | Tests independence from default status |
| Effect size | Rank-biserial r / Cramér's V | p-value alone is misleading |

**Result:** 15 out of 16 features statistically significant at α = 0.05.

---

### 3. Logistic Regression — The Statistical Way

Used `statsmodels` (not sklearn) to get full statistical output:

```python
result = sm.Logit(y_train, X_train_sm).fit(method='newton')
print(result.summary2())
```

**Why this matters for interviews:**
- Every coefficient has a p-value and confidence interval
- Odds ratios are directly interpretable:  
  *"A 1-unit increase in WoE increases the odds of default by e^β"*
- McFadden's R² for goodness-of-fit
- AIC/BIC for model comparison
- Wald test for individual coefficient significance

---

### 4. Model Comparison

Three models compared on 6 finance-specific metrics:

| Model | ROC-AUC | KS Stat | Gini | PR-AUC | Brier | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7765 | 0.4310 | 0.5531 | — | 0.1687 | — |
| Random Forest | — | — | — | — | — | — |
| XGBoost (Optuna) | — | — | — | — | — | — |

*(Run the notebook to see full comparison results)*

**KS Statistic benchmarks** (finance-specific):
- < 0.20 → Poor
- 0.20 – 0.40 → Average
- 0.40 – 0.60 → **Good ✅ (our model)**
- > 0.60 → Excellent

---

### 5. Credit Scorecard

Converts logistic regression to a **FICO-style points system**:

$$\text{Factor} = \frac{PDO}{\ln(2)}, \quad \text{Offset} = \text{Base Score} - \text{Factor} \times \ln(\text{Base Odds})$$

$$\text{Points}_{ij} = -\left(\beta_i \times WoE_{ij} + \frac{\beta_0}{k}\right) \times \text{Factor}$$

**Parameters:**
- Base Score: 600
- PDO (Points to Double Odds): 20
- Base Odds: 1:1

**Score Bands:**
| Score | Risk Band | Decision |
|---|---|---|
| 750+ | Very Low Risk | Auto Approve ✅ |
| 650 – 749 | Low Risk | Approve ✅ |
| 550 – 649 | Medium Risk | Manual Review 🟡 |
| 450 – 549 | High Risk | Decline ❌ |
| < 450 | Very High Risk | Auto Decline ❌ |

---

### 6. Portfolio Risk Simulation

**Basel II Expected Loss Framework:**

$$EL = PD \times LGD \times EAD$$

Where:
- **PD** — Probability of Default (from our model)
- **LGD** — Loss Given Default (45%, Beta-distributed stochastic)
- **EAD** — Exposure at Default (loan amount)

**Monte Carlo:** 10,000 simulations → empirical loss distribution

**Risk Metrics:**

| Metric | Description |
|---|---|
| Expected Loss (EL) | Average loss across all simulations |
| VaR @ 95% | Loss exceeded in only 5% of scenarios |
| VaR @ 99% | Loss exceeded in only 1% of scenarios |
| CVaR @ 99% | Average loss given you're in the 99% tail |
| Economic Capital | VaR(99%) − EL = capital buffer needed |

**Stress Testing:** 5 scenarios from Base Case (1×PD) to Extreme (3×PD), following Basel III requirements.

---

## 📊 Key Results

| Metric | Value | Benchmark |
|---|---|---|
| ROC-AUC | 0.7765 | > 0.70 = Good ✅ |
| KS Statistic | 0.4310 | > 0.40 = Good ✅ |
| Gini Coefficient | 0.5531 | > 0.40 = Good ✅ |
| McFadden R² | 0.2695 | 0.20–0.40 = Good fit ✅ |
| Brier Score | 0.1687 | Lower is better ✅ |
| Significant Features | 15/16 | — |
| Features Dropped (IV < 0.02) | 5/20 | — |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `statsmodels` | Statistical logistic regression (p-values, odds ratios, MLE) |
| `scipy` | Hypothesis testing (Mann-Whitney U, Chi-Square, Shapiro-Wilk) |
| `optbinning` / custom WoE | Weight of Evidence encoding |
| `xgboost` | Gradient boosting model |
| `optuna` | Hyperparameter tuning (TPE algorithm) |
| `shap` | Model explainability (global + individual) |
| `scikit-learn` | Model evaluation, train/test split, Random Forest |
| `numpy` | Monte Carlo simulation (vectorized) |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `streamlit` | Interactive web application |

---

## 📂 Project Structure

```
Credit-Default-Risk-Model/
│
├── data/
│   └── german_credit_data.csv
│
├── notebooks/
│   ├── 00_EDA.ipynb                    ← Exploratory Data Analysis
│   ├── 02_WoE_IV_Analysis.ipynb        ← Weight of Evidence + Information Value
│   ├── 03_Hypothesis_Testing.ipynb     ← Statistical feature validation
│   ├── 04_Logistic_Regression.ipynb    ← Statistical modeling (statsmodels)
│   ├── 05_Advanced_Modeling.ipynb      ← RF + XGBoost + SHAP + comparison
│   ├── 06_Scorecard_System.ipynb       ← FICO-style credit scorecard
│   └── 07_Portfolio_Risk_Simulation.ipynb  ← Monte Carlo, VaR, stress testing
│
├── src/
│   ├── preprocessing.py                ← Column renaming, target encoding
│   ├── woe_iv.py                       ← WoE/IV computation from scratch
│   ├── modeling.py                     ← Model training utilities
│   ├── evaluation.py                   ← KS, Gini, calibration metrics
│   └── scorecard.py                    ← Scorecard point calculation
│
├── outputs/
│   ├── credit_scorecard.csv            ← Full scorecard table
│   ├── applicants_scored.csv           ← All applicants scored
│   └── risk_summary.csv                ← Portfolio risk metrics
│
├── app.py                              ← Streamlit application
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Credit-Default-Risk-Model.git
cd Credit-Default-Risk-Model

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the **German Credit Dataset** from Kaggle:  
👉 [https://www.kaggle.com/datasets/uciml/german-credit](https://www.kaggle.com/datasets/uciml/german-credit)

Place `german_credit_data.csv` in the `data/` folder.

### Run the Notebooks

Run in order:

```bash
jupyter notebook notebooks/00_EDA.ipynb
jupyter notebook notebooks/02_WoE_IV_Analysis.ipynb
jupyter notebook notebooks/03_Hypothesis_Testing.ipynb
jupyter notebook notebooks/04_Logistic_Regression.ipynb
jupyter notebook notebooks/05_Advanced_Modeling.ipynb
jupyter notebook notebooks/06_Scorecard_System.ipynb
jupyter notebook notebooks/07_Portfolio_Risk_Simulation.ipynb
```

### Launch the App

```bash
# Place german_credit_data.csv in the same folder as app.py
streamlit run app.py
```

---

## 🖥️ The Streamlit App

The app has 4 pages:

### 🎯 Page 1 — Score an Applicant
- Input loan details, demographics, financial history
- Get instant credit score + default probability
- See risk band (Very Low → Very High) and decision (Approve / Review / Decline)
- View points breakdown per feature — explains WHY the score is what it is

### 📊 Page 2 — Model Performance
- ROC-AUC, KS Statistic, Gini Coefficient
- ROC Curve + Score Distribution (Good vs Default)
- Default Rate by Score Band

### 📉 Page 3 — Portfolio Risk
- Interactive Monte Carlo simulation (adjustable simulations, LGD, stress multiplier)
- EL, VaR 95%, VaR 99%, Economic Capital metrics
- Loss distribution + CDF charts
- Business interpretation

### 🃏 Page 4 — Scorecard Table
- Full scorecard filterable by feature
- Points color-coded (green = adds score, red = subtracts)
- Points range per feature

---

## 💬 Interview Talking Points

These are the answers to questions interviewers at finance/DS firms actually ask:

**"Why did you use statsmodels instead of sklearn for logistic regression?"**
> *In credit risk, interpretability is a regulatory requirement. statsmodels gives me p-values, confidence intervals, and odds ratios — every coefficient can be explained to regulators. sklearn is optimized for prediction, not statistical inference.*

**"What is WoE encoding and why use it?"**
> *Weight of Evidence transforms features to log-odds scale, handles both numeric and categorical features uniformly, naturally deals with outliers through binning, and makes logistic regression coefficients linearly additive. It's the standard in credit scoring since the 1960s.*

**"Interpret this logistic regression coefficient."**
> *A 1-unit increase in the WoE of checking_account multiplies the odds of default by e^β — if β = 0.8, that's a 2.2× increase in default odds, holding all else constant.*

**"What is the KS Statistic?"**
> *KS measures the maximum separation between cumulative distributions of scores for Good and Default customers. Banks use it as a primary model discrimination metric. A KS above 0.40 is considered good — ours is 0.43.*

**"Why Monte Carlo over the analytical Vasicek model?"**
> *Monte Carlo makes fewer assumptions, handles portfolio heterogeneity naturally, supports stochastic LGD, and gives the full loss distribution — not just its moments. The cost is computational, but 10,000 simulations is fast with vectorization.*

**"What is Economic Capital?"**
> *Economic Capital = VaR(99%) − Expected Loss. It's the capital buffer a bank must hold to survive a 1-in-100 loss scenario. Basel II requires banks to hold this as tier 1 capital.*

**"How did you handle class imbalance?"**
> *Three approaches: class_weight='balanced' in Random Forest, scale_pos_weight in XGBoost, and cost-sensitive threshold optimization in logistic regression — because misclassifying a default (false negative) costs far more than a false rejection.*

---

## 🔮 Future Work

- [ ] **Bayesian Logistic Regression** — incorporate prior beliefs about default rates
- [ ] **Survival Analysis** — model time-to-default, not just default/no-default
- [ ] **Reject Inference** — handle the fact that we only observe outcomes for approved loans
- [ ] **Dynamic PD Models** — update scores as macroeconomic conditions change
- [ ] **IFRS 9 Staging** — classify loans into Stage 1/2/3 based on credit deterioration
- [ ] **API Deployment** — wrap the scorer in a FastAPI endpoint for real-time scoring
- [ ] **Platt Scaling** — post-hoc calibration of XGBoost probabilities

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Vidit**  
Data Science Portfolio Project  
Built with statistical rigor, business intuition, and finance domain knowledge.

[![GitHub](https://img.shields.io/badge/GitHub-yourusername-181717?style=flat&logo=github)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/yourusername)

---

> *"Most students train XGBoost and show accuracy. This project does what risk teams at JPMorgan actually do."*
