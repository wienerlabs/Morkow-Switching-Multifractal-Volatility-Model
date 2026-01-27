# 📊 MSM-VaR: Market Risk Measurement System

> **Statistical model for financial risk quantification using Markov-Switching Multifractal (MSM) and Value-at-Risk (VaR)**

---

## 🎯 What does this project do?

This project implements a **market risk measurement system** that answers the fundamental question in finance:

> *"How much can I lose tomorrow, in the worst reasonable case?"*

**⚠️ Important clarification:** This is a **risk measurement** model, NOT a crash prediction model. It doesn't predict when the market will fall, but **quantifies the current risk level** based on recent volatility.

---

## 🧠 How does it work? (Simple Explanation)

### The "Risk Thermometer" Analogy

Think of the model as a **thermometer for financial markets**:
- A medical thermometer doesn't predict when you'll get a fever, but tells you your temperature NOW
- Similarly, MSM-VaR doesn't predict crashes, but tells you how "hot" (volatile) the market is NOW

### Model Steps:

```
1. OBSERVE the market    →  Volatility from recent days
2. IDENTIFY the regime   →  Are we in a "calm" or "turbulent" period?
3. CALCULATE risk        →  "With 95% probability, I won't lose more than X%"
4. VALIDATE the model    →  Test if estimates were historically correct
```

---

## 📐 Mathematical Foundations

### 1. Markov-Switching Multifractal (MSM) Model

The model assumes the market can be in **K different states/regimes** (default 5):

| State | Description | Typical Volatility |
|-------|-------------|-------------------|
| 1 | Very calm market | ~0.3% per day |
| 2 | Normal-calm market | ~0.6% per day |
| 3 | Normal market | ~1.0% per day |
| 4 | Agitated market | ~1.8% per day |
| 5 | Crisis market | ~3.0%+ per day |

**Markov Transitions:** The market can switch from one state to another according to a **transition matrix**:
- High probability (~97%) of staying in the same state
- Low probability (~0.75%) of transitioning to any other state

**Bayesian Filtering:** Each day, the model:
1. Observes the realized return
2. Updates probabilities for each state using Bayes' rule
3. Calculates expected volatility as a weighted average

```
σ_t = Σ P(state_k | data) × σ_k
```

### 2. Value-at-Risk (VaR)

VaR answers: *"What's the maximum loss I'll suffer with probability α?"*

**Formula:**
```
VaR(α) = z_α × σ_{t|t-1}
```

Where:
- `z_α` = normal distribution quantile (e.g., -1.645 for α=5%)
- `σ_{t|t-1}` = FORECAST volatility (calculated BEFORE seeing the day's return)

**VaR(5%) Interpretation:**
> "There's only a 5% chance that tomorrow's loss will exceed this value"

### 3. Critical Distinction: Forecast vs. Filtered

| Type | Formula | When Calculated | Usage |
|------|---------|-----------------|-------|
| **Forecast** (σ_{t\|t-1}) | E[σ \| info up to t-1] | BEFORE day t | VaR, backtesting |
| **Filtered** (σ_t) | E[σ \| info up to t] | AFTER day t | Analysis, visualization |

**Why does it matter?** Using "filtered" volatility for VaR would introduce **look-ahead bias** - we'd use information we didn't have at decision time.

---

## ✅ Statistical Validation (Backtesting)

### Kupiec Test (Unconditional Coverage)

**Question:** *"Does the VaR breach frequency match the theoretical level?"*

For VaR(5%), we expect ~5% of days to have losses greater than VaR.

**Test Statistic:**
```
LR_UC = -2 × [ln L(π₀) - ln L(π̂)]

where:
- π₀ = 0.05 (theoretical frequency)
- π̂ = breaches / total days (empirical frequency)
```

**Interpretation:**
- p-value ≥ 0.05 → ✅ Correctly calibrated model
- p-value < 0.05 → ❌ Breach rate significantly differs from 5%

### Christoffersen Test (Independence)

**Question:** *"Are breaches independent or do they cluster?"*

A good model should have randomly dispersed breaches, not grouped ones.

**Breach Transition Matrix:**
```
              Tomorrow OK    Tomorrow Breach
Today OK          n₀₀             n₀₁
Today Breach      n₁₀             n₁₁
```

**Interpretation:**
- p-value ≥ 0.05 → ✅ Breaches are independent
- p-value < 0.05 → ❌ Breaches cluster (model underestimates risk persistence)

### Conditional Coverage (CC)

Combines both tests:
```
LR_CC = LR_UC + LR_IND ~ χ²(2)
```

---

## 🔧 Calibration Methods

The model offers 4 methods for parameter estimation:

### 1. MLE (Maximum Likelihood Estimation)
```python
calibrate_msm_advanced(returns, method='mle')
```
- **How it works:** Finds parameters that maximize the probability of observing the data
- **Advantages:** Statistically optimal, efficiently uses all information
- **Disadvantages:** May converge to local optima

### 2. Grid Search
```python
calibrate_msm_advanced(returns, method='grid')
```
- **How it works:** Tests all combinations on a parameter grid
- **Advantages:** Guarantees finding the best within the grid
- **Disadvantages:** Slow, limited by grid resolution

### 3. Empirical
```python
calibrate_msm_advanced(returns, method='empirical')
```
- **How it works:** Uses empirical quantiles of returns
- **Advantages:** Fast, robust, intuitive
- **Disadvantages:** Doesn't optimize likelihood

### 4. Hybrid (Recommended)
```python
calibrate_msm_advanced(returns, method='hybrid')
```
- **How it works:** MLE + iterative adjustment for breach rate
- **Advantages:** Combines statistical optimization with VaR calibration
- **Disadvantages:** More complex, slower

---

## 📊 Typical Results

### Example Output (BTC-USD)

```
============================================================
   MSM ADVANCED CALIBRATION - Method: HYBRID
============================================================
   Returns: 4,235 observations
   Empirical std: 3.421%
   Target VaR breach: 5.0%

   CALIBRATION RESULTS
============================================================
   σ_low:    1.2847%
   σ_high:   8.9234%
   p_stay:   0.9712
   
   Sigma states: [1.285, 1.957, 2.981, 4.539, 8.923]

   --- Quality Metrics (In-Sample) ---
   VaR breach rate: 5.02% (target: 5.0%)  ✅
   Corr(|r|, σ):    0.684  [in-sample]
   Log-likelihood:  -8234.52
   AIC: 16475.04
   BIC: 16494.18
============================================================

--- Kupiec / Christoffersen Backtests ---
Kupiec UC: LR=0.024 | p-value=0.8762          ✅ PASS
Christoffersen IND: LR=1.234 | p-value=0.2667 ✅ PASS
Conditional Coverage: LR=1.258 | p-value=0.5331 ✅ PASS
```

### Results Interpretation (In-Sample)

**⚠️ Note:** These metrics are calculated on training data (in-sample). For realistic evaluation, out-of-sample validation is recommended.

| Metric | Value | Meaning |
|--------|-------|---------|
| VaR breach rate | 5.02% | Almost exactly 5% - well-calibrated model (in-sample) |
| Corr(\|r\|, σ) | 0.684 | In-sample correlation between estimated volatility and \|r\| |
| Kupiec p-value | 0.876 | ≥0.05 → Correct breach rate |
| Christoffersen p-value | 0.267 | ≥0.05 → Independent breaches |

### Expected Out-of-Sample Performance

| Metric | In-Sample | Out-of-Sample (typical) |
|--------|-----------|-------------------------|
| Corr(\|r\|, σ) | ~0.68 | ~0.30 |
| VaR breach rate | ~5.0% | 4-7% |

**Note:** Out-of-sample correlation of ~0.3 is realistic and indicates genuine predictive power compared to alternatives (GARCH: ~0.25-0.35, constant volatility: ~0).

---

## 🚀 How to Use

### Installation

```bash
# Clone the repository
git clone https://github.com/Johan948/MSM_VAR_MODEL.git
cd MSM_VAR_MODEL

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Run complete analysis
python MSM-VaR_MODEL.py
```

### Customization

In the `MSM-VaR_MODEL.py` file, modify:

```python
# Asset symbol (crypto, stocks, indices)
ticker = "BTC-USD"       # Bitcoin
ticker = "^SPX"          # S&P 500
ticker = "AAPL"          # Apple

# Forecast date
FORECAST_DATE = "2026-01-27"

# Calibration method
CALIBRATION_METHOD = 'hybrid'  # 'mle', 'grid', 'empirical', 'hybrid'
```

---

## 📁 Project Structure

```
MSM_VAR_MODEL/
├── MSM-VaR_MODEL.py      # Main script
├── README.md             # Documentation (Romanian)
├── README_EN.md          # Documentation (English - this file)
├── requirements.txt      # Python dependencies
└── output/               # Charts and results (optional)
    └── var_backtest.png
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.8+ |
| **Data Processing** | NumPy, Pandas |
| **Statistics** | SciPy (optimize, stats) |
| **Visualization** | Matplotlib |
| **Financial Data** | yfinance (Yahoo Finance API) |

---

## 📚 Academic References

1. **Calvet, L. E., & Fisher, A. J. (2004)**
   *"How to Forecast Long-Run Volatility: Regime Switching and the Estimation of Multifractal Processes"*
   Journal of Financial Econometrics, 2(1), 49-83.

2. **Kupiec, P. H. (1995)**
   *"Techniques for Verifying the Accuracy of Risk Measurement Models"*
   The Journal of Derivatives, 3(2), 73-84.

3. **Christoffersen, P. F. (1998)**
   *"Evaluating Interval Forecasts"*
   International Economic Review, 39(4), 841-862.

4. **Hamilton, J. D. (1989)**
   *"A New Approach to the Economic Analysis of Nonstationary Time Series"*
   Econometrica, 57(2), 357-384.

---

## ⚖️ Limitations and Disclaimer

### What the model CAN do:
- ✅ Quantify current risk based on recent volatility
- ✅ Estimate VaR with rigorous statistical validation
- ✅ Identify volatility regimes (calm vs. turbulent)
- ✅ Provide conditional tail probabilities based on current regime

### What the model CANNOT do:
- ❌ **Does NOT predict crashes** before they happen
- ❌ **Does NOT provide trading signals** (buy/sell)
- ❌ **Does NOT guarantee profits** or loss protection
- ❌ **Does NOT capture "black swan" events** (extreme rare events)

### Disclaimer
> This model is developed for educational and research purposes. It does not constitute financial advice. Past performance does not guarantee future results. Any investment decision should be made in consultation with an authorized financial professional.

---

## 👤 Author

**Tontici Sergiu**

📧 Email: tonticisergiu236@gmail.com  
🔗 LinkedIn: [linkedin.com/in/sergiu-tontici-71aa96361](https://www.linkedin.com/in/sergiu-tontici-71aa96361/)  
💻 GitHub: [github.com/Johan948](https://github.com/Johan948)

---

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributions

Contributions are welcome! For major changes, please open an issue first to discuss what you'd like to change.

```bash
# Fork repository
# Create feature branch
git checkout -b feature/FeatureName

# Commit changes
git commit -m 'Add FeatureName'

# Push to branch
git push origin feature/FeatureName

# Open Pull Request
```

---

<p align="center">
  <i>Project developed with 📊 for understanding financial risk</i>
</p>
