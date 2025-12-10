# Macro Regime-Switching Model for Multi-Asset Allocation

Streamlit Dashboard: https://macroregimeswitchingmodel-rzlhm7tgpmfxv2f5umjsak.streamlit.app/

This project tests whether a **small set of macroeconomic indicators** can systematically improve multi-asset allocation decisions. I build a Hidden Markov Model (HMM) on U.S. growth and inflation, map the latent states into intuitive macro regimes (crisis, stagnation, expansion, boom), and then run a **regime-aware backtest** on a simple 5-asset universe (SPY, TLT, HYG, DBC, GLD).

The code is written in Python using `pandas`, `NumPy`, `statsmodels`, `pandas_datareader`, `yfinance`, and `matplotlib`, and is implemented end-to-end in a Jupyter notebook.

---

## TL;DR

* Estimated a **4-state Hidden Markov Model (HMM)** on U.S. Industrial Production growth using `statsmodels` MarkovRegression, with regime-specific means and variances to capture very different behaviour in recessions vs expansions.
* Interpreted the latent states as macro regimes using **growth and inflation**:

  * Crisis / Deep Recession
  * Slow Growth / Stagnation
  * Expansion / Goldilocks
  * Boom / Overheating
* Linked these regimes to a **5-asset universe** (SPY, TLT, HYG, DBC, GLD) and measured regime-conditional returns, Sharpe ratios, and regime-dependent correlations (e.g. stock–bond correlation flips sign across regimes).
* Built a **regime-based asset allocation strategy** that shifts weights by macro regime, using:

  * **Filtered** regime probabilities (real-time)
  * **1-month lag** on regime signals to reflect macro data release delays
* Backtested the regime strategy vs an equal-weight benchmark (2007–2025, monthly):

  * Total return: **349% vs 220%**
  * Ann. volatility: **9.9% vs 9.2%**
  * Sharpe: **0.88 vs 0.74**
  * Max drawdown: **−15.9% vs −24.4%**
* Added **performance attribution by regime** and **regime-dependent correlation tables** to understand *where* the alpha comes from and how diversification breaks down in different macro environments.

> **This is a research/learning project – not investment advice.**

---

## What this project demonstrates

* Design and implementation of a **regime-switching macro model** in Python using a Hidden Markov Model (HMM).
* Careful handling of **look-ahead bias**: use of filtered probabilities for trading, smoothed probabilities only for ex-post analysis, plus one-month lags to reflect macro data availability.
* Separation of **structural analysis** (long history of macro regimes) from **portfolio backtesting** (shorter asset history) while keeping a consistent regime taxonomy.
* Construction of a simple, interpretable **“playbook” of regime-specific weights** instead of black-box optimisation, and evaluation against a realistic multi-asset benchmark.
* Use of **regime-dependent correlations** to show how stock–bond–commodity relationships change across Crisis vs Boom regimes, and what that implies for diversification.

---

## Data & Universe

**Macro data (FRED via `pandas_datareader`):**

* `INDPRO` – Industrial Production Index
* `CPIAUCSL` – CPI, All Urban Consumers (Headline)

From these series the notebook constructs:

* `growth_yoy` – YoY log change of Industrial Production (proxy for real activity)
* `inflation_yoy` – YoY log change of CPI (headline inflation)

**Asset universe (Yahoo Finance via `yfinance`):**

* `SPY` – S&P 500
* `TLT` – Long-term U.S. Treasuries
* `HYG` – High Yield Credit
* `DBC` – Broad Commodities
* `GLD` – Gold

Daily adjusted close prices are resampled to **month-end** and converted into simple monthly returns.

---

## Model Overview

1. **Hidden Markov Model on Growth**

   * Fit a Gaussian HMM using `statsmodels.tsa.regime_switching.MarkovRegression` on `growth_yoy`.
   * Specification:

     * Regime-specific intercepts (different mean growth per regime).
     * Switching variance (volatility can differ across regimes).
   * Estimate models with 3 and 4 regimes, select **4-regime** model by **BIC**.
   * Extract:

     * **Smoothed** regime probabilities (best ex-post estimate of state at each date).
     * **Filtered** probabilities (real-time estimate given information up to that date).
     * Most likely regime per month.

2. **Economic Labelling of Regimes**

   * For each raw HMM state, compute average `growth_yoy` and `inflation_yoy`.
   * Sort states by mean growth (low → high) and map to ordered labels `0..3`.
   * Attach human-readable names:

     * 0 – Crisis / Deep Recession
     * 1 – Slow Growth / Stagnation
     * 2 – Expansion / Goldilocks
     * 3 – Boom / Overheating

3. **Structural Regime Analysis**

   * Align monthly asset returns with **smoothed** regimes (for ex-post analysis).
   * For each regime and asset compute:

     * Annualised return, volatility and Sharpe ratio.
   * Compute regime-specific **correlation matrices** (e.g. SPY–TLT, SPY–DBC, TLT–DBC) to see how diversification behaves in different macro environments.

4. **Trading Signal (Real-Time View)**

   * Use **filtered probabilities** rather than smoothed.
   * Convert filtered probabilities to a hard regime label via `argmax`.
   * Map raw filtered label → ordered regime index using the same ordering as above.
   * **Lag the regime signal by 1 month** so that month *t* portfolio uses regime inferred from data available at *t−1*.

5. **Regime-Based Portfolio Strategy**

   * Define a simple “playbook” of weights by regime over the 5 assets, e.g.:

     ```text
     0: Crisis / Deep Recession   → Long TLT & GLD, small SPY, no HYG/DBC
     1: Slow Growth / Stagnation  → Mix of SPY, HYG, GLD
     2: Expansion / Goldilocks    → Balanced SPY/TLT plus some HYG/DBC
     3: Boom / Overheating        → SPY + DBC heavy, low/zero TLT
     ```
---

## Disclaimer

This repository is for **educational and research purposes only**.
Nothing here constitutes investment advice or an offer to buy or sell any security.

