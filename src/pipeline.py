import numpy as np
import pandas as pd
from pandas_datareader import data as fred
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import yfinance as yf

# Global settings / defaults
START_MACRO = "1970-01-01"
START_ASSETS = "2007-05-01"
ASSET_LIST = ["SPY", "TLT", "HYG", "DBC", "GLD"]

# End of training sample for HMM estimation / model selection
# This is to conduct in-sample and out-of-sample
TRAIN_END = "2006-12-31"


# MACRO DATA & HMM
def fetch_macro_data(start_date: str = START_MACRO, end_date: str | None = None) -> pd.DataFrame:
    """
    Fetch macro data from FRED and build Growth and Inflation proxies.
    - Growth proxy: Industrial Production (INDPRO) YoY % change
    - Inflation proxy: CPI (CPIAUCSL) YoY % change
    """
    series_list = ["INDPRO", "CPIAUCSL"]

    macro_data = fred.DataReader(series_list, "fred", start_date, end_date)
    macro_data = macro_data.dropna()

    macro_data["growth_yoy"] = np.log(macro_data["INDPRO"]).diff(12) * 100.0
    macro_data["inflation_yoy"] = np.log(macro_data["CPIAUCSL"]).diff(12) * 100.0

    macro_data = macro_data.dropna()
    return macro_data


def fit_growth_regime_model(macro_data: pd.DataFrame, k_list: tuple[int, ...] = (3, 4)):
    """
    Original "fit on full sample" version (kept for reference / experimentation).
    Not used in the strict train/test pipeline, but handy for a fully in-sample structural model.
    """
    growth_series = macro_data["growth_yoy"].astype(float)

    best_result = None
    best_k = None
    best_bic = np.inf

    for k in k_list:
        try:
            model = MarkovRegression(
                endog=growth_series,
                k_regimes=k,
                trend="c",
                switching_variance=True,
            )

            result = model.fit(
                maxiter=1000,
                em_iter=20,
                search_reps=50,
                disp=False,
            )

            if result.bic < best_bic:
                best_bic = result.bic
                best_result = result
                best_k = k
        except Exception as e:
            print(f"Failed to fit {k}-regime model: {e}")

    if best_result is None:
        raise RuntimeError("All candidate regime models failed to converge.")

    print(f"[FULL] Selected {best_k}-regime model (BIC={best_bic:.2f})")
    return best_result


def fit_growth_regime_model_train_test(
    macro_data: pd.DataFrame,
    train_end: str = TRAIN_END,
    k_list: tuple[int, ...] = (3, 4),
):
    """
    Train/test-aware HMM fitting:

    1) Use data up to train_end to:
         - choose k (number of regimes) by BIC
         - estimate HMM parameters
    2) Freeze those parameters and run the model on the *full* sample
       (1970–present) to obtain smoothed / filtered probabilities.
    """

    # Full and train slices of growth series
    growth_full = macro_data["growth_yoy"].astype(float)
    growth_train = growth_full.loc[:train_end]

    if growth_train.empty:
        raise ValueError("Training sample is empty – check TRAIN_END date.")

    best_result = None
    best_k = None
    best_bic = np.inf

    # Step 1: fit on TRAIN sample only
    for k in k_list:
        try:
            model_train = MarkovRegression(
                endog=growth_train,
                k_regimes=k,
                trend="c",
                switching_variance=True,
            )

            result_train = model_train.fit(
                maxiter=1000,
                em_iter=20,
                search_reps=50,
                disp=False,
            )

            if result_train.bic < best_bic:
                best_bic = result_train.bic
                best_result = result_train
                best_k = k
        except Exception as e:
            print(f"Failed to fit {k}-regime TRAIN model: {e}")

    if best_result is None:
        raise RuntimeError("All candidate TRAIN models failed to converge.")

    print(
        f"[TRAIN] Selected {best_k}-regime model on "
        f"{growth_train.index[0].date()}–{train_end} (BIC={best_bic:.2f})"
    )

    # Step 2: apply the trained parameters to the FULL series
    full_model = MarkovRegression(
        endog=growth_full,
        k_regimes=best_k,
        trend="c",
        switching_variance=True,
    )
    # Only smooth using the parameters from the TRAIN fit (no re-estimation)
    full_result = full_model.smooth(best_result.params)

    return best_result, full_result, best_k


def attach_regime_probabilities(macro_data: pd.DataFrame, model_result):
    """
    Attach smoothed and filtered regime probabilities to the macro data.

    Returns:
        macro_with_regimes, smoothed_df, filtered_df
    """
    smoothed_df = model_result.smoothed_marginal_probabilities.add_prefix("Regime_")
    filtered_df = model_result.filtered_marginal_probabilities.add_prefix("FiltRegime_")

    macro_with_regimes = macro_data.join(smoothed_df, how="left")
    macro_with_regimes = macro_with_regimes.join(filtered_df, how="left")

    # Most likely regime per month (using smoothed probabilities – ex post view)
    macro_with_regimes["Regime"] = (
        smoothed_df.idxmax(axis=1).str.replace("Regime_", "").astype(int)
    )

    return macro_with_regimes, smoothed_df, filtered_df


def summarise_and_order_regimes(macro_regime_df: pd.DataFrame):
    """
    Summarise each regime's average Growth and Inflation,
    then impose a systematic ordering by Growth (low -> high).
    """
    summary = macro_regime_df.groupby("Regime")[["growth_yoy", "inflation_yoy"]].agg(
        ["mean", "std"]
    )

    # Order by mean growth (Crisis -> Stagnation -> Expansion -> Boom)
    summary_sorted = summary.sort_values(by=("growth_yoy", "mean"))

    ordered_regime_indices = list(summary_sorted.index)
    regime_order_map = {
        orig_regime: new_order
        for new_order, orig_regime in enumerate(ordered_regime_indices)
    }

    macro_ordered = macro_regime_df.copy()
    macro_ordered["RegimeOrdered"] = macro_ordered["Regime"].map(regime_order_map)

    return macro_ordered, summary_sorted, regime_order_map


def add_regime_names(macro_regime_df: pd.DataFrame):
    """
    Map ordered regime indices to human-readable macro regime names.
    """
    regime_name_map = {
        0: "Crisis / Deep Recession",
        1: "Slow Growth / Stagnation",
        2: "Expansion / Goldilocks",
        3: "Boom / Overheating",
    }

    df = macro_regime_df.copy()
    df["RegimeName"] = df["RegimeOrdered"].map(regime_name_map)

    return df, regime_name_map

# ASSET RETURNS & ALIGNMENT
def fetch_monthly_asset_returns(
    start_date: str = START_ASSETS,
    end_date: str | None = None,
    tickers: list[str] = ASSET_LIST,
) -> pd.DataFrame:
    """
    Fetch daily prices for the macro asset universe from Yahoo Finance,
    then convert them into monthly simple returns (Adj Close).
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        price_data = data["Adj Close"].copy()
    else:
        price_data = data["Adj Close"].to_frame()

    monthly_prices = price_data.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


def align_returns_with_regimes(
    asset_returns: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    regime_col: str = "RegimeOrdered",
):
    """
    Align monthly asset returns with monthly regime labels.
    """
    ar = asset_returns.copy()
    ar["Period"] = ar.index.to_period("M")
    ar = ar.groupby("Period").last()

    mr = macro_regime_df[[regime_col, "RegimeName"]].copy()
    mr["Period"] = mr.index.to_period("M")
    mr = mr.groupby("Period").last()

    common_periods = ar.index.intersection(mr.index)
    aligned_returns = ar.loc[common_periods]
    aligned_regimes = mr.loc[common_periods, regime_col]

    return aligned_returns, aligned_regimes


def compute_regime_performance(
    returns: pd.DataFrame,
    regimes: pd.Series,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """
    Compute annualised return, volatility and Sharpe by regime for each asset.
    """
    returns, regimes = returns.align(regimes, join="inner", axis=0)

    df = returns.copy()
    df["Regime"] = regimes

    all_results = []

    for regime, group in df.groupby("Regime"):
        r = group[returns.columns]

        mean_m = r.mean()
        vol_m = r.std()

        ann_return = mean_m * 12.0
        ann_vol = vol_m * np.sqrt(12.0)

        excess_return = ann_return - rf_annual
        sharpe = excess_return / ann_vol

        regime_stats = pd.DataFrame(
            {
                "Regime": regime,
                "Asset": mean_m.index,
                "AnnReturn": ann_return.values,
                "AnnVol": ann_vol.values,
                "Sharpe": sharpe.values,
            }
        )

        all_results.append(regime_stats)

    stats_df = pd.concat(all_results, ignore_index=True)
    return stats_df


# REGIME WEIGHTS & BACKTEST
DEFAULT_REGIME_WEIGHTS: dict[int, dict[str, float]] = {
    0: {"DBC": 0.0,  "GLD": 0.30, "HYG": 0.0,  "SPY": 0.10, "TLT": 0.60},  # Crisis
    1: {"DBC": 0.10, "GLD": 0.20, "HYG": 0.30, "SPY": 0.40, "TLT": 0.0},  # Stagnation / recovery
    2: {"DBC": 0.10, "GLD": 0.10, "HYG": 0.20, "SPY": 0.30, "TLT": 0.30},  # Goldilocks
    3: {"DBC": 0.40, "GLD": 0.10, "HYG": 0.10, "SPY": 0.40, "TLT": 0.0},  # Boom / overheating
}


def build_filtered_regime_column(
    macro_with_regimes_named: pd.DataFrame,
    regime_order_map: dict[int, int],
    filt_prefix: str = "FiltRegime_",
):
    """
    Use filtered probabilities (real-time) and map to ordered regime indices.
    """
    df = macro_with_regimes_named.copy()
    filt_cols = [c for c in df.columns if c.startswith(filt_prefix)]

    # Raw regime index based on max filtered probability (real-time)
    df["RegimeFiltRaw"] = (
        df[filt_cols].idxmax(axis=1).str.replace(filt_prefix, "").astype(int)
    )
    df["RegimeOrderedFilt"] = df["RegimeFiltRaw"].map(regime_order_map)

    return df


def run_regime_backtest(
    returns: pd.DataFrame,
    regimes: pd.Series,
    regime_weights: dict[int, dict[str, float]] = DEFAULT_REGIME_WEIGHTS,
):
    """
    Simulates a portfolio that changes weights based on the macro regime.
    Uses lagged filtered regime signal to avoid look-ahead.
    """
    signal = regimes.shift(1).dropna()

    common_idx = returns.index.intersection(signal.index)
    active_ret = returns.loc[common_idx]
    active_sig = signal.loc[common_idx]

    n_assets = len(returns.columns)
    benchmark_ret = active_ret.mean(axis=1)  # naive equal-weight benchmark

    strategy_returns = []

    for period in active_ret.index:
        current_regime = active_sig.loc[period]

        weights_dict = regime_weights.get(
            current_regime,
            {col: 1 / n_assets for col in returns.columns},  # fallback: equal-weight
        )
        w_vector = np.array(
            [weights_dict.get(col, 0.0) for col in active_ret.columns]
        )

        month_ret = np.dot(w_vector, active_ret.loc[period].values)
        strategy_returns.append(month_ret)

    backtest_df = pd.DataFrame(index=active_ret.index)
    backtest_df["Strategy"] = strategy_returns
    backtest_df["Benchmark"] = benchmark_ret

    backtest_df["Cum_Strategy"] = (1 + backtest_df["Strategy"]).cumprod()
    backtest_df["Cum_Benchmark"] = (1 + backtest_df["Benchmark"]).cumprod()

    return backtest_df


def calculate_final_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary performance table for Strategy vs Benchmark.
    """
    stats = {}

    for col in ["Strategy", "Benchmark"]:
        series = backtest_df[col]
        cum_ret = (1 + series).cumprod()

        tot_ret = cum_ret.iloc[-1] - 1
        ann_vol = series.std() * np.sqrt(12)
        sharpe = (series.mean() / series.std()) * np.sqrt(12)
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()

        stats[col] = {
            "Total Return": f"{tot_ret * 100:.1f}%",
            "Ann. Vol": f"{ann_vol * 100:.1f}%",
            "Sharpe": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd * 100:.1f}%",
        }

    return pd.DataFrame(stats)


def analyze_strategy_by_regime(
    backtest_df: pd.DataFrame,
    regime_signal: pd.Series,
    regime_names: dict[int, str],
) -> pd.DataFrame:
    """
    Decomposes the Strategy vs Benchmark performance by Macro Regime.
    """
    df = backtest_df.copy()
    common_idx = df.index.intersection(regime_signal.index)
    df = df.loc[common_idx]
    df["Regime"] = regime_signal.loc[common_idx]

    results = []

    for r_id in sorted(df["Regime"].unique()):
        subset = df[df["Regime"] == r_id]

        strat_ann_ret = subset["Strategy"].mean() * 12
        strat_ann_vol = subset["Strategy"].std() * np.sqrt(12)
        strat_sharpe = strat_ann_ret / strat_ann_vol if strat_ann_vol > 0 else 0

        bench_ann_ret = subset["Benchmark"].mean() * 12
        bench_ann_vol = subset["Benchmark"].std() * np.sqrt(12)
        bench_sharpe = bench_ann_ret / bench_ann_vol if bench_ann_vol > 0 else 0

        alpha = strat_ann_ret - bench_ann_ret

        results.append(
            {
                "Regime": regime_names.get(r_id, f"Regime {r_id}"),
                "Months": len(subset),
                "Strat Return": f"{strat_ann_ret * 100:.1f}%",
                "Bench Return": f"{bench_ann_ret * 100:.1f}%",
                "Alpha": f"{alpha * 100:.1f}%",
                "Strat Sharpe": f"{strat_sharpe:.2f}",
                "Bench Sharpe": f"{bench_sharpe:.2f}",
            }
        )

    return pd.DataFrame(results)


def compute_regime_correlations(
    returns: pd.DataFrame,
    regimes: pd.Series,
    regime_names: dict[int, str],
) -> pd.DataFrame:
    """
    Calculates key correlations by regime (e.g. SPY–TLT, SPY–DBC, TLT–DBC).
    """
    df = returns.copy()
    df["Regime"] = regimes
    df = df.dropna()

    results = []

    for r_id in sorted(df["Regime"].unique()):
        subset = df[df["Regime"] == r_id].drop(columns=["Regime"])
        corr = subset.corr()

        results.append(
            {
                "Regime": regime_names.get(r_id, f"Regime {r_id}"),
                "SPY-TLT": f"{corr.loc['SPY', 'TLT']:.2f}",
                "SPY-DBC": f"{corr.loc['SPY', 'DBC']:.2f}",
                "TLT-DBC": f"{corr.loc['TLT', 'DBC']:.2f}",
            }
        )

    return pd.DataFrame(results)

# CONVENIENCE PIPELINE WRAPPER
def run_full_pipeline(
    k_list: tuple[int, ...] = (3, 4),
    rf_annual: float = 0.0,
    train_end: str = TRAIN_END,
):
    """
    Runs the full pipeline:

        - Fetch macro
        - Fit HMM on TRAIN sample only (1970–train_end)
        - Apply trained parameters to FULL macro sample (1970–present)
        - Attach & order regimes
        - Name regimes
        - Fetch assets
        - Align returns
        - Compute per-regime stats
        - Build filtered (real-time) regime column
        - Run backtest & attribution

    Returns a dictionary of key DataFrames / objects.
    """
    # 1. Macro + HMM
    macro_data = fetch_macro_data()

    regime_train_result, regime_full_result, best_k = fit_growth_regime_model_train_test(
        macro_data,
        train_end=train_end,
        k_list=k_list,
    )

    macro_with_regimes, smoothed_probs, filtered_probs = attach_regime_probabilities(
        macro_data, regime_full_result
    )

    macro_with_regimes_ordered, regime_summary, regime_order_map = (
        summarise_and_order_regimes(macro_with_regimes)
    )
    macro_with_regimes_named, regime_name_map = add_regime_names(
        macro_with_regimes_ordered
    )

    # 2. Assets + alignment (structural, using smoothed/ordered regimes)
    asset_returns = fetch_monthly_asset_returns()
    aligned_returns, aligned_regimes = align_returns_with_regimes(
        asset_returns, macro_with_regimes_named, "RegimeOrdered"
    )

    asset_regime_stats = compute_regime_performance(
        aligned_returns, aligned_regimes, rf_annual=rf_annual
    )

    # 3. Build filtered (real-time) regimes for trading backtest
    macro_with_regimes_named = build_filtered_regime_column(
        macro_with_regimes_named, regime_order_map
    )
    aligned_returns_trading, aligned_regimes_trading = align_returns_with_regimes(
        asset_returns, macro_with_regimes_named, "RegimeOrderedFilt"
    )

    # 4. Backtest + attribution
    backtest_results = run_regime_backtest(
        aligned_returns_trading, aligned_regimes_trading, DEFAULT_REGIME_WEIGHTS
    )

    final_metrics = calculate_final_metrics(backtest_results)
    attribution_table = analyze_strategy_by_regime(
        backtest_results, aligned_regimes_trading, regime_name_map
    )

    correlations = compute_regime_correlations(
        aligned_returns, aligned_regimes, regime_name_map
    )

    return {
        "macro_data": macro_data,
        "regime_train_result": regime_train_result,
        "regime_full_result": regime_full_result,
        "best_k": best_k,
        "macro_with_regimes": macro_with_regimes,
        "macro_with_regimes_ordered": macro_with_regimes_ordered,
        "macro_with_regimes_named": macro_with_regimes_named,
        "regime_summary": regime_summary,
        "regime_order_map": regime_order_map,
        "regime_name_map": regime_name_map,
        "asset_returns": asset_returns,
        "aligned_returns": aligned_returns,
        "aligned_regimes": aligned_regimes,
        "asset_regime_stats": asset_regime_stats,
        "aligned_returns_trading": aligned_returns_trading,
        "aligned_regimes_trading": aligned_regimes_trading,
        "backtest_results": backtest_results,
        "final_metrics": final_metrics,
        "attribution_table": attribution_table,
        "correlations": correlations,
    }