# src/plots.py

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")


def plot_macro_series(macro_data):
    """
    Plot Growth YoY (Industrial Production) and Inflation YoY (CPI).
    Returns a matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(macro_data.index, macro_data["growth_yoy"], label="Growth YoY")
    axes[0].axhline(0.0, color="grey", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Growth YoY (%)")
    axes[0].set_title("Industrial Production – YoY Growth")
    axes[0].legend(loc="upper left")

    axes[1].plot(
        macro_data.index,
        macro_data["inflation_yoy"],
        label="Inflation YoY",
        color="tab:red",
    )
    axes[1].axhline(0.0, color="grey", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Inflation YoY (%)")
    axes[1].set_title("CPI – YoY Inflation")
    axes[1].legend(loc="upper left")

    axes[1].set_xlabel("Date")
    plt.tight_layout()
    return fig


def plot_regime_timeline(macro_regime_df):
    """
    Stacked Area Chart of smoothed Regime Probabilities over time.
    """
    prob_cols = [
        col
        for col in macro_regime_df.columns
        if col.startswith("Regime_") and not col.startswith("Filt")
    ]

    probs = macro_regime_df[prob_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(probs.index, probs.T, labels=prob_cols, alpha=0.7)

    ax.set_title("Historical Regime Probabilities (Smoothed)", fontsize=14)
    ax.set_ylabel("Probability")
    ax.set_xlim(probs.index[0], probs.index[-1])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")

    plt.tight_layout()
    return fig


def plot_growth_inflation_scatter(macro_regime_df):
    """
    Growth vs Inflation scatter, colored by ordered regime.
    """
    df = macro_regime_df.dropna(
        subset=["growth_yoy", "inflation_yoy", "RegimeOrdered"]
    ).copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    for regime in sorted(df["RegimeOrdered"].unique()):
        subset = df[df["RegimeOrdered"] == regime]
        ax.scatter(
            subset["inflation_yoy"],
            subset["growth_yoy"],
            s=20,
            alpha=0.7,
            label=f"Regime {regime}",
            color=cmap(regime % 10),
        )

    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1)

    ax.set_xlabel("Inflation YoY (%) – CPIAUCSL")
    ax.set_ylabel("Growth YoY (%) – INDPRO")
    ax.set_title("Growth vs Inflation, Colored by Ordered Regime")
    ax.legend(title="Ordered Regime", fontsize=9)

    plt.tight_layout()
    return fig


def plot_backtest_results(backtest_df):
    """
    Equity curve of regime strategy vs benchmark.
    """
    if hasattr(backtest_df.index, "to_timestamp"):
        x_index = backtest_df.index.to_timestamp()
    else:
        x_index = backtest_df.index

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_index,
        backtest_df["Cum_Strategy"],
        label="Regime Macro Strategy",
        linewidth=2,
        color="navy",
    )
    ax.plot(
        x_index,
        backtest_df["Cum_Benchmark"],
        label="Equal Weight Benchmark",
        linewidth=2,
        color="gray",
        linestyle="--",
    )

    ax.set_title(
        "Cumulative Performance: Macro Regime Model vs Benchmark", fontsize=14
    )
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
