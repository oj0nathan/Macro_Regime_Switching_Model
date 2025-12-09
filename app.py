import streamlit as st
from src.pipeline import run_full_pipeline

st.set_page_config(
    page_title="Macro Regime HMM Dashboard",
    layout="wide",
)

@st.cache_data
def get_data():
    """Run the full HMM + backtest pipeline once and cache the result."""
    return run_full_pipeline()


def main():
    st.title("Macro Regime Detection with Hidden Markov Models")

    st.markdown(
        """
This dashboard uses a Hidden Markov Model (HMM) on US growth and inflation proxies
to infer latent macro regimes, and shows how those regimes relate to asset returns
and a simple regime-based allocation strategy.
        """
    )

    data = get_data()

    # Sidebar navigation (now only 3 main sections)
    st.sidebar.header("Navigation")
    view = st.sidebar.radio(
        "Choose a view:",
        (
            "Overview",
            "Macro regimes",
            "Strategy performance",
        ),
    )

    # OVERVIEW (dynamic equity curve + metrics)
    if view == "Overview":
        st.subheader("Strategy vs Benchmark – Summary")

        metrics_df = data["final_metrics"].copy()

        # Full backtest results (for filtering & plotting)
        bt_full = data["backtest_results"].copy()
        # Convert PeriodIndex -> Timestamp for plotting
        if hasattr(bt_full.index, "to_timestamp"):
            bt_full.index = bt_full.index.to_timestamp()

        # Sidebar date controls for the equity curve
        min_date = bt_full.index.min().date()
        max_date = bt_full.index.max().date()

        start_date = st.sidebar.date_input(
            "Backtest start",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
        end_date = st.sidebar.date_input(
            "Backtest end",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

        # Ensure start <= end
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        mask = (bt_full.index.date >= start_date) & (bt_full.index.date <= end_date)
        bt = bt_full.loc[mask]

        equity_curves = bt[["Cum_Strategy", "Cum_Benchmark"]].rename(
            columns={
                "Cum_Strategy": "Regime Macro Strategy",
                "Cum_Benchmark": "Equal Weight Benchmark",
            }
        )

        # Layout: big chart on the left, metrics on the right
        col_chart, col_metrics = st.columns([3, 1])

        with col_chart:
            st.markdown("#### Equity Curve")
            st.line_chart(equity_curves)

        with col_metrics:
            st.markdown("#### Key Metrics (Strategy)")
            st.metric("Total Return", metrics_df.loc["Total Return", "Strategy"])
            st.metric("Ann. Vol", metrics_df.loc["Ann. Vol", "Strategy"])
            st.metric("Sharpe", metrics_df.loc["Sharpe", "Strategy"])
            st.metric("Max Drawdown", metrics_df.loc["Max Drawdown", "Strategy"])

        st.markdown("#### Full Performance Table")
        st.dataframe(metrics_df)

        with st.expander("What is a Hidden Markov Model?"):
            st.write(
                """
A Hidden Markov Model assumes that the economy moves through a small number
of unobserved 'states' (regimes). We only observe noisy variables such as
growth and inflation. The HMM estimates which regime is most likely at each
point in time, as well as the probability of being in each regime.
                """
            )

    # MACRO REGIMES (all macro/HMM views) 
    elif view == "Macro regimes":
        st.subheader("Macro Regimes")

        tab1, tab2, tab3 = st.tabs(
            ["Macro series", "Regime probabilities", "Growth vs inflation"]
        )

        #  Tab 1: macro series 
        with tab1:
            st.markdown("### Growth and Inflation Time Series")

            df = data["macro_data"].copy()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Growth YoY (INDPRO)")
                st.line_chart(df[["growth_yoy"]])
            with col2:
                st.markdown("#### Inflation YoY (CPI)")
                st.line_chart(df[["inflation_yoy"]])

        # Tab 2: regime probabilities 
        with tab2:
            st.markdown("### Historical Regime Probabilities (Smoothed)")

            mr = data["macro_with_regimes"].copy()
            prob_cols = [
                c
                for c in mr.columns
                if c.startswith("Regime_") and not c.startswith("Filt")
            ]
            probs = mr[prob_cols].copy()

            if hasattr(probs.index, "to_timestamp"):
                probs.index = probs.index.to_timestamp()

            st.area_chart(probs)

        # Tab 3: growth vs inflation scatter 
        with tab3:
            st.markdown("### Growth vs Inflation, Coloured by Regime")

            macro_named = data["macro_with_regimes_named"].copy()
            macro_named = macro_named.dropna(
                subset=["growth_yoy", "inflation_yoy", "RegimeName"]
            )

            plot_df = macro_named[
                ["inflation_yoy", "growth_yoy", "RegimeName"]
            ].rename(
                columns={
                    "inflation_yoy": "Inflation YoY",
                    "growth_yoy": "Growth YoY",
                }
            )

            col_chart, col_table = st.columns([3, 1])

            with col_chart:
                st.markdown("#### Growth–Inflation Scatter")
                st.scatter_chart(
                    plot_df,
                    x="Inflation YoY",
                    y="Growth YoY",
                    color="RegimeName",
                )

            with col_table:
                st.markdown("#### Regime Averages (Growth & Inflation)")
                regime_summary = data["regime_summary"]
                means = regime_summary.xs("mean", axis=1, level=1)
                means = means.rename(
                    columns={
                        "growth_yoy": "Growth mean",
                        "inflation_yoy": "Inflation mean",
                    }
                )
                st.dataframe(means)

            st.markdown(
                """
Each colour corresponds to a macro regime. Low growth / high inflation
points tend to map to 'Crisis / Stagflation', while high growth / low inflation
points look more like 'Goldilocks' periods.
                """
            )

    # STRATEGY PERFORMANCE (all performance views)
    else:  # "Strategy performance"
        st.subheader("Strategy Performance")

        tab1, tab2, tab3 = st.tabs(
            ["By regime", "Backtest", "Attribution & correlations"]
        )

        # Tab 1: asset performance by regime 
        with tab1:
            st.markdown("### Annualised Returns, Volatility and Sharpe by Regime")
            st.dataframe(data["asset_regime_stats"])

        # Tab 2: full backtest curves & stats
        with tab2:
            st.markdown("### Regime-Based Allocation vs Equal-Weight Benchmark")

            bt = data["backtest_results"].copy()
            if hasattr(bt.index, "to_timestamp"):
                bt.index = bt.index.to_timestamp()

            equity_curves = bt[["Cum_Strategy", "Cum_Benchmark"]].rename(
                columns={
                    "Cum_Strategy": "Regime Macro Strategy",
                    "Cum_Benchmark": "Equal Weight Benchmark",
                }
            )

            st.markdown("#### Equity Curves")
            st.line_chart(equity_curves)

            st.markdown("#### Summary statistics")
            st.dataframe(data["final_metrics"])

        # Tab 3: attribution & correlations
        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Performance Attribution by Regime")
                st.dataframe(data["attribution_table"])

            with col2:
                st.markdown("### Key Correlations by Regime")
                st.dataframe(data["correlations"])


if __name__ == "__main__":
    main()