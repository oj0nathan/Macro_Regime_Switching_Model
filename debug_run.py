# debug_run.py
# Small helper script to test the HMM pipeline outside of Streamlit.
from src.pipeline import run_full_pipeline
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Run the full macro + HMM + backtest pipeline
    results = run_full_pipeline()

    # Print the high-level performance table
    print("\n=== Final Metrics: Strategy vs Benchmark ===")
    print(results["final_metrics"])

    print("\n=== Regime Summary ===")
    print(results["regime_summary"])
    
    print("\n=== Per-Regime Attribution ===")
    print(results["attribution_table"])
