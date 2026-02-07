import numpy as np
import pandas as pd
from effects_core import (
    IO, 
    fetch_stock_history_io, 
    run_monte_carlo_io, 
    prophet_predict_io, 
    save_to_ledger_io, 
    save_chart_io,
    valyu_search_io
)

# --- PURE FUNCTIONS (Data transformation only) ---

def calculate_brownian_params_pure(prices: pd.Series) -> dict:
    if len(prices) < 2: raise ValueError("Not enough data")
    daily_returns = ((prices / prices.shift(1)) - 1).dropna()
    return {
        "mu": np.mean(daily_returns),
        "sigma": np.std(daily_returns),
        "last_price": float(prices.iloc[-1]),
        "annual_vol": np.std(daily_returns) * np.sqrt(252),
        "annual_drift": np.mean(daily_returns) * 252
    }

def format_brownian_output_pure(sim_df: pd.DataFrame, ticker: str, params: dict) -> str:
    final_prices = sim_df.iloc[-1]
    return (f"Brownian Motion Analysis for {ticker}:\n"
            f"Annualized Volatility: {params['annual_vol']:.2%}\n"
            f"Annualized Drift: {params['annual_drift']:.2%}\n"
            f"Mean Target: ${np.mean(final_prices):.2f}\n"
            f"90% CI: ${np.percentile(final_prices, 5):.2f} - ${np.percentile(final_prices, 95):.2f}")

def prepare_prophet_data_pure(data: pd.DataFrame) -> pd.DataFrame:
    df = data.reset_index()
    if 'Date' in df.columns: df['ds'] = df['Date'].dt.tz_localize(None)
    else: df['ds'] = df.index.tz_localize(None)
    df['y'] = df.iloc[:, 1] if 'Close' in df.columns else df.iloc[:, 0]
    return df[['ds', 'y']]

def format_prophet_output(forecast: pd.DataFrame, ticker: str) -> str:
    latest = forecast.iloc[-1]['yhat']
    return (f"Prophet ML Analysis for {ticker}:\n"
            f"30-Day Target: ${latest:.2f}\n"
            f"Trend: {'UP' if latest > forecast.iloc[0]['yhat'] else 'DOWN'}")

# --- PIPELINES (The Workflows) ---

def build_brownian_pipeline(ticker: str) -> IO[str]:
    """Constructs the Brownian Motion Effect Chain."""
    return (
        fetch_stock_history_io(ticker)
        .map(calculate_brownian_params_pure)
        .flat_map(lambda params: 
            run_monte_carlo_io(params['mu'], params['sigma'], params['last_price'])
            .flat_map(lambda sim_df: 
                # Run parallel effects: Save Chart AND Save Ledger
                save_chart_io(sim_df, ticker).flat_map(lambda chart_msg:
                    save_to_ledger_io(
                        ticker, 
                        "Brownian", 
                        format_brownian_output_pure(sim_df, ticker, params)
                    ).map(lambda text: text + chart_msg)
                )
            )
        )
    )

def build_ml_pipeline(ticker: str) -> IO[str]:
    """Constructs the Machine Learning Effect Chain."""
    return (
        fetch_stock_history_io(ticker)
        .map(prepare_prophet_data_pure)
        .flat_map(lambda df: prophet_predict_io(df))
        .flat_map(lambda forecast:
            save_chart_io(forecast, ticker).flat_map(lambda chart_msg:
                save_to_ledger_io(
                    ticker, 
                    "Prophet", 
                    format_prophet_output(forecast, ticker)
                ).map(lambda text: text + chart_msg)
            )
        )
    )

def build_search_pipeline(query: str) -> IO[str]:
    return valyu_search_io(query)