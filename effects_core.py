import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import TypeVar, Callable, Generic, Any
from prophet import Prophet
from valyu import Valyu

T = TypeVar("T")
U = TypeVar("U")

# --- 1. THE IO MONAD (The Core Architecture) ---
class IO(Generic[T]):
    """
    A pure description of a side-effectful computation.
    Nothing runs until .unsafe_run() is called.
    """
    def __init__(self, effect: Callable[[], T]):
        self.effect = effect

    @staticmethod
    def pure(value: T) -> "IO[T]":
        return IO(lambda: value)

    @staticmethod
    def fail(error: Exception) -> "IO[Any]":
        def _raise(): raise error
        return IO(_raise)

    def map(self, f: Callable[[T], U]) -> "IO[U]":
        return IO(lambda: f(self.effect()))

    def flat_map(self, f: Callable[[T], "IO[U]"]) -> "IO[U]":
        return IO(lambda: f(self.effect()).unsafe_run())

    def attempt(self) -> "IO[T | Exception]":
        def _safe_run():
            try:
                return self.effect()
            except Exception as e:
                return e
        return IO(_safe_run)

    def unsafe_run(self) -> T:
        """The 'Edge' - actually executes the side effects."""
        return self.effect()

# --- 2. EFFECT INVENTORY (The Actions) ---

def fetch_stock_history_io(ticker: str, years: int = 2) -> IO[pd.DataFrame]:
    """Effect: Network Call to Yahoo Finance."""
    def _fetch():
        end_date = pd.Timestamp.today().normalize()
        start_date = end_date - pd.DateOffset(years=years)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Cleanup logic
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
            if isinstance(data, pd.DataFrame) and ticker in data.columns:
                 data = data[ticker]
        elif 'Close' in data.columns:
            data = data['Close']
        if isinstance(data, pd.DataFrame):
             data = data.iloc[:, 0]
        return data
    return IO(_fetch)

def run_monte_carlo_io(mu: float, sigma: float, last_price: float, days: int = 30) -> IO[pd.DataFrame]:
    """Effect: Random Number Generation & Simulation."""
    def _sim():
        scenarios = 1000
        dt = 1
        returns = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=(days, scenarios))
        price_paths = np.vstack([np.full((1, scenarios), last_price), last_price * np.exp(np.cumsum(returns, axis=0))])
        return pd.DataFrame(price_paths)
    return IO(_sim)

def prophet_predict_io(df: pd.DataFrame, days: int = 30) -> IO[pd.DataFrame]:
    """Effect: Heavy Computation / Model Training."""
    def _train_and_predict():
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        return forecast
    return IO(_train_and_predict)

def valyu_search_io(query: str) -> IO[str]:
    """Effect: External API Search."""
    def _search():
        client = Valyu(api_key=os.environ.get("VALYU_API_KEY"))
        return str(client.answer(query))
    return IO(_search)

# --- 3. MISSING EFFECTS (Persistence & Charting) ---

def save_to_ledger_io(ticker: str, model_name: str, result_text: str) -> IO[str]:
    """Effect: Appends result to a local JSON ledger file."""
    def _save():
        entry = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "model": model_name,
            "summary": result_text[:200] + "..."
        }
        with open("prediction_ledger.json", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return result_text
    return IO(_save)

def save_chart_io(forecast_df: pd.DataFrame, ticker: str) -> IO[str]:
    """Effect: Renders and saves a matplotlib chart to disk."""
    def _plot():
        try:
            filename = f"{ticker}_forecast.png"
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            # Handle both Prophet (yhat) and Monte Carlo (mean)
            if 'yhat' in forecast_df.columns:
                plt.plot(forecast_df['ds'], forecast_df['yhat'])
                plt.title(f"{ticker} Prophet Forecast")
            else:
                plt.plot(forecast_df.mean(axis=1))
                plt.title(f"{ticker} Monte Carlo Forecast")
            plt.savefig(filename)
            plt.close()
            return f"\n[Chart saved to {filename}]"
        except Exception:
            return ""
    return IO(_plot)