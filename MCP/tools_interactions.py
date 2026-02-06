from langchain.tools import tool
from mpc import MPCController
from predictor import SimpleGBMPredictor
import numpy as np
import yfinance as yf

@tool
def fetch_market_data(ticker: str):
    """Fetches real historical price data for a ticker to use in MPC."""
    df = yf.download(ticker, period="1mo", interval="1d")
    prices = df['Close'].values.flatten().tolist()
    return {"prices": prices, "current_price": prices[-1]}

@tool
def execute_mpc_optimization(ticker: str, trend_mu: float, prices: list):
    """
    Feeds the Agent's trend analysis into the MPC Controller.
    trend_mu: The drift value (mu) calculated by the Trend Agent.
    """
    predictor = SimpleGBMPredictor()
    # Define your allowed trades (e.g., Sell 10, Hold, Buy 10)
    controller = MPCController(predictor=predictor, action_grid=[-10, 0, 10])
    
    hist_arr = np.array(prices)
    result = controller.select_action(
        current_price=prices[-1],
        current_position=0,
        history=hist_arr,
        mu_override=trend_mu
    )
    return result