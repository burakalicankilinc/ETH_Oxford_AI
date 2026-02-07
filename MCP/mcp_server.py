import logging
import sys
import os

# Create a log file so we can see errors even if the terminal is "frozen"
logging.basicConfig(
    filename="mcp_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("MCP Server script started")

# Redirect ALL library noise to the log file so it doesn't break the JSON protocol
sys.stderr = open("mcp_debug.log", "a")

from mcp import FastMCP
import numpy as np
import yfinance as yf
from mpc import MPCController
from predictor import SimpleGBMPredictor

# Initialize the MCP Server
mcp = FastMCP("Finance-Optimizer")

@mcp.tool()
def get_market_data(ticker: str) -> dict:
    """Fetches real-time market data for a stock ticker."""
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if df.empty:
            return {"error": f"No data found for {ticker}"}
        
        prices = df['Close'].values.flatten().tolist()
        return {
            "prices": prices, 
            "current_price": prices[-1],
            "ticker": ticker
        }
    except Exception as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

@mcp.tool()
def run_trading_mpc(prices: list[float], trend_mu: float) -> str:
    """
    Runs Model Predictive Control (MPC) to find the optimal trade.
    trend_mu: The drift/trend predicted by the AI.
    """
    predictor = SimpleGBMPredictor()
    controller = MPCController(predictor=predictor, action_grid=[-10, 0, 10])
    
    result = controller.select_action(
        current_price=prices[-1],
        current_position=0,
        history=np.array(prices),
        mu_override=trend_mu
    )
    
    return (f"MPC Recommendation: {result['action']} shares. "
            f"Expected Target Price: ${result['expected_final_price']:.2f}")

if __name__ == "__main__":
    mcp.run()