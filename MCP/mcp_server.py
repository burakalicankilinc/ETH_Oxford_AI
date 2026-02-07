from mcp.server.fastmcp import FastMCP
import numpy as np
import yfinance as yf
from mpc import MPCController
from predictor import SimpleGBMPredictor

# Initialize the MCP Server
mcp = FastMCP("Finance-Optimizer")

@mcp.tool()
def get_market_data(ticker: str) -> dict:
    """Fetches real-time market data for a stock ticker."""
    df = yf.download(ticker, period="1mo", interval="1d", progress=False)
    prices = df['Close'].values.flatten().tolist()
    return {"prices": prices, "current_price": prices[-1]}

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