import numpy as np
from effects_system import GetMarketData, SaveToAuditLog, LogInfo

def brownian_forecast_workflow(ticker: str):
    """
    Pure Generator. Describes the workflow but performs NO actions.
    """
    # 1. Ask Runtime for Data
    yield LogInfo(f"Starting analysis for {ticker}")
    prices = yield GetMarketData(ticker)
    
    if prices is None or len(prices) < 2:
        return "Error: Insufficient data."

    # 2. Pure Math (Deterministic)
    daily_returns = ((prices / prices.shift(1)) - 1).dropna()
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)
    
    annual_drift = mu * 252
    annual_vol = sigma * np.sqrt(252)
    
    last_price = prices.iloc[-1]
    prediction = last_price * np.exp(annual_drift * (30/252))
    
    # 3. Ask Runtime to Save
    payload = {
        "start_price": float(last_price),
        "target_price": float(prediction),
        "volatility": float(annual_vol)
    }
    yield SaveToAuditLog(ticker, payload, "BrownianMotion")
    
    # 4. Return Summary to Agent
    return (f"Brownian Motion Analysis for {ticker}:\n"
            f"- Volatility: {annual_vol:.2%}\n"
            f"- Drift: {annual_drift:.2%}\n"
            f"- 30-Day Target: ${prediction:.2f}\n"
            f"(Verified: Result persisted to audit log).")