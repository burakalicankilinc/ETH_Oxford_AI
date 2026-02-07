import numpy as np
import pandas as pd
from effects_system import GetMarketData, RunProphet, ValyuSearch, SaveToAuditLog, LogInfo

# --- WORKFLOW 1: BROWNIAN (Same as before) ---
def brownian_workflow(ticker: str):
    yield LogInfo(f"Starting Brownian for {ticker}")
    prices = yield GetMarketData(ticker)
    
    if prices is None or len(prices) < 2: return "Error: No data"
    
    # Pure Math...
    daily_returns = ((prices / prices.shift(1)) - 1).dropna()
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)
    annual_vol = sigma * np.sqrt(252)
    
    yield SaveToAuditLog(ticker, {"volatility": annual_vol}, "Brownian")
    return f"Brownian Volatility: {annual_vol:.2%}"

# --- WORKFLOW 2: PROPHET (Machine Learning) ---
def prophet_workflow(ticker: str):
    """Pure Logic for ML Tool"""
    yield LogInfo(f"Starting Prophet ML for {ticker}")
    
    # 1. Get Data
    prices = yield GetMarketData(ticker, years=2)
    if prices is None or len(prices) < 10: return "Error: Not enough data for ML"
    
    # 2. Run Prophet (Delegate complex fit to Runtime)
    forecast = yield RunProphet(history_df=prices, days=30)
    
    # 3. Process Result (Pure)
    future_data = forecast.tail(30)
    latest_pred = forecast.iloc[-1]['yhat']
    trend = "UP" if latest_pred > prices.iloc[-1] else "DOWN"
    
    # 4. Save
    yield SaveToAuditLog(ticker, {"trend": trend, "target": latest_pred}, "Prophet")
    
    # 5. Return Formatted String
    return (f"Prophet ML Analysis for {ticker}:\n"
            f"Trend: {trend}\n"
            f"30-Day Target: ${latest_pred:.2f}\n"
            f"(Verified: ML Run logged).")

# --- WORKFLOW 3: SEARCH (Research Agent) ---
def search_workflow(query: str):
    """Pure Logic for News Search"""
    yield LogInfo(f"Searching Valyu for: {query}")
    
    # 1. Execute Search
    results = yield ValyuSearch(query)
    
    # 2. Save
    yield SaveToAuditLog("SEARCH_QUERY", {"query": query}, "ValyuSearch")
    
    return f"Search Results:\n{results[:500]}..." # Return first 500 chars
