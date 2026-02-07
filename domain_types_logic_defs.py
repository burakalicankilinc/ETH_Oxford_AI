# --- PURE DOMAIN TYPES & LOGIC ---
# Defined FIRST so they can be used as types in Effects

def calculate_brownian_params_pure(prices: pd.Series) -> BrownianParams:
    """Pure: Extract statistical parameters from data."""
    if len(prices) < 2:
        raise ValueError("Not enough data")

    daily_returns = ((prices / prices.shift(1)) - 1).dropna()
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)
    last_price = float(prices.iloc[-1])
    
    return {
        "mu": mu,
        "sigma": sigma,
        "last_price": last_price,
        "annual_vol": sigma * np.sqrt(252),
        "annual_drift": mu * 252
    }

def format_brownian_output_pure(sim_df: pd.DataFrame, ticker: str, params: BrownianParams) -> str:
    """Pure: Format the simulation results into text."""
    final_prices = sim_df.iloc[-1]
    low = np.percentile(final_prices, 5)
    high = np.percentile(final_prices, 95)
    mean_price = np.mean(final_prices)
    
    return (f"Brownian Motion Analysis for {ticker}:\n"
            f"--- TECHNICAL PARAMETERS ---\n"
            f"Annualized Volatility: {params['annual_vol']:.2%}\n"
            f"Annualized Drift: {params['annual_drift']:.2%}\n"
            f"Confidence Interval (90%): ${low:.2f} - ${high:.2f}\n"
            f"Mean Target: ${mean_price:.2f}")

def prepare_prophet_data_pure(data: pd.DataFrame) -> pd.DataFrame:
    """Pure Logic: Rename columns for Prophet."""
    df = data.reset_index()
    if 'Date' in df.columns:
        df['ds'] = df['Date'].dt.tz_localize(None)
    else:
        df['ds'] = df.index.tz_localize(None)
        
    if 'Close' in df.columns:
        df['y'] = df['Close']
    elif df.shape[1] > 0:
        df['y'] = df.iloc[:, 0]
        
    return df[['ds', 'y']]

def format_prophet_output(forecast: pd.DataFrame, ticker: str) -> str:
    """Pure transformation of Prophet results to text."""
    future_data = forecast.tail(30)
    latest_pred = forecast.iloc[-1]['yhat']
    trend = "UP" if latest_pred > forecast.iloc[0]['yhat'] else "DOWN"
    
    table = future_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_string()
    return (f"ML Analysis for {ticker}\n"
            f"Trend: {trend}\n"
            f"Forecast:\n{table}")