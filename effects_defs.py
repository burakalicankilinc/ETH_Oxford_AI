# --- EFFECT DEFINITIONS (I/O Boundary) ---

def fetch_stock_history_io(ticker: str, years: int = 2) -> IO[pd.DataFrame]:
    """Effect: Network Call to Yahoo Finance."""
    def _fetch():
        end_date = pd.Timestamp.today().normalize()
        start_date = end_date - pd.DateOffset(years=years)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Cleanup logic (part of the fetch IO boundary)
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

def run_monte_carlo_io(params: BrownianParams, days: int = 30, scenarios: int = 1000) -> IO[pd.DataFrame]:
    """Effect: Random Number Generation & Simulation."""
    def _sim():
        mu, sigma, S0 = params['mu'], params['sigma'], params['last_price']
        dt = 1
        returns = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=(days, scenarios))
        price_paths = np.vstack([np.full((1, scenarios), S0), S0 * np.exp(np.cumsum(returns, axis=0))])
        return pd.DataFrame(price_paths)
    return IO(_sim)

def valyu_search_io(query: str) -> IO[str]:
    """Effect: External API Search."""
    def _search():
        client = Valyu(api_key=os.environ.get("VALYU_API_KEY"))
        return str(client.answer(query))
    return IO(_search)

def prophet_predict_io(df: pd.DataFrame, days: int = 30) -> IO[pd.DataFrame]:
    """Effect: Heavy Computation / Model Training."""
    def _train_and_predict():
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        return forecast
    return IO(_train_and_predict)