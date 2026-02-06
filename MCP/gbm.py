import numpy as np

def simulate_gbm_mc(s0, mu, sigma, dt, n_paths=1000):
    """Simulates paths for Geometric Brownian Motion."""
    # Standard GBM Formula: S(t) = S(0)*exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    z = np.random.standard_normal(n_paths)
    st = s0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return st.reshape(1, -1) # Returns final state paths

def estimate_mu_sigma_from_prices(prices, dt):
    """Calculates annualised drift and volatility from price history."""
    log_returns = np.diff(np.log(prices))
    mu = np.mean(log_returns) / dt
    sigma = np.std(log_returns) / np.sqrt(dt)
    return mu, sigma

