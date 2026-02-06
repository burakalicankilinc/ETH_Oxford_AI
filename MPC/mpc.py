import numpy as np
from typing import Dict, List, Any
from gbm import simulate_gbm_mc
from predictor import SimpleGBMPredictor

class MPCController:
    def __init__(self, predictor: Any, action_grid: List[float], horizon: float = 1/252, transaction_cost: float = 0.001):
        self.predictor = predictor
        self.action_grid = action_grid
        self.dt = horizon
        self.transaction_cost = transaction_cost

    def select_action(self, current_price: float, current_position: float, history: np.ndarray, 
                      mu_override: float = None) -> Dict:
        
        # Logic: If the Agent provides a mu, use it. Otherwise, estimate from history.
        mu_hist, sigma = self.predictor.estimate_mu_sigma(history, self.dt)
        mu = mu_override if mu_override is not None else mu_hist

        best_action = 0.0
        best_score = -float('inf')
        
        for action in self.action_grid:
            paths = simulate_gbm_mc(current_price, mu, sigma, self.dt, n_paths=1000)
            final_prices = paths[-1, :]
            
            # Objective: Expected Portfolio Value - Transaction Costs
            expected_val = np.mean(final_prices) * (current_position + action)
            costs = abs(action) * current_price * self.transaction_cost
            score = expected_val - costs
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return {
            "action": best_action,
            "expected_final_price": np.mean(final_prices),
            "mu": mu,
            "sigma": sigma
        }