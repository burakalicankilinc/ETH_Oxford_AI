import numpy as np
from gbm import estimate_mu_sigma_from_prices

class SimpleGBMPredictor:
    def estimate_mu_sigma(self, history, dt):
        return estimate_mu_sigma_from_prices(history, dt)