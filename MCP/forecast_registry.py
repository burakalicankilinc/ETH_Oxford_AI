import json
import os
import yfinance as yf
from datetime import datetime, timedelta

REGISTRY_FILE = "prediction_ledger.json"

class ForecastRegistry:
    def __init__(self):
        self._load_state()

    def _load_state(self):
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, 'r') as f:
                self.state = json.load(f)
        else:
            # State: List of active predictions and a running accuracy score
            self.state = {
                "accuracy_score": 100.0,  # Starts at 100% or 0
                "total_predictions": 0,
                "active_forecasts": [],  # Predictions waiting for verification
                "history": []            # Completed/Verified predictions
            }
            self._save_state()

    def _save_state(self):
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def log_prediction(self, ticker: str, predicted_price: float, horizon_days: int, model_confidence: float):
        """Effect: Commits a prediction to the ledger."""
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "target_date": (datetime.now() + timedelta(days=horizon_days)).strftime("%Y-%m-%d"),
            "ticker": ticker,
            "predicted_price": predicted_price,
            "confidence": model_confidence
        }
        self.state["active_forecasts"].append(entry)
        self.state["total_predictions"] += 1
        self._save_state()
        return f"Prediction logged. Target: {entry['target_date']}"

    def verify_forecasts(self):
        """
        Effect: Checks active forecasts against real market data.
        Updates the 'Accuracy Score' state.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        verified_count = 0
        
        # Iterate over a copy to safely modify the list
        for forecast in self.state["active_forecasts"][:]:
            if forecast["target_date"] <= today:
                # 1. Fetch Real Outcome
                df = yf.download(forecast["ticker"], period="1d", progress=False)
                if df.empty: continue
                
                real_price = df['Close'].values[-1]
                
                # 2. Calculate Error
                error = abs(real_price - forecast["predicted_price"])
                percent_error = (error / real_price) * 100
                
                # 3. Update State (Move to history, update score)
                forecast["real_price"] = float(real_price)
                forecast["error_pct"] = float(percent_error)
                
                self.state["history"].append(forecast)
                self.state["active_forecasts"].remove(forecast)
                
                # Simple moving average of accuracy (simplified)
                current_score = self.state["accuracy_score"]
                # Penalize score by error magnitude
                new_score = (current_score * 0.9) + ((100 - percent_error) * 0.1)
                self.state["accuracy_score"] = new_score
                
                verified_count += 1
        
        self._save_state()
        return {
            "verified": verified_count, 
            "current_accuracy": self.state["accuracy_score"]
        }