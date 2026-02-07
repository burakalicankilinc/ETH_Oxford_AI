from dataclasses import dataclass
from typing import Any, Dict, Generator
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime
from prophet import Prophet 
from valyu import Valyu

# --- 1. THE EFFECT ALGEBRA (Expanded) ---
class Effect: pass

@dataclass
class GetMarketData(Effect):
    ticker: str
    years: int = 2

@dataclass
class RunProphet(Effect):
    history_df: pd.DataFrame
    days: int = 30

@dataclass
class ValyuSearch(Effect):
    query: str

@dataclass
class SaveToAuditLog(Effect):
    ticker: str
    prediction_data: Any
    model_name: str

@dataclass
class LogInfo(Effect):
    message: str

# --- 2. THE RUNTIME (The "Doer" for Everything) ---
def execute_effect_program(program: Generator) -> Any:
    """Interprets ALL tools (Math, ML, and Search)."""
    try:
        current_effect = next(program)
        
        while True:
            result = None
            
            # --- INTERPRETER SWITCH ---
            
            # 1. MARKET DATA EFFECT
            if isinstance(current_effect, GetMarketData):
                end = pd.Timestamp.today().normalize()
                start = end - pd.DateOffset(years=current_effect.years)
                df = yf.download(current_effect.ticker, start=start, end=end, progress=False)
                # Cleanup logic (same as before)
                if isinstance(df.columns, pd.MultiIndex): df = df['Close']
                elif 'Close' in df.columns: df = df['Close']
                if isinstance(df, pd.DataFrame) and current_effect.ticker in df.columns:
                    df = df[current_effect.ticker]
                result = df

            # 2. PROPHET ML EFFECT
            elif isinstance(current_effect, RunProphet):
                # Prepare Data
                df = current_effect.history_df.reset_index()
                df.columns = ['ds', 'y']
                df['ds'] = df['ds'].dt.tz_localize(None)
                
                # Run Model (Side Effect: Uses CPU/Memory heavily)
                m = Prophet(daily_seasonality=True)
                m.fit(df)
                future = m.make_future_dataframe(periods=current_effect.days)
                forecast = m.predict(future)
                result = forecast

            # 3. SEARCH EFFECT
            elif isinstance(current_effect, ValyuSearch):
                # Side Effect: Network Call
                client = Valyu(api_key=os.environ.get("VALYU_API_KEY"))
                response = client.answer(current_effect.query)
                # Handle response formats
                if hasattr(response, 'contents'): result = str(response.contents)
                elif isinstance(response, dict) and 'contents' in response: result = str(response['contents'])
                else: result = str(response)

            # 4. PERSISTENCE EFFECT
            elif isinstance(current_effect, SaveToAuditLog):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "ticker": current_effect.ticker,
                    "model": current_effect.model_name,
                    "data_snippet": str(current_effect.prediction_data)[:200] # Truncate for log
                }
                with open("prediction_audit_log.json", "a") as f:
                    f.write(json.dumps(entry) + "\n")
                result = True

            # 5. LOGGING
            elif isinstance(current_effect, LogInfo):
                print(f"[Runtime]: {current_effect.message}")
                result = None

            # --- RESUME LOGIC ---
            current_effect = program.send(result)
            
    except StopIteration as e:
        return e.value
    except Exception as e:
        return f"Runtime Error: {str(e)}"
