from dataclasses import dataclass
from typing import Any, Dict, Generator
import pandas as pd
import yfinance as yf
import json
from datetime import datetime

# --- THE EFFECT ALGEBRA (Data) ---
class Effect: pass

@dataclass
class GetMarketData(Effect):
    ticker: str
    years: int = 2

@dataclass
class SaveToAuditLog(Effect):
    ticker: str
    prediction_data: Dict[str, Any]
    model_name: str

@dataclass
class LogInfo(Effect):
    message: str

# --- THE RUNTIME (Execution) ---
def execute_effect_program(program: Generator) -> Any:
    """Interprets the Pure Plan and executes Side Effects."""
    try:
        current_effect = next(program)
        
        while True:
            result = None
            
            # ROUTER: Matches Data -> Action
            if isinstance(current_effect, GetMarketData):
                # EFFECT: Download Data
                end = pd.Timestamp.today().normalize()
                start = end - pd.DateOffset(years=current_effect.years)
                df = yf.download(current_effect.ticker, start=start, end=end, progress=False)
                
                # Cleanup (MultiIndex handling)
                if isinstance(df.columns, pd.MultiIndex): df = df['Close']
                elif 'Close' in df.columns: df = df['Close']
                if isinstance(df, pd.DataFrame) and current_effect.ticker in df.columns:
                    df = df[current_effect.ticker]
                
                result = df if not df.empty else None

            elif isinstance(current_effect, SaveToAuditLog):
                # EFFECT: Write to Disk
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "ticker": current_effect.ticker,
                    "model": current_effect.model_name,
                    "data": current_effect.prediction_data
                }
                with open("prediction_audit_log.json", "a") as f:
                    f.write(json.dumps(entry) + "\n")
                result = True

            elif isinstance(current_effect, LogInfo):
                # EFFECT: Console Log
                print(f"[Runtime]: {current_effect.message}")
                result = None

            # Feed result back to Logic
            current_effect = program.send(result)
            
    except StopIteration as e:
        return e.value
    except Exception as e:
        return f"Runtime Error: {str(e)}"