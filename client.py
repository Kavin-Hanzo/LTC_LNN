import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
URL = "http://127.0.0.1:8000/predict"
TICKER = "META"
LOOKBACK = 30  
# ---------------------

def get_market_data(ticker: str, lookback: int):
    """
    Fetches raw data and constructs the 6-feature matrix:
    [Open, High, Low, Close, Volume, Close_Return]
    """
    data = yf.download(ticker, period="60d", interval="1d")
    
    if data.empty:
        raise ValueError(f"No data found for {ticker}")

    df = data.copy()
    
    # Calculate the 6th feature
    df["Close_Return"] = df["Close"].pct_change().clip(-0.30, 0.30)
    
    # Exact 6 features expected by your .pkl scaler
    features = ["Open", "High", "Low", "Close", "Volume", "Close_Return"]
    df_final = df[features].dropna()
    
    # Slice exact lookback
    df_ready = df_final.tail(lookback)
    
    if len(df_ready) < lookback:
        raise ValueError(f"Insufficient data. Need {lookback}, got {len(df_ready)}")
        
    return df_ready

try:
    print(f"Fetching data for {TICKER} from Yahoo Finance...")
    df_inference = get_market_data(TICKER, LOOKBACK)
    
    history_list = df_inference.values.tolist()
    
    payload = {
        "ticker": TICKER,
        "data": history_list
    }

    print(f"Sending request to server (Lookback: {len(history_list)}, Features: {len(history_list[0])})...")
    response = requests.post(URL, json=payload)
    response.raise_for_status()
    
    result = response.json()
    forecast = result["forecast_next_5_days"]
    using_id = result.get("using_identity", False)
    scaler_source = result.get("scaler_source", "Unknown")
    
    print(f"\n--- Prediction Successful ---")
    print(f"Identity Injected: {using_id}")
    print(f"Scaler Used: {scaler_source}")
    # UPDATED: We now proudly print that these are real prices!
    print(f"Next 5 Days (Real Prices): {forecast}")

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    
    history_close = df_inference["Close"].values
    plt.plot(range(LOOKBACK), history_close, label="Historical Close", marker='o', color='blue')
    
    # UPDATED: The forecast line will now connect naturally to the historical prices
    x_forecast = range(LOOKBACK, LOOKBACK + 5)
    plt.plot(x_forecast, forecast, label="5-Day Forecast (Real Prices)", marker='D', linestyle='--', color='red')
    
    plt.title(f"{TICKER} - LNN 5-Day Forecast (Lookback: {LOOKBACK})")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)") # UPDATED label
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Server detail: {e.response.text}")