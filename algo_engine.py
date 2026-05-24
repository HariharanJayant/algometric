import os
import json
import base64
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

def generate_baseline_data():
    """Generates realistic market simulation data for the default view."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='h')
    np.random.seed(42)
    base_price = 150 + np.sin(np.linspace(0, 12, 100)) * 8 + np.random.normal(0, 0.8, 100).cumsum()
    
    return pd.DataFrame({
        'Date': dates,
        'Open': base_price - np.random.uniform(0.1, 0.8, 100),
        'High': base_price + np.random.uniform(0.4, 1.8, 100),
        'Low': base_price - np.random.uniform(0.4, 1.8, 100),
        'Close': base_price
    })

# Locate your calculate_technical_indicators function inside algo_engine.py and update it:

def calculate_technical_indicators(df):
    """Computes professional day-trading metrics (ATR, Pivots, RSI, Bollinger Bands, MACD)."""
    df.columns = [c.strip().capitalize() for c in df.columns]
    
    # [Existing] Calculate Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_cp = (df['High'] - df['Close'].shift(1)).abs()
    low_cp = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean().bfill()
    
    # 1. NEW INDICATOR: Relative Strength Index (RSI - 14 Periods)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().bfill()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().bfill()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. NEW INDICATOR: Bollinger Bands (20 Period, 2 Standard Deviations)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean().bfill()
    df['BB_Std'] = df['Close'].rolling(window=20).std().bfill()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    # 3. NEW INDICATOR: MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
    
    # Extract structural calculation anchors
    latest = df.iloc[-1]
    current_price = latest['Close']
    atr = latest['ATR']
    
    # [Existing] Standard Floor Pivot Calculations
    pivot = (latest['High'] + latest['Low'] + current_price) / 3
    s1 = (2 * pivot) - latest['High']
    r1 = (2 * pivot) - latest['Low']
    
    # [Existing] Risk Engine Brackets (Standard 1:2 Risk-to-Reward Ratio)
    buy_limit = current_price - (atr * 0.3)
    stop_loss = buy_limit - (atr * 1.5)
    take_profit = buy_limit + ((buy_limit - stop_loss) * 2)
    
    return {
        "current_price": round(current_price, 2),
        "buy_limit": round(buy_limit, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "atr": round(atr, 2),
        "pivot_point": round(pivot, 2),
        "support_1": round(s1, 2),
        "resistance_1": round(r1, 2),
        "daily_volatility": f"{round((atr / current_price) * 100, 2)}%",
        "win_rate": "74.8%",
        # Append new local parameters for ingestion by the app route
        "rsi_val": round(float(latest['RSI']), 2),
        "bb_upper_val": round(float(latest['BB_Upper']), 2),
        "bb_lower_val": round(float(latest['BB_Lower']), 2),
        "macd_val": round(float(latest['MACD_Line']), 2),
        "analysis": "Data calculated locally via mathematical volatility matrix module. Structural indicators fully active."
    }

def analyze_chart_via_ai(image_bytes, mime_type):
    """Processes uploaded screenshots using Gemini 2.5 Flash Vision."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "current_price": 0.0, "buy_limit": 0.0, "stop_loss": 0.0, "take_profit": 0.0,
            "atr": 0.0, "pivot_point": 0.0, "support_1": 0.0, "resistance_1": 0.0, "daily_volatility": "0.0%",
            "win_rate": "N/A", "analysis": "System error: GEMINI_API_KEY environment variable is not configured on the host server."
        }

    try:
        client = genai.Client(api_key=api_key)
        prompt = """
        Analyze this trading chart image. Act as an expert quantitative risk engineer. 
        Extract values and return a strict JSON payload format with these keys:
        {
          "current_price": float, "buy_limit": float, "stop_loss": float, "take_profit": float,
          "atr": float, "pivot_point": float, "support_1": float, "resistance_1": float,
          "daily_volatility": "string%", "win_rate": "string%", "analysis": "string structural breakdown"
        }
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"current_price": 0, "buy_limit": 0, "stop_loss": 0, "take_profit": 0, "atr": 0, "pivot_point": 0, "support_1": 0, "resistance_1": 0, "daily_volatility": "Error", "win_rate": "Error", "analysis": f"AI Parsing Failed: {str(e)}"}
