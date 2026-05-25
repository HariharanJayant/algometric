import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import yfinance as yf
import pandas as pd
import numpy as np
import ta

app = Flask(__name__)

# Initialize the Gemini Client
api_key = os.environ.get("GEMINI_API_KEY", "YOUR_LOCAL_API_KEY_FALLBACK")
client = genai.Client(api_key=api_key)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

# === PIPELINE 1: LIVE SCANNING & SCREENSHOT ANALYSIS ===
@app.route('/analyze', methods=['POST'])
def analyze():
    ticker_symbol = request.form.get('ticker', '').upper().strip()
    if not ticker_symbol:
        return jsonify({'error': 'Target Ticker input field required.'})

    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="60d", interval="1d")

        if df.empty:
            return jsonify({'error': f"Ticker target '{ticker_symbol}' returned an empty dataset frame."})

        current_price = round(df['Close'].iloc[-1], 2)

        atr_series = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        atr_val = round(atr_series.iloc[-1], 2) if not pd.isna(atr_series.iloc[-1]) else 0.0

        last_high, last_low, last_close = df['High'].iloc[-2], df['Low'].iloc[-2], df['Close'].iloc[-2]
        pivot = round((last_high + last_low + last_close) / 3, 2)
        r1 = round((2 * pivot) - last_low, 2)
        s1 = round((2 * pivot) - last_high, 2)

        fast_ma_series = ta.trend.sma_indicator(df['Close'], window=9)
        slow_ma_series = ta.trend.sma_indicator(df['Close'], window=21)
        fast_ma = round(fast_ma_series.iloc[-1], 2) if not pd.isna(fast_ma_series.iloc[-1]) else 0.0
        slow_ma = round(slow_ma_series.iloc[-1], 2) if not pd.isna(slow_ma_series.iloc[-1]) else 0.0

        signal = "BULLISH" if fast_ma > slow_ma else "BEARISH" if fast_ma < slow_ma else "NEUTRAL"
        rsi_series = ta.momentum.rsi(df['Close'], window=14)
        rsi_val = round(rsi_series.iloc[-1], 2) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        macd_series = ta.trend.macd(df['Close'])
        macd_val = round(macd_series.iloc[-1], 2) if not pd.isna(macd_series.iloc[-1]) else 0.0
        bb_upper = round(ta.volatility.bollinger_hband(df['Close']).iloc[-1], 2)
        bb_lower = round(ta.volatility.bollinger_lband(df['Close']).iloc[-1], 2)

        uploaded_file = request.files.get('chart_image')
        ai_prompt = f"Perform a technical review for {ticker_symbol}. Close: ${current_price}, ATR: {atr_val}, RSI: {rsi_val}, Signal: {signal}."
        contents_payload = [ai_prompt]

        if uploaded_file and uploaded_file.filename != '':
            image_bytes = uploaded_file.read()
            mime_type = uploaded_file.content_type or 'image/png'
            contents_payload.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        response = client.models.generate_content(model='gemini-2.5-flash', contents=contents_payload)
        
        return jsonify({
            'current_price': current_price, 'buy_limit': current_price, 'stop_loss': round(current_price - (2*atr_val), 2), 'take_profit': round(current_price + (3*atr_val), 2),
            'atr': atr_val, 'pivot_point': pivot, 'support_1': s1, 'resistance_1': r1,
            'fast_ma_val': fast_ma, 'slow_ma_val': slow_ma, 'strategy_signal': signal,
            'rsi_val': rsi_val, 'macd_val': macd_val, 'bb_upper_val': bb_upper, 'bb_lower_val': bb_lower,
            'analysis': response.text.strip()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# === PIPELINE 2: MATCHED HISTORICAL SIMULATION ENGINE ===
@app.route('/simulate', methods=['POST'])
def simulate():
    ticker_symbol = request.form.get('ticker', 'MSFT').upper().strip()
    fast_length = int(request.form.get('fast_length', 9))
    slow_length = int(request.form.get('slow_length', 21))
    data_window = request.form.get('data_window', '180') # coming in as string number of days

    # Convert window string to yfinance period style safely
    period_map = {"180": "6mo", "365": "1y", "90": "3mo"}
    yf_period = period_map.get(data_window, "6mo")

    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=yf_period, interval="1d")

        if df.empty:
            return jsonify({'error': "No historical data found for target ticker."})

        # Calculations matching your model indicators
        df['Fast_MA'] = ta.trend.sma_indicator(df['Close'], window=fast_length)
        df['Slow_MA'] = ta.trend.sma_indicator(df['Close'], window=slow_length)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df = df.dropna()

        initial_capital = 1000.0  # Matches your UI parameter ($1,000.00 Fixed)
        capital = initial_capital
        position = 0
        total_signals = 0
        signals_feed = []

        # Calculate Buy and Hold Return
        bh_shares = initial_capital / df['Close'].iloc[0]
        bh_final_value = bh_shares * df['Close'].iloc[-1]
        bh_return_pct = round(((bh_final_value - initial_capital) / initial_capital) * 100, 2)

        for i in range(1, len(df)):
            prev_fast, prev_slow = df['Fast_MA'].iloc[i-1], df['Slow_MA'].iloc[i-1]
            curr_fast, curr_slow = df['Fast_MA'].iloc[i], df['Slow_MA'].iloc[i]
            price = round(df['Close'].iloc[i], 2)
            date_str = df.index[i].strftime('%Y-%m-%d')
            rsi_val = round(df['RSI'].iloc[i], 2) if not pd.isna(df['RSI'].iloc[i]) else 50.0
            atr_val = round(df['ATR'].iloc[i], 2) if not pd.isna(df['ATR'].iloc[i]) else 0.0

            # Bullish Crossover (BUY Trigger)
            if prev_fast <= prev_slow and curr_fast > curr_slow and position == 0:
                position = capital / price
                capital = 0
                total_signals += 1
                signals_feed.append({
                    'date': date_str, 'action': 'BUY', 'close': f"${price}", 'rsi': rsi_val, 'atr': atr_val
                })

            # Bearish Crossover (SELL Trigger)
            elif prev_fast >= prev_slow and curr_fast < curr_slow and position > 0:
                capital = position * price
                position = 0
                total_signals += 1
                signals_feed.append({
                    'date': date_str, 'action': 'SELL', 'close': f"${price}", 'rsi': rsi_val, 'atr': atr_val
                })

        if position > 0:
            capital = position * df['Close'].iloc[-1]

        strategy_return_pct = round(((capital - initial_capital) / initial_capital) * 100, 2)

        return jsonify({
            'success': True,
            'strategy_return': f"{strategy_return_pct}%",
            'buy_hold_return': f"{bh_return_pct}%",
            'total_signals': total_signals,
            'signals': signals_feed
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
