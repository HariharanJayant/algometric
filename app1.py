import os
from flask import Flask, render_template, request, jsonify
from google import genai
import yfinance as yf
import pandas as pd
import numpy as np
import ta

app = Flask(__name__)

# Initialize the Gemini Client
# Note: Render automatically picks up your GEMINI_API_KEY if you set it in their dashboard environment variables
api_key = os.environ.get("GEMINI_API_KEY", "YOUR_LOCAL_API_KEY_FALLBACK")
client = genai.Client(api_key=api_key)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Grab the dynamic ticker symbol from the frontend form
    ticker_symbol = request.form.get('ticker', 'NVDA').upper().strip()
    if not ticker_symbol:
        ticker_symbol = 'NVDA'

    print(f"Initializing Quantitative Scan Matrix Sequence for Target: {ticker_symbol}")

    try:
        # 2. Fetch live stock data framework from yfinance
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="60d", interval="1d")

        if df.empty:
            return jsonify({'error': f"Ticker target symbol '{ticker_symbol}' returned an empty dataset frame. Verify exchange availability."})

        # Calculate current last price
        current_price = round(df['Close'].iloc[-1], 2)

        # 3. Calculate Technical Indicators using 'ta' library
        # Average True Range (ATR)
        atr_series = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        atr_val = round(atr_series.iloc[-1], 2) if not pd.isna(atr_series.iloc[-1]) else 0.0

        # Pivot Points (Classic Daily)
        last_high = df['High'].iloc[-2]
        last_low = df['Low'].iloc[-2]
        last_close = df['Close'].iloc[-2]
        
        pivot = round((last_high + last_low + last_close) / 3, 2)
        r1 = round((2 * pivot) - last_low, 2)
        s1 = round((2 * pivot) - last_high, 2)

        # Moving Averages (Fast 9 vs Slow 21)
        fast_ma_series = ta.trend.sma_indicator(df['Close'], window=9)
        slow_ma_series = ta.trend.sma_indicator(df['Close'], window=21)
        
        fast_ma = round(fast_ma_series.iloc[-1], 2) if not pd.isna(fast_ma_series.iloc[-1]) else 0.0
        slow_ma = round(slow_ma_series.iloc[-1], 2) if not pd.isna(slow_ma_series.iloc[-1]) else 0.0

        # Strategy Signal Logic
        if fast_ma > slow_ma:
            signal = "BULLISH CROSSOVER (BUY)"
        elif fast_ma < slow_ma:
            signal = "BEARISH CROSSOVER (SELL)"
        else:
            signal = "NEUTRAL"

        # RSI (14)
        rsi_series = ta.momentum.rsi(df['Close'], window=14)
        rsi_val = round(rsi_series.iloc[-1], 2) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        # MACD
        macd_series = ta.trend.macd(df['Close'])
        macd_val = round(macd_series.iloc[-1], 2) if not pd.isna(macd_series.iloc[-1]) else 0.0

        # Bollinger Bands
        bb_upper_series = ta.volatility.bollinger_hband(df['Close'])
        bb_lower_series = ta.volatility.bollinger_lband(df['Close'])
        bb_upper = round(bb_upper_series.iloc[-1], 2) if not pd.isna(bb_upper_series.iloc[-1]) else 0.0
        bb_lower = round(bb_lower_series.iloc[-1], 2) if not pd.isna(bb_lower_series.iloc[-1]) else 0.0

        # Simple automated target generation lines based on Pivot rules
        buy_limit = round(current_price, 2)
        stop_loss = round(current_price - (2 * atr_val), 2)
        take_profit = round(current_price + (3 * atr_val), 2)

        # 4. Generate AI Prompt Dynamic Context Matrix
        ai_prompt = f"""
        Perform a professional quantitative technical analysis review for asset ticker symbol: {ticker_symbol}.
        Current Market Metrics:
        - Last Traded Close Price: ${current_price}
        - 14-Day ATR Volatility: {atr_val}
        - Pivot Point: ${pivot} (S1 Support: ${s1} / R1 Resistance: ${r1})
        - Moving Averages: Fast MA(9) at ${fast_ma} vs Slow MA(21) at ${slow_ma} -> Current State: {signal}
        - Momentum: RSI(14) is at {rsi_val} and MACD is at {macd_val}
        - Volatility Envelope: Bollinger Upper: ${bb_upper} / Lower: ${bb_lower}

        Provide a sharp, 3-sentence institutional market execution summary covering current structural direction and immediate risk zones.
        """

        # Call Gemini text engine safely
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=ai_prompt,
        )
        ai_response_text = response.text.strip()

        # 5. Return JSON payload back to front-end dashboard scripts
        return jsonify({
            'current_price': current_price,
            'buy_limit': buy_limit,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr_val,
            'pivot_point': pivot,
            'support_1': s1,
            'resistance_1': r1,
            'fast_ma_val': fast_ma,
            'slow_ma_val': slow_ma,
            'strategy_signal': signal,
            'rsi_val': rsi_val,
            'macd_val': macd_val,
            'bb_upper_val': bb_upper,
            'bb_lower_val': bb_lower,
            'analysis': ai_response_text
        })

    except Exception as e:
        print(f"Execution Exception Crash: {str(e)}")
        return jsonify({'error': f"System pipeline failure: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
