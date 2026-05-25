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

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker_symbol = request.form.get('ticker', 'NVDA').upper().strip()
    if not ticker_symbol:
        ticker_symbol = 'NVDA'

    print(f"Initializing Multimodal Quantitative Ingestion Sequence for Target: {ticker_symbol}")

    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="60d", interval="1d")

        if df.empty:
            return jsonify({'error': f"Ticker target symbol '{ticker_symbol}' returned an empty dataset frame. Verify exchange availability."})

        current_price = round(df['Close'].iloc[-1], 2)

        # Technical Indicators calculation
        atr_series = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        atr_val = round(atr_series.iloc[-1], 2) if not pd.isna(atr_series.iloc[-1]) else 0.0

        last_high = df['High'].iloc[-2]
        last_low = df['Low'].iloc[-2]
        last_close = df['Close'].iloc[-2]
        
        pivot = round((last_high + last_low + last_close) / 3, 2)
        r1 = round((2 * pivot) - last_low, 2)
        s1 = round((2 * pivot) - last_high, 2)

        fast_ma_series = ta.trend.sma_indicator(df['Close'], window=9)
        slow_ma_series = ta.trend.sma_indicator(df['Close'], window=21)
        
        fast_ma = round(fast_ma_series.iloc[-1], 2) if not pd.isna(fast_ma_series.iloc[-1]) else 0.0
        slow_ma = round(slow_ma_series.iloc[-1], 2) if not pd.isna(slow_ma_series.iloc[-1]) else 0.0

        if fast_ma > slow_ma:
            signal = "BULLISH CROSSOVER (BUY)"
        elif fast_ma < slow_ma:
            signal = "BEARISH CROSSOVER (SELL)"
        else:
            signal = "NEUTRAL"

        rsi_series = ta.momentum.rsi(df['Close'], window=14)
        rsi_val = round(rsi_series.iloc[-1], 2) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        macd_series = ta.trend.macd(df['Close'])
        macd_val = round(macd_series.iloc[-1], 2) if not pd.isna(macd_series.iloc[-1]) else 0.0

        bb_upper_series = ta.volatility.bollinger_hband(df['Close'])
        bb_lower_series = ta.volatility.bollinger_lband(df['Close'])
        bb_upper = round(bb_upper_series.iloc[-1], 2) if not pd.isna(bb_upper_series.iloc[-1]) else 0.0
        bb_lower = round(bb_lower_series.iloc[-1], 2) if not pd.isna(bb_lower_series.iloc[-1]) else 0.0

        buy_limit = round(current_price, 2)
        stop_loss = round(current_price - (2 * atr_val), 2)
        take_profit = round(current_price + (3 * atr_val), 2)

        uploaded_file = request.files.get('chart_image')
        
        ai_prompt = f"""
        Perform a professional quantitative technical analysis review for asset ticker symbol: {ticker_symbol}.
        Current Market Metrics:
        - Last Traded Close Price: ${current_price}
        - 14-Day ATR Volatility: {atr_val}
        - Pivot Point: ${pivot} (S1 Support: ${s1} / R1 Resistance: ${r1})
        - Moving Averages: Fast MA(9) at ${fast_ma} vs Slow MA(21) at ${slow_ma} -> Current State: {signal}
        - Momentum: RSI(14) is at {rsi_val} and MACD is at {macd_val}
        - Volatility Envelope: Bollinger Upper: ${bb_upper} / Lower: ${bb_lower}

        If an image chart is provided alongside this data, analyze its visual structure (trendlines, candlestick shapes, chart patterns) to confirm or reject these numerical metrics. 
        Provide a sharp, 3-sentence institutional market execution summary covering current structural direction and immediate risk zones.
        """
        
        contents_payload = [ai_prompt]

        if uploaded_file and uploaded_file.filename != '':
            image_bytes = uploaded_file.read()
            mime_type = uploaded_file.content_type or 'image/png'
            
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
            contents_payload.append(image_part)
            print("Screenshot detected and appended to the model payload context matrix.")

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents_payload,
        )
        ai_response_text = response.text.strip()

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


# === STRATEGY SIMULATION ROUTE FOR BACKTESTING ===
@app.route('/simulate', methods=['POST'])
def simulate():
    ticker_symbol = request.form.get('ticker', 'NVDA').upper().strip()
    if not ticker_symbol:
        ticker_symbol = 'NVDA'

    try:
        stock = yf.Ticker(ticker_symbol)
        # Pulling 1 year of historical daily track data
        df = stock.history(period="1y", interval="1d")

        if df.empty:
            return jsonify({'error': f"No historical trace records found for {ticker_symbol}."})

        # Process simple moving average crossovers vectors
        df['Fast_MA'] = ta.trend.sma_indicator(df['Close'], window=9)
        df['Slow_MA'] = ta.trend.sma_indicator(df['Close'], window=21)
        df = df.dropna()

        initial_capital = 10000.0
        capital = initial_capital
        position = 0
        total_trades = 0
        winning_trades = 0
        entry_price = 0.0

        for i in range(1, len(df)):
            prev_fast, prev_slow = df['Fast_MA'].iloc[i-1], df['Slow_MA'].iloc[i-1]
            curr_fast, curr_slow = df['Fast_MA'].iloc[i], df['Slow_MA'].iloc[i]
            price = round(df['Close'].iloc[i], 2)

            # Golden Cross: Buy Signal
            if prev_fast <= prev_slow and curr_fast > curr_slow and position == 0:
                position = capital / price
                entry_price = price
                capital = 0
                total_trades += 1
            
            # Death Cross: Sell Signal
            elif prev_fast >= prev_slow and curr_fast < curr_slow and position > 0:
                capital = position * price
                position = 0
                if price > entry_price:
                    winning_trades += 1

        # Check closing position valuation if asset is still held
        if position > 0:
            capital = position * df['Close'].iloc[-1]
            if df['Close'].iloc[-1] > entry_price:
                winning_trades += 1

        net_profit_pct = round(((capital - initial_capital) / initial_capital) * 100, 2)
        win_rate_pct = round((winning_trades / total_trades) * 100, 2) if total_trades > 0 else 0.0

        return jsonify({
            'success': True,
            'total_trades': total_trades,
            'win_rate': f"{win_rate_pct}%",
            'net_profit_pct': f"{net_profit_pct}%",
            'final_balance': f"{round(capital, 2):,}"
        })
    except Exception as e:
        return jsonify({'error': f"Sim-Engine Crash: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
