import os
import json
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import yfinance as yf

app = Flask(__name__)

def analyze_chart_via_ai(image_bytes, mime_type):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY environment variable is not configured."}
    try:
        client = genai.Client(api_key=api_key)
        prompt = """
        Analyze this trading chart image as an expert quantitative risk engineer. 
        Extract key technical levels and return a strict JSON payload format with these keys:
        {
          "current_price": float, "buy_limit": float, "stop_loss": float, "take_profit": float,
          "atr": float, "pivot_point": float, "support_1": float, "resistance_1": float,
          "daily_volatility": "string%", "win_rate": "string%", 
          "rsi_val": float, "bb_upper_val": float, "bb_lower_val": float, "macd_val": float,
          "analysis": "string structural breakdown"
        }
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"AI Parsing Failed: {str(e)}"}

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/portfolio')
def portfolio_page():
    return render_template('portfolio.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker', 'NVDA').upper()
    file = request.files.get('chart_image')
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="60d")
        if not hist.empty:
            close_p = float(hist['Close'].iloc[-1])
            high_p = float(hist['High'].iloc[-1])
            low_p = float(hist['Low'].iloc[-1])
            calc_atr = abs(high_p - low_p)
            
            hist['FastMA'] = hist['Close'].rolling(window=9).mean()
            hist['SlowMA'] = hist['Close'].rolling(window=21).mean()
            current_fast = round(float(hist['FastMA'].iloc[-1]), 2)
            current_slow = round(float(hist['SlowMA'].iloc[-1]), 2)
            
            prev_fast = float(hist['FastMA'].iloc[-2])
            prev_slow = float(hist['SlowMA'].iloc[-2])
            strategy_signal = "HOLD / NEUTRAL"
            if prev_fast <= prev_slow and current_fast > current_slow:
                strategy_signal = "🟩 DUAL MA CROSSOVER: STRONG BUY ALERT"
            elif prev_fast >= prev_slow and current_fast < current_slow:
                strategy_signal = "🟥 DUAL MA CROSSUNDER: STRONG SELL ALERT"
            
            paragraph_analysis = (
                f"LOCAL ENGINE STATUS: Mathematical processing complete for symbol {ticker}. "
                f"The 9-period Fast Moving Average is holding at ${current_fast:.2f} relative to the 21-period Slow Moving Average at ${current_slow:.2f}, "
                f"confirming a state of [{strategy_signal}]. To achieve secondary structural verification via computer vision, "
                f"take a screenshot of your active TradingView stream below, upload the image file into the 'Screenshot Pipeline Ingestion' panel on the left, "
                f"and re-initialize the matrix scan. This allows the multimodal AI agent to cross-examine your local indicators against live trendline "
                f"support vectors, liquidity voids, and chart patterns for comprehensive risk evaluation."
            )
            fallback_data = {
                "current_price": round(close_p, 2), "buy_limit": round(low_p * 1.002, 2),
                "stop_loss": round(close_p - (calc_atr * 1.5), 2), "take_profit": round(close_p + (calc_atr * 2), 2),
                "atr": round(calc_atr, 2), "pivot_point": round((close_p + high_p + low_p)/3, 2),
                "support_1": round((2 * ((close_p + high_p + low_p)/3)) - high_p, 2),
                "resistance_1": round((2 * ((close_p + high_p + low_p)/3)) - low_p, 2),
                "daily_volatility": f"{round((calc_atr/close_p)*100, 2)}%", "win_rate": "54%",
                "fast_ma_val": current_fast, "slow_ma_val": current_slow,
                "strategy_signal": strategy_signal, "analysis": paragraph_analysis
            }
        else:
            fallback_data = {"error": "No ticker data found."}
    except Exception as e:
        fallback_data = {"error": f"Backend indicator compiler failure: {str(e)}"}

    if file and file.filename != '':
        image_bytes = file.read()
        mime_type = file.mimetype
        ai_result = analyze_chart_via_ai(image_bytes, mime_type)
        if "error" not in ai_result:
            ai_result["fast_ma_val"] = fallback_data.get("fast_ma_val")
            ai_result["slow_ma_val"] = fallback_data.get("slow_ma_val")
            ai_result["strategy_signal"] = fallback_data.get("strategy_signal")
            return jsonify(ai_result)
    return jsonify(fallback_data)

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    data = request.json
    ticker = data.get('ticker', 'NVDA').upper()
    fast_len = data.get('fast', 9)
    slow_len = data.get('slow', 21)
    days_window = data.get('window', 180)

    try:
        # Request buffer days to properly set up initial moving averages
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days_window + 50}d")
        if hist.empty or len(hist) < slow_len:
            return jsonify({"error": "Insufficient trade depth to execute model parameters."})

        # Calculate user indicators
        hist['FastMA'] = hist['Close'].rolling(window=fast_len).mean()
        hist['SlowMA'] = hist['Close'].rolling(window=slow_len).mean()

        # Trim data down to only evaluate inside the requested historical window bounds
        test_df = hist.iloc[-days_window:].copy()
        
        trade_logs = []
        in_position = False
        initial_price = float(test_df['Close'].iloc[0])
        final_price = float(test_df['Close'].iloc[-1])
        
        # Calculate baseline benchmark return performance
        baseline_return = ((final_price - initial_price) / initial_price) * 100

        # Simulate strategy parameters iteratively down the timeline array
        for i in range(1, len(test_df)):
            idx = test_df.index[i]
            prev_idx = test_df.index[i-1]
            
            p_fast = float(test_df.loc[prev_idx, 'FastMA'])
            p_slow = float(test_df.loc[prev_idx, 'SlowMA'])
            c_fast = float(test_df.loc[idx, 'FastMA'])
            c_slow = float(test_df.loc[idx, 'SlowMA'])
            close_val = float(test_df.loc[idx, 'Close'])
            date_str = idx.strftime('%Y-%m-%d')

            if p_fast <= p_slow and c_fast > c_slow and not in_position:
                trade_logs.append({"date": date_str, "signal": "BUY", "price": close_val})
                in_position = True
            elif p_fast >= p_slow and c_fast < c_slow and in_position:
                trade_logs.append({"date": date_str, "signal": "SELL", "price": close_val})
                in_position = False

        # Calculate performance yields of all executed strategy cycles
        strategy_return = 0.0
        if len(trade_logs) > 0:
            # Simple approximation of compounding trade returns
            perf_factor = 1.0
            buy_price = None
            for tx in trade_logs:
                if tx["signal"] == "BUY":
                    buy_price = tx["price"]
                elif tx["signal"] == "SELL" and buy_price is not None:
                    trade_perf = (tx["price"] - buy_price) / buy_price
                    perf_factor *= (1.0 + trade_perf)
                    buy_price = None
            # If still open, calculate current mark-to-market performance
            if buy_price is not None:
                trade_perf = (final_price - buy_price) / buy_price
                perf_factor *= (1.0 + trade_perf)
            strategy_return = (perf_factor - 1.0) * 100

        return jsonify({
            "strategy_perf": round(strategy_return, 2),
            "baseline_perf": round(baseline_return, 2),
            "total_signals": len(trade_logs),
            "log": trade_logs
        })
    except Exception as e:
        return jsonify({"error": f"Simulation failure context: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
