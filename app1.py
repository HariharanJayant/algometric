# === REPAIRED DATA COUPLING MAPPER FOR BACKTEST SIMULATION ===
@app.route('/simulate', methods=['POST'])
def simulate():
    # Grabs input parameters directly from the tab form elements
    ticker_symbol = request.form.get('ticker', 'MSFT').upper().strip()
    fast_length = int(request.form.get('fast_length', 9))
    slow_length = int(request.form.get('slow_length', 21))
    data_window = request.form.get('data_window', '365') # Defaulting to 1-Year as specified by template

    # Convert select element value to historical period syntax
    period_map = {"180": "6mo", "365": "1y", "90": "3mo"}
    yf_period = period_map.get(data_window, "1y")

    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=yf_period, interval="1d")

        if df.empty:
            return jsonify({'error': f"No historical trace records found for {ticker_symbol}."})

        # Calculate Indicators
        df['Fast_MA'] = ta.trend.sma_indicator(df['Close'], window=fast_length)
        df['Slow_MA'] = ta.trend.sma_indicator(df['Close'], window=slow_length)
        df = df.dropna()

        initial_capital = 1000.0
        capital = initial_capital
        position = 0
        total_trades = 0
        winning_trades = 0
        entry_price = 0.0

        for i in range(1, len(df)):
            prev_fast, prev_slow = df['Fast_MA'].iloc[i-1], df['Slow_MA'].iloc[i-1]
            curr_fast, curr_slow = df['Fast_MA'].iloc[i], df['Slow_MA'].iloc[i]
            price = round(df['Close'].iloc[i], 2)

            # Buy Signal
            if prev_fast <= prev_slow and curr_fast > curr_slow and position == 0:
                position = capital / price
                entry_price = price
                capital = 0
                total_trades += 1
            
            # Sell Signal
            elif prev_fast >= prev_slow and curr_fast < curr_slow and position > 0:
                capital = position * price
                position = 0
                if price > entry_price:
                    winning_trades += 1

        if position > 0:
            capital = position * df['Close'].iloc[-1]
            if df['Close'].iloc[-1] > entry_price:
                winning_trades += 1

        net_profit_pct = round(((capital - initial_capital) / initial_capital) * 100, 2)
        win_rate_pct = round((winning_trades / total_trades) * 100, 2) if total_trades > 0 else 0.0

        # Matches exact keys expected by JavaScript variable mappings
        return jsonify({
            'success': True,
            'total_trades': total_trades,
            'win_rate': f"{win_rate_pct}%",
            'net_profit_pct': f"{net_profit_pct}%",
            'final_balance': f"{round(capital, 2):,}"
        })
    except Exception as e:
        return jsonify({'error': f"Sim-Engine Crash: {str(e)}"})
