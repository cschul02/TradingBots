import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# --- FINANCIAL CONFIGURATION ---
STARTING_EQUITY = 1000.00          # Initial capital.
TRANSACTION_COST_PER_TRADE = 5.00   # Round-trip cost per contract (5 contracts * $1.00 each way).
CONTRACTS_PER_TRADE = 1             # Number of NQ futures contracts traded per signal. (Reduced Risk)
ANNUALIZED_RISK_FREE_RATE = 0.04    # 4% annual risk-free rate for Sharpe Ratio

# --- STRATEGY PARAMETERS (Pullback Logic) ---
WICK_TOLERANCE_PERCENT = 0.05       # Max size of the wick opposite to momentum (e.g., 0.05 = 5% of the candle body)
MAX_PULLBACK_BARS = 5               # Maximum number of bars after the signal candle to wait for the pullback entry (5m bars * 5 = 25 minutes)

# --- RISK MANAGEMENT PARAMETERS (Based on NQ Index Points) ---
STOP_LOSS_POINTS = 30.0             # 30 NQ points SL.
TAKE_PROFIT_POINTS = 60.0           # 60 NQ points TP (1:2 Risk/Reward).
BE_TRIGGER_POINTS = 30.0            # Points needed to trigger break-even stop move (50% of TP)
MAX_DAILY_TRADES = 2                # Max number of entries per calendar day.

# --- DATA FETCHING CONFIGURATION (NQ Futures) ---
TICKER = 'NQ=F'                     # Nasdaq 100 Futures
END_DATE_API = datetime.now()
START_DATE_API = END_DATE_API - timedelta(days=55) # Use 55 days to avoid yfinance limits
INTERVAL = '5m'                     # Using 5m bars for entry
TIMEZONE = 'America/New_York'       # Crucial for time handling

# --- REPORTING CONFIGURATION ---
REPORT_FILE_PATH = 'momentum_pullback_report.txt'

# --- CORE HELPER FUNCTIONS ---

def is_bullish_momentum(bar, tolerance_percent):
    """Checks for a bullish (green) candle with a small or non-existent bottom wick."""
    
    # Extract scalar values safely
    close = bar['Close'].item()
    open_p = bar['Open'].item()
    low = bar['Low'].item()
    
    if close <= open_p:
        return False # Must be a green candle

    body_size = close - open_p
    if body_size <= 0: return False # Guard against zero body size
    
    max_wick_size = body_size * tolerance_percent
    bottom_wick_size = open_p - low
    
    # Bullish momentum if the bottom wick is very small relative to the body
    return bottom_wick_size <= max_wick_size

def is_bearish_momentum(bar, tolerance_percent):
    """Checks for a bearish (red) candle with a small or non-existent top wick."""
    
    # Extract scalar values safely
    close = bar['Close'].item()
    open_p = bar['Open'].item()
    high = bar['High'].item()
    
    if close >= open_p:
        return False # Must be a red candle
    
    body_size = open_p - close
    if body_size <= 0: return False # Guard against zero body size
    
    max_wick_size = body_size * tolerance_percent
    top_wick_size = high - open_p
    
    # Bearish momentum if the top wick is very small relative to the body
    return top_wick_size <= max_wick_size

def generate_signals(df):
    """
    Generates delayed entry trade signals based on momentum candle pullback.
    
    A signal is marked if a momentum candle is followed by a pullback (in subsequent bars) 
    to the candle's extreme (High for Long, Low for Short).
    """
    df = df.copy()
    # Initialize columns which will be set when a trade entry is confirmed
    df['Signal'] = 0          # 1=Long Entry, -1=Short Entry, 0=Hold
    df['Entry_Level'] = np.nan # The exact price the trade should be executed at
    
    # Localize and convert index to New York time (if not already done)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(TIMEZONE)
    else:
        df.index = df.index.tz_convert(TIMEZONE)

    for i in range(len(df)):
        current_bar = df.iloc[i]
        
        # Check for Long Setup: Bullish momentum candle (no bottom wick)
        if is_bullish_momentum(current_bar, WICK_TOLERANCE_PERCENT):
            
            # Entry is at the HIGH of the momentum candle
            entry_level = current_bar['High'].item() # Use .item() for scalar
            
            # Look ahead for a pullback entry
            for j in range(i + 1, min(i + 1 + MAX_PULLBACK_BARS, len(df))):
                pullback_bar = df.iloc[j]
                
                # Entry Condition: The pullback bar's LOW hits or goes below the entry level (High of signal candle)
                if pullback_bar['Low'].item() <= entry_level:
                    # Mark the entry for this bar at the exact Entry_Level price
                    df.loc[pullback_bar.name, 'Signal'] = 1
                    df.loc[pullback_bar.name, 'Entry_Level'] = entry_level
                    break # Stop looking for entry after the first fill
                    
        # Check for Short Setup: Bearish momentum candle (no top wick)
        elif is_bearish_momentum(current_bar, WICK_TOLERANCE_PERCENT):
            
            # Entry is at the LOW of the momentum candle
            entry_level = current_bar['Low'].item() # Use .item() for scalar
            
            # Look ahead for a pullback entry
            for j in range(i + 1, min(i + 1 + MAX_PULLBACK_BARS, len(df))):
                pullback_bar = df.iloc[j]

                # Entry Condition: The pullback bar's HIGH hits or goes above the entry level (Low of signal candle)
                if pullback_bar['High'].item() >= entry_level:
                    # Mark the entry for this bar at the exact Entry_Level price
                    df.loc[pullback_bar.name, 'Signal'] = -1
                    df.loc[pullback_bar.name, 'Entry_Level'] = entry_level
                    break # Stop looking for entry after the first fill

    # FIX: Filter explicitly for rows where a signal was set (Signal != 0) 
    # instead of dropping NaNs, which prevents the KeyError.
    return df[df['Signal'] != 0].copy()

def run_backtest(df):
    """
    Executes the backtest using the Momentum Pullback signals and fixed-point risk management.
    Uses the Entry_Level price marked in the signal generation.
    """
    
    equity = STARTING_EQUITY
    trade_log = []
    equity_curve = pd.DataFrame(columns=['Date', 'Equity'])
    in_trade = False
    daily_trade_count = {}
    
    # Variables to track an active trade
    direction = 0
    entry_price = 0.0
    entry_date = None
    tp_target = 0.0
    
    # DYNAMIC STOP VARIABLES
    current_stop = 0.0
    be_triggered = False
    
    for index, current_bar in df.iterrows():
        current_date = index.date()
        equity_curve.loc[len(equity_curve)] = {'Date': index, 'Equity': equity}
        
        # --- OPEN TRADE LOGIC ---
        if not in_trade:
            if current_date not in daily_trade_count:
                daily_trade_count[current_date] = 0
            
            signal_value = int(current_bar['Signal'])
            
            if (signal_value != 0) and (daily_trade_count[current_date] < MAX_DAILY_TRADES):
                
                # Execute Trade
                in_trade = True
                entry_price = float(current_bar['Entry_Level']) # CRITICAL: Use the calculated Entry_Level
                entry_date = index
                direction = int(signal_value)
                daily_trade_count[current_date] += 1

                # Calculate fixed targets based on the actual entry price
                sl_target = entry_price - (STOP_LOSS_POINTS * direction)
                tp_target = entry_price + (TAKE_PROFIT_POINTS * direction)
                
                # Initialize the current active stop to the static stop loss
                current_stop = sl_target
                be_triggered = False

        # --- CLOSE TRADE LOGIC ---
        elif in_trade:
            # We assume price can hit targets/SL/BE in the bars following the entry.
            high = float(current_bar['High'].item())
            low = float(current_bar['Low'].item())
            current_close = float(current_bar['Close'].item())
            
            exit_reason = None
            exit_price = 0.0
            
            # 1. DYNAMIC STOP MANAGEMENT (Break-Even Move)
            current_profit_points = (current_close - entry_price) * direction
            
            if (current_profit_points >= BE_TRIGGER_POINTS) and (not be_triggered):
                # Move the current_stop to the entry price (break-even)
                current_stop = entry_price
                be_triggered = True
                
            
            # 2. Check for Take Profit (TP) or Stop Loss (SL)
            
            if direction == 1: # Long Trade
                if high >= tp_target:
                    exit_reason = "Take Profit (TP)"
                    exit_price = tp_target
                # Check against the dynamic stop
                elif low <= current_stop:
                    exit_reason = "Break-Even SL" if be_triggered else "Stop Loss (SL)"
                    exit_price = current_stop
                    
            elif direction == -1: # Short Trade
                if low <= tp_target:
                    exit_reason = "Take Profit (TP)"
                    exit_price = tp_target
                # Check against the dynamic stop
                elif high >= current_stop:
                    exit_reason = "Break-Even SL" if be_triggered else "Stop Loss (SL)"
                    exit_price = current_stop
            
            # 3. Process Exit
            if exit_reason is not None:
                pnl_points = (exit_price - entry_price) * direction
                
                # NQ futures are typically $20 per point 
                pnl_dollars = (pnl_points * CONTRACTS_PER_TRADE * 20) - TRANSACTION_COST_PER_TRADE
                
                equity += pnl_dollars
                
                trade_log.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': index,
                    'Direction': 'Long' if direction == 1 else 'Short',
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'PnL_Dollars': pnl_dollars,
                    'Exit_Reason': exit_reason,
                    'PnL_Points': pnl_points
                })
                
                # Reset trade state
                in_trade = False
                direction = 0
                entry_price = 0.0
                
    # Append final equity point
    if not equity_curve.empty and not df.empty:
        equity_curve.loc[len(equity_curve)] = {'Date': df.index[-1], 'Equity': equity}

    return pd.DataFrame(trade_log), equity, equity_curve

# --- METRICS AND REPORTING FUNCTIONS (Copied from previous file for consistency) ---

def calculate_metrics(trade_df, final_equity, equity_curve):
    """Calculates key trading performance metrics, including Sharpe Ratio and Drawdown."""
    total_trades = len(trade_df)
    total_pnl = trade_df['PnL_Dollars'].sum()
    
    if total_trades == 0:
        return {'Total Net PnL': 0.0, 'Total Trades': 0, 'Win Rate': 0.0, 'Avg PnL': 0.0, 'Winning Trades': 0, 'Losing Trades': 0, 'Sharpe Ratio': 0.0, 'Max Drawdown': 0.0, 'Recovery Factor': 0.0}

    # --- Standard Metrics ---
    winning_trades = len(trade_df[trade_df['PnL_Dollars'] > 0])
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100
    avg_pnl = total_pnl / total_trades
    
    # --- Equity Curve Analysis (for Sharpe and Drawdown) ---
    daily_equity = equity_curve.set_index('Date')['Equity'].resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    
    annualization_factor = np.sqrt(252) 
    
    avg_daily_return = daily_returns.mean()
    std_dev_daily_return = daily_returns.std()
    
    if std_dev_daily_return == 0:
        sharpe_ratio = 0.0
    else:
        annualized_return = avg_daily_return * 252
        annualized_volatility = std_dev_daily_return * annualization_factor
        sharpe_ratio = (annualized_return - ANNUALIZED_RISK_FREE_RATE) / annualized_volatility

    # 2. Maximum Drawdown (MDD)
    equity_values = equity_curve['Equity']
    peak = equity_values.cummax()
    peak_positive = peak.replace(0, 1e-6) 
    drawdown = (peak - equity_values) / peak_positive
    max_drawdown = drawdown.max()
    
    # 3. Recovery Factor
    max_loss_dollar = max_drawdown * STARTING_EQUITY # Using the percentage of starting equity
    if max_loss_dollar > 0:
        recovery_factor = total_pnl / max_loss_dollar
    else:
        recovery_factor = float('inf') if total_pnl > 0 else 0.0

    return {
        'Total Net PnL': total_pnl,
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate': win_rate,
        'Avg PnL': avg_pnl,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Recovery Factor': recovery_factor
    }

def print_results(trade_df, metrics, final_equity):
    """Prints the backtest results to the redirected output file."""
    
    print("\n" + "=" * 80)
    print("  MOMENTUM NO-WICK PULLBACK STRATEGY BACKTEST REPORT")
    print("=" * 80)
    print(f"Ticker: {TICKER} | Interval: {INTERVAL} | Timezone: {TIMEZONE}")
    print(f"Time Period: {START_DATE_API.strftime('%Y-%m-%d')} to {END_DATE_API.strftime('%Y-%m-%d')}")
    print(f"Strategy: Momentum No-Wick (Tolerance {WICK_TOLERANCE_PERCENT*100:.0f}%) Pullback (Max {MAX_PULLBACK_BARS} bars)")
    print(f"Risk Management: SL {STOP_LOSS_POINTS:.0f} pts | TP {TAKE_PROFIT_POINTS:.0f} pts | BE Trigger {BE_TRIGGER_POINTS:.0f} pts | R:R 1:{TAKE_PROFIT_POINTS/STOP_LOSS_POINTS:.0f}")
    print("-" * 80)

    # --- Financial Performance Summary ---
    print(f"Starting Equity: ${STARTING_EQUITY:,.2f}")
    print(f"Final Balance:   ${final_equity:,.2f} {'(Profit)' if metrics['Total Net PnL'] >= 0 else '(Loss)'}")
    print(f"Total Net PnL:   ${metrics['Total Net PnL']:,.2f}")
    print("-" * 80)

    # --- Analytical Metrics ---
    print("\n--- ANALYTICAL METRICS ---")
    print(f"Sharpe Ratio (Annualized): {metrics['Sharpe Ratio']:.2f} (Higher is better)")
    print(f"Maximum Drawdown (MDD):    {metrics['Max Drawdown'] * 100:.2f}% (Lower is better)")
    print(f"Recovery Factor:           {metrics['Recovery Factor']:.2f} (Above 1 is good)")
    print("--------------------------")
    
    # --- Trade Statistics ---
    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades Executed: {metrics['Total Trades']}")
    print(f"Winning Trades:        {metrics['Winning Trades']}")
    print(f"Losing Trades:         {metrics['Losing Trades']}")
    print(f"Win Rate:              {metrics['Win Rate']:.2f}%")
    print(f"Average PnL per Trade: ${metrics['Avg PnL']:,.2f}")
    print("="*80)

    # Display the trade log (formatted)
    if not trade_df.empty:
        trade_df_display = trade_df.copy()
        
        # Ensure Entry_Date and Exit_Date are timezone-aware before formatting
        if trade_df_display['Entry_Date'].dt.tz is None:
             trade_df_display['Entry_Date'] = trade_df_display['Entry_Date'].dt.tz_localize(TIMEZONE)
        if trade_df_display['Exit_Date'].dt.tz is None:
             trade_df_display['Exit_Date'] = trade_df_display['Exit_Date'].dt.tz_localize(TIMEZONE)
             
        trade_df_display['Entry_Time'] = trade_df_display['Entry_Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        trade_df_display['Exit_Time'] = trade_df_display['Exit_Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        print("\n--- COMPLETE TRADE LOG ---")
        cols = ['Entry_Time', 'Exit_Time', 'Direction', 'Entry_Price', 'Exit_Price', 'PnL_Points', 'PnL_Dollars', 'Exit_Reason']
        print(trade_df_display[cols].to_string(index=False, float_format="%.2f"))

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    
    original_stdout = sys.stdout 
    
    try:
        print(f"Starting MOMENTUM PULLBACK backtest for {TICKER} using the {INTERVAL} interval. Report saved to {REPORT_FILE_PATH}")
        
        # --- Data Fetching ---
        df = yf.download(TICKER, start=START_DATE_API, end=END_DATE_API, interval=INTERVAL)
        
        if df.empty:
            raise Exception("DataFrame is empty after fetching. Check ticker, dates, and interval.")
            
        # --- 1. Generate Signals ---
        # No indicators needed, just raw price action analysis
        df = generate_signals(df) 
        
        # --- 2. Run Backtest ---
        trade_df, final_equity, equity_curve = run_backtest(df.copy())
        
        # --- 3. Calculate Metrics ---
        metrics = calculate_metrics(trade_df, final_equity, equity_curve)
        
        # --- 4. Redirect and Print Results to File ---
        with open(REPORT_FILE_PATH, 'w') as f:
            sys.stdout = f 
            print_results(trade_df, metrics, final_equity)
            
        sys.stdout = original_stdout 
        print(f"\nSUCCESS! Backtest completed. Detailed report saved to {REPORT_FILE_PATH}")

    except Exception as e:
        sys.stdout = original_stdout 
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        sys.stdout = original_stdout