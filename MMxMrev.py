import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# --- FINANCIAL CONFIGURATION ---
STARTING_EQUITY = 1000.00          # Initial capital.
TRANSACTION_COST_PER_TRADE = 5.00   # Round-trip cost per contract.
CONTRACTS_PER_TRADE = 1             # Number of NQ futures contracts traded per signal. <--- REDUCED LEVERAGE TO 10
ANNUALIZED_RISK_FREE_RATE = 0.04    # 4% annual risk-free rate for Sharpe Ratio

# --- STRATEGY PARAMETERS (MMxM Bias and Timing) ---
# BIAS: The EMA 200 on the 15m chart is used as a proxy for the long-term daily bias.
BIAS_EMA_PERIOD = 195

# MMxM / PO3 TIMING: Focus on the NY Open manipulation window.
NY_OPEN_HOUR = 9                    # 9 AM EST (New York)
NY_OPEN_MINUTE = 30                 # 9:30 AM EST
# EXTENDED WINDOW FOR 15M BARS: Looking for signals up until the 10:45 AM bar close.
MANIPULATION_END_HOUR = 11          # Stop looking for entries after 11:00 AM EST

# --- RISK MANAGEMENT PARAMETERS (Based on NQ Index Points) ---
STOP_LOSS_POINTS = 30.0             # 30 NQ points SL.
TAKE_PROFIT_POINTS = 60.0           # 60 NQ points TP (1:2 Risk/Reward).
BE_TRIGGER_POINTS = 30.0            # Points needed to trigger break-even stop move (50% of TP)
MAX_DAILY_TRADES = 3                # Max number of entries per calendar day.

# --- DATA FETCHING CONFIGURATION (NQ Futures) ---
TICKER = 'NQ=F'                     # Nasdaq 100 Futures
# CRITICAL FIX: Reduced days to 55 to respect yfinance's 60-day limit for intraday data.
END_DATE_API = datetime.now()
START_DATE_API = END_DATE_API - timedelta(days=55) 
INTERVAL = '5m'                    # Using 5m bars for entry
TIMEZONE = 'America/New_York'       # Crucial for defining the NY open window

# --- REPORTING CONFIGURATION ---
REPORT_FILE_PATH = 'mmxm_po3_report_reversed.txt' # <-- New report name

# --- CORE FUNCTIONS ---

def calculate_indicators(df):
    """Calculates the 200 EMA used for long-term bias determination."""
    df[f'EMA_{BIAS_EMA_PERIOD}'] = df['Close'].ewm(span=BIAS_EMA_PERIOD, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    """
    Generates REVERSED MMxM trade signals based on Liquidity Sweep (Manipulation) and Reversal.
    
    If the original MMxM strategy fails (sweep->reversal), we fade the sweep 
    (sweep->continuation in the opposite direction).
    """
    df = df.copy()
    df['Signal'] = 0 # 1=Long, -1=Short, 0=Hold

    # Localize and convert index to New York time for session-based logic
    df.index = df.index.tz_convert(TIMEZONE)
    
    start_index = df[f'EMA_{BIAS_EMA_PERIOD}'].first_valid_index()
    if start_index is None:
        return df # Handle case where DF is too short for EMA

    start_i = df.index.get_loc(start_index)
    
    for i in range(start_i, len(df)):
        current_bar = df.iloc[i]
        
        # Check if the current bar is within the NY Open manipulation window
        current_time = current_bar.name.time()
        entry_time_start = current_bar.name.replace(hour=NY_OPEN_HOUR, minute=NY_OPEN_MINUTE, second=0, microsecond=0).time()
        entry_time_end = current_bar.name.replace(hour=MANIPULATION_END_HOUR, minute=0, second=0, microsecond=0).time()

        if entry_time_start <= current_time < entry_time_end:
            
            if i > 0:
                prev_bar = df.iloc[i-1]
                
                # Explicitly cast to float
                ema = float(prev_bar[f'EMA_{BIAS_EMA_PERIOD}'])
                prev_close = float(prev_bar['Close'])
                prev_high = float(prev_bar['High'])
                prev_low = float(prev_bar['Low'])
                current_close = float(current_bar['Close'])
                current_open = float(current_bar['Open'])
                current_high = float(current_bar['High'])
                current_low = float(current_bar['Low'])

                # --- REVERSED BUY MODEL (Fading a Bearish Sweep) ---
                # ORIGINAL: Bullish bias, sweep down, then buy.
                # REVERSED: Bullish bias, sweep down, but the reversal fails, so we short.
                if prev_close > ema: # Bullish Bias
                    
                    # 1. Liquidity Sweep: Previous bar must dip below the EMA (stop run)
                    sweep_low = prev_low < ema
                    
                    # 2. Confirmation Failure (The 'Reversal' fails to hold): Current bar closes bearishly, confirming the breakdown
                    reversal_failure_close = current_close < ema
                    
                    # 3. Aggressive Bearish Confirmation: Current bar closes red and below the previous bar's low
                    strong_confirmation = (current_close < current_open) and (current_close < prev_low)
                    
                    if sweep_low and reversal_failure_close and strong_confirmation:
                        df.loc[current_bar.name, 'Signal'] = -1 # <--- REVERSED TO SHORT SIGNAL
                        
                # --- REVERSED SELL MODEL (Fading a Bullish Sweep) ---
                # ORIGINAL: Bearish bias, sweep up, then sell.
                # REVERSED: Bearish bias, sweep up, but the reversal fails, so we long.
                elif prev_close < ema: # Bearish Bias
                    
                    # 1. Liquidity Sweep: Previous bar must spike above the EMA (stop run)
                    sweep_high = prev_high > ema

                    # 2. Confirmation Failure (The 'Reversal' fails to hold): Current bar closes bullishly, confirming the breakout
                    reversal_failure_close = current_close > ema
                    
                    # 3. Aggressive Bullish Confirmation: Current bar closes green and above the previous bar's high
                    strong_confirmation = (current_close > current_open) and (current_close > prev_high)
                    
                    if sweep_high and reversal_failure_close and strong_confirmation:
                        df.loc[current_bar.name, 'Signal'] = 1 # <--- REVERSED TO LONG SIGNAL

    return df

def run_backtest(df):
    """
    Executes the backtest using the MMxM signals and fixed-point risk management.
    Includes dynamic stop logic to move to break-even after 50% TP is hit.
    """
    
    equity = STARTING_EQUITY
    trade_log = []
    equity_curve = pd.DataFrame(columns=['Date', 'Equity'])
    in_trade = False
    direction = 0
    daily_trade_count = {}
    
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
                entry_price = float(current_bar['Close'])
                entry_date = index
                direction = int(signal_value)
                daily_trade_count[current_date] += 1

                # Calculate fixed targets
                sl_target = entry_price - (STOP_LOSS_POINTS * direction)
                tp_target = entry_price + (TAKE_PROFIT_POINTS * direction)
                
                # Initialize the current active stop to the static stop loss
                current_stop = sl_target
                be_triggered = False

        # --- CLOSE TRADE LOGIC ---
        elif in_trade:
            high = float(current_bar['High'])
            low = float(current_bar['Low'])
            current_close = float(current_bar['Close'])
            
            exit_reason = None
            exit_price = 0.0
            
            # 1. DYNAMIC STOP MANAGEMENT (Break-Even Move)
            # Check if profit threshold has been reached and BE hasn't been triggered yet
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
                # Check against the dynamic stop (which is either SL or Break-Even)
                elif low <= current_stop:
                    exit_reason = "Break-Even SL" if be_triggered else "Stop Loss (SL)"
                    exit_price = current_stop
                    
            elif direction == -1: # Short Trade
                if low <= tp_target:
                    exit_reason = "Take Profit (TP)"
                    exit_price = tp_target
                # Check against the dynamic stop (which is either SL or Break-Even)
                elif high >= current_stop:
                    exit_reason = "Break-Even SL" if be_triggered else "Stop Loss (SL)"
                    exit_price = current_stop
            
            # 3. Process Exit
            if exit_reason is not None:
                pnl_points = (exit_price - entry_price) * direction
                
                # NQ futures are typically $20 per point 
                pnl_dollars = (pnl_points * CONTRACTS_PER_TRADE * 20) - TRANSACTION_COST_PER_TRADE # Assuming $20 per point NQ
                
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
                
    # Append final equity point
    if not equity_curve.empty and not df.empty:
        equity_curve.loc[len(equity_curve)] = {'Date': df.index[-1], 'Equity': equity}

    return pd.DataFrame(trade_log), equity, equity_curve

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
    
    # 1. Sharpe Ratio
    # Calculate daily returns (must resample the curve to daily, padding with last known value)
    daily_equity = equity_curve.set_index('Date')['Equity'].resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    
    # Assume 252 trading days in a year for annualization
    annualization_factor = np.sqrt(252) 
    
    avg_daily_return = daily_returns.mean()
    std_dev_daily_return = daily_returns.std()
    
    if std_dev_daily_return == 0:
        sharpe_ratio = 0.0
    else:
        # Annualized Sharpe Ratio = (Annualized Return - Annualized Risk-Free Rate) / Annualized Volatility
        annualized_return = avg_daily_return * 252
        annualized_volatility = std_dev_daily_return * annualization_factor
        
        sharpe_ratio = (annualized_return - ANNUALIZED_RISK_FREE_RATE) / annualized_volatility

    # 2. Maximum Drawdown (MDD)
    equity_values = equity_curve['Equity']
    peak = equity_values.cummax()
    # Check if peak is ever zero or negative to prevent ZeroDivisionError
    peak_positive = peak.replace(0, 1e-6) 
    drawdown = (peak - equity_values) / peak_positive
    max_drawdown = drawdown.max()
    
    # 3. Recovery Factor
    # Using the absolute dollar value of the largest drawdown. We'll use the initial equity for MDD calculation
    # to avoid the 416% issue, but for the factor, we use the MDD percentage * STARTING_EQUITY
    max_loss_dollar = max_drawdown * STARTING_EQUITY

    if max_loss_dollar > 0:
        recovery_factor = total_pnl / max_loss_dollar
    else:
        recovery_factor = float('inf') if total_pnl > 0 else 0.0 # Handles scenario with no drawdown or MDD=0

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
    print("  MMxM REVERSED (FADER MODEL) + PO3 INTRADAY FUTURES BACKTEST REPORT")
    print("=" * 80)
    print(f"Ticker: {TICKER} | Interval: {INTERVAL} | Timezone: {TIMEZONE}")
    print(f"Time Period: {START_DATE_API.strftime('%Y-%m-%d')} to {END_DATE_API.strftime('%Y-%m-%d')}")
    print(f"Strategy: REVERSED MMxM (9:30-11:00 AM EST) Fades the EMA {BIAS_EMA_PERIOD} Sweep/Reversal")
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
        # Create a new DataFrame for clean display, converting timestamps to string format
        trade_df_display = trade_df.copy()
        
        # Ensure Entry_Date and Exit_Date are timezone-aware before formatting
        if trade_df_display['Entry_Date'].dt.tz is None:
             trade_df_display['Entry_Date'] = trade_df_display['Entry_Date'].dt.tz_localize(TIMEZONE)
        if trade_df_display['Exit_Date'].dt.tz is None:
             trade_df_display['Exit_Date'] = trade_df_display['Exit_Date'].dt.tz_localize(TIMEZONE)
             
        trade_df_display['Entry_Time'] = trade_df_display['Entry_Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        trade_df_display['Exit_Time'] = trade_df_display['Exit_Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        print("\n--- COMPLETE TRADE LOG ---")
        # Ensure all columns are present before printing
        cols = ['Entry_Time', 'Exit_Time', 'Direction', 'Entry_Price', 'Exit_Price', 'PnL_Points', 'PnL_Dollars', 'Exit_Reason']
        print(trade_df_display[cols].to_string(index=False, float_format="%.2f"))

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    
    # Setup output redirection to file
    original_stdout = sys.stdout 
    
    try:
        # Print status message to the console
        print(f"Starting REVERSED MMxM (Fader) backtest for {TICKER} using the {INTERVAL} interval. Full report will be saved to {REPORT_FILE_PATH}")
        
        # --- Data Fetching ---
        # The date range is now fixed to 55 days to resolve the Yahoo Finance API error.
        df = yf.download(TICKER, start=START_DATE_API, end=END_DATE_API, interval=INTERVAL)
        
        if df.empty:
            raise Exception("DataFrame is empty after fetching. Check ticker, dates, and interval.")
            
        # --- 1. Prepare Data and Signals ---
        df = calculate_indicators(df)
        if df.empty:
            raise Exception("DataFrame is empty after calculating indicators (too few bars for EMA).")
            
        df = generate_signals(df) # Timezone conversion and signal generation
        
        # --- 2. Run Backtest ---
        trade_df, final_equity, equity_curve = run_backtest(df.copy())
        
        # --- 3. Calculate Metrics (Now requires equity_curve) ---
        metrics = calculate_metrics(trade_df, final_equity, equity_curve)
        
        # --- 4. Redirect and Print Results to File ---
        with open(REPORT_FILE_PATH, 'w') as f:
            sys.stdout = f # Redirect print output to file 'f'
            print_results(trade_df, metrics, final_equity)
            
        # Final success message to console
        sys.stdout = original_stdout # Restore console output
        print(f"\nSUCCESS! Backtest completed. Detailed report saved to {REPORT_FILE_PATH}")

    except Exception as e:
        sys.stdout = original_stdout # Restore console output for error reporting
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ensures console output is restored even if an error occurred before the try block finished
        sys.stdout = original_stdout