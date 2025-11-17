import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# --- CONFIGURATION ---
STARTING_EQUITY = 10000.00
TRANSACTION_COST_PER_TRADE = 5.00  # $5.00 round trip
CONTRACTS_PER_TRADE = 10

# Strategy Parameters (Fixed)
EMA_FAST = 10
EMA_MID = 40
EMA_SLOW = 200

# NEW PERCENTAGE RISK MANAGEMENT PARAMETERS (Required for QQQ and long-term scaling)
# Percentage-based risk scales correctly across different price levels.
STOP_LOSS_PERCENT = 0.005   # 0.5% Stop Loss
TAKE_PROFIT_PERCENT = 0.010 # 1.0% Take Profit

# Profit Trailing Logic (Percentage-based)
# Move SL when price moves 0.5% in profit
MOVE_SL_PROFIT_PERCENT = 0.005
# Move SL to lock in 0.25% profit
NEW_SL_PROFIT_PERCENT = 0.0025

# General Risk Management
MAX_HOLDING_DAYS = 1        # Maximum number of days a trade can be held
MAX_DAILY_TRADES = 2        # Maximum number of entries per calendar day

# --- DATA FETCHING CONFIGURATION (Using yfinance) ---
# Switched to QQQ ETF for deep 60m historical data
TICKER = 'QQQ' 
START_DATE_API = '2023-11-07' # Extended date range
END_DATE_API = datetime.date.today().strftime('%Y-%m-%d') # End date is today
INTERVAL = '60m'

# --- DATA HANDLING ---

def load_data():
    """
    Fetches real historical data using the yfinance library.
    """
    print(f"Fetching real historical data for {TICKER} ({INTERVAL} bars) from {START_DATE_API} to {END_DATE_API}...")

    try:
        # yfinance automatically fetches OHLCV data
        df = yf.download(TICKER, start=START_DATE_API, end=END_DATE_API, interval=INTERVAL)

        if df.empty:
            print(f"ERROR: No data returned for {TICKER}. Check the ticker, dates, and internet connection.")
            return pd.DataFrame() # Return empty DataFrame

        print(f"Successfully loaded {len(df)} bars of real data.")

        # Force all selected columns to lowercase
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower).copy()

    except Exception as e:
        print(f"FATAL API ERROR: Could not fetch data using yfinance. Check your setup. Error: {e}")
        return pd.DataFrame()


# --- INDICATOR CALCULATION ---

def calculate_emas(df):
    """Calculates the 10, 40, and 200 EMAs and adds them to the DataFrame."""
    if df.empty:
        return df

    df['ema13'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema48'] = df['close'].ewm(span=EMA_MID, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    # Drop initial rows where EMAs are invalid
    df.dropna(inplace=True)
    return df


# --- CORE BACKTEST LOGIC ---

def run_backtest(df):
    """
    Runs the backtest strategy with percentage SL/TP and a trailing mechanism,
    enforcing daily trade limits.
    """

    trades = []
    position = 0      # 0: Flat, 1: Long, -1: Short
    entry_price = 0
    entry_date = None
    sl_price = 0
    tp_price = 0
    sl_moved = False  # Tracks if the Stop Loss has been moved up/down for profit protection

    # Daily Tracking Variables
    current_date = None
    trades_today = 0

    # Using iterrows() for safe scalar access
    for index, row in df.iterrows():

        # Setup daily counter reset
        bar_date = index.date()
        if bar_date != current_date:
            # Start of a new day: reset counter and update current_date
            trades_today = 0
            current_date = bar_date

        # Extract the scalar value using .item()
        ema13 = row['ema13'].item()
        ema48 = row['ema48'].item()
        ema200 = row['ema200'].item()
        current_close = row['close'].item()

        # Signals
        is_bullish_stack = (ema13 > ema48) and (ema48 > ema200)
        is_bearish_stack = (ema200 > ema48) and (ema48 > ema13)

        # --- 1. Manage Active Position (SL/TP/Time Stop/Crossover Checks) ---
        if position != 0:
            exit_price = 0
            exit_reason = None

            # Calculate duration based on calendar days
            duration_days_passed = (index.date() - entry_date.date()).days

            # A. Trailing Stop Loss Logic (Move SL to lock in 0.25% profit)
            if not sl_moved:
                if position == 1:
                    # Long: Check if price moved 0.5% up (MOVE_SL_PROFIT_PERCENT)
                    if current_close >= entry_price * (1 + MOVE_SL_PROFIT_PERCENT):
                        # Move SL to entry + 0.25% (NEW_SL_PROFIT_PERCENT)
                        sl_price = entry_price * (1 + NEW_SL_PROFIT_PERCENT)
                        sl_moved = True
                
                elif position == -1:
                    # Short: Check if price moved 0.5% down (MOVE_SL_PROFIT_PERCENT)
                    if current_close <= entry_price * (1 - MOVE_SL_PROFIT_PERCENT):
                        # Move SL to entry - 0.25% (NEW_SL_PROFIT_PERCENT)
                        sl_price = entry_price * (1 - NEW_SL_PROFIT_PERCENT)
                        sl_moved = True

            # B. Check for Time Stop
            if exit_reason is None and duration_days_passed >= MAX_HOLDING_DAYS:
                exit_price = current_close
                exit_reason = 'Time Stop'

            # C. Check for Stop Loss or Take Profit hit (Only if Time Stop hasn't triggered)
            if exit_reason is None:
                if position == 1:
                    # Long Position: TP is high, SL is low
                    if current_close >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                    elif current_close <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'

                elif position == -1:
                    # Short Position: TP is low, SL is high
                    if current_close <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                    elif current_close >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'

            # D. Check for structural exit (Reverse Crossover)
            if exit_reason is None:
                if position == 1 and is_bearish_stack:
                    exit_price = current_close
                    exit_reason = 'Crossover'
                elif position == -1 and is_bullish_stack:
                    exit_price = current_close
                    exit_reason = 'Crossover'

            # --- Execute Exit if trigger found ---
            if exit_reason is not None:
                exit_date = index

                # PnL calculation depends on position direction
                if position == 1:
                    pnl_per_contract = exit_price - entry_price
                else: # position == -1 (Short)
                    pnl_per_contract = entry_price - exit_price

                pnl_gross = pnl_per_contract * CONTRACTS_PER_TRADE
                pnl_net = pnl_gross - TRANSACTION_COST_PER_TRADE
                
                # Calculate Percentage Gain/Loss
                percent_change = (pnl_per_contract / entry_price) * 100 

                trades.append({
                    'Entry_Date': entry_date, 
                    'Exit_Date': exit_date, 
                    'Direction': 'Long' if position == 1 else 'Short',
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'PnL_Dollars': pnl_net,
                    'Percent_Gain': percent_change,
                    'Duration_Days': int(duration_days_passed),
                    'Exit_Reason': exit_reason
                })

                # Reset position variables
                position = 0
                entry_price = 0
                entry_date = None
                sl_price = 0
                tp_price = 0
                sl_moved = False # Reset SL moved flag

        # --- 2. Check for New Entry Signal (Only if flat AND trade limit not hit) ---
        if position == 0 and trades_today < MAX_DAILY_TRADES:

            entry_executed = False

            # Try to Enter Long
            if is_bullish_stack:
                position = 1
                entry_price = current_close
                entry_date = index
                # Calculate SL/TP using percentages of the entry price
                sl_price = entry_price * (1 - STOP_LOSS_PERCENT) 
                tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT) 
                entry_executed = True

            # Try to Enter Short
            elif is_bearish_stack:
                position = -1
                entry_price = current_close
                entry_date = index
                # Calculate SL/TP using percentages of the entry price
                sl_price = entry_price * (1 + STOP_LOSS_PERCENT) 
                tp_price = entry_price * (1 - TAKE_PROFIT_PERCENT) 
                entry_executed = True

            # If an entry was executed, increment the counter for the day
            if entry_executed:
                trades_today += 1
                sl_moved = False # Ensure this is False for new trade


    # --- 3. Handle Final Open Position ---
    if position != 0:
        last_row = df.iloc[-1]
        exit_price = last_row['close'].item()
        exit_date = last_row.name

        duration_days_passed = (exit_date.date() - entry_date.date()).days

        if position == 1:
            pnl_per_contract = exit_price - entry_price
        else: # position == -1
            pnl_per_contract = entry_price - exit_price

        pnl_gross = pnl_per_contract * CONTRACTS_PER_TRADE
        pnl_net = pnl_gross - (TRANSACTION_COST_PER_TRADE / 2) # Half cost for open trade
        
        # Calculate Percentage Gain/Loss for open position
        percent_change = (pnl_per_contract / entry_price) * 100 

        trades.append({
            'Entry_Date': entry_date, 
            'Exit_Date': exit_date, 
            'Direction': 'Long' if position == 1 else 'Short',
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'PnL_Dollars': pnl_net,
            'Percent_Gain': percent_change,
            'Duration_Days': int(duration_days_passed),
            'Exit_Reason': 'Open'
        })

    return pd.DataFrame(trades)


# --- PERFORMANCE ANALYSIS AND REPORTING ---

def analyze_performance(trade_df):
    """Calculates and prints key performance metrics and the full trade log."""

    if trade_df.empty:
        print("\n--- PERFORMANCE SUMMARY ---")
        print("No trades executed.")
        return

    trade_df['PnL_Dollars'] = pd.to_numeric(trade_df['PnL_Dollars'], errors='coerce')
    trade_df['Percent_Gain'] = pd.to_numeric(trade_df['Percent_Gain'], errors='coerce') 

    total_trades = len(trade_df)
    total_pnl = trade_df['PnL_Dollars'].sum()
    winning_trades = len(trade_df[trade_df['PnL_Dollars'] > 0])
    losing_trades = total_trades - winning_trades

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    avg_percent_gain = trade_df['Percent_Gain'].mean() if total_trades > 0 else 0 

    final_balance = STARTING_EQUITY + total_pnl

    # Format for clean output
    print("\n" + "="*70)
    print("  DYNAMIC TRIPLE EMA CROSSOVER BACKTEST REPORT")
    print("="*70)
    print(f"Data Source: yfinance API ({TICKER} - {INTERVAL} bars)")
    print(f"Period Tested: {START_DATE_API} to {END_DATE_API}")
    print(f"Strategy: Percentage-Based SL/TP with Profit Protection")
    print("-" * 70)
    # Updated output to reflect percentage settings
    print(f"Risk/Reward: SL {STOP_LOSS_PERCENT*100:.2f}% | TP {TAKE_PROFIT_PERCENT*100:.2f}%")
    print(f"Profit Protection: Move SL to +{NEW_SL_PROFIT_PERCENT*100:.2f}% when +{MOVE_SL_PROFIT_PERCENT*100:.2f}% is reached.")
    print(f"Daily Trade Limit: Max {MAX_DAILY_TRADES} Entries Per Day")
    print("-" * 70)

    print(f"Starting Equity: ${STARTING_EQUITY:,.2f}")
    print(f"Final Balance:   ${final_balance:,.2f} {'(Profit)' if total_pnl >= 0 else '(Loss)'}")
    print(f"Total Net PnL:   ${total_pnl:,.2f}")
    print("-" * 70)

    print(f"Total Trades Executed: {total_trades}")
    print(f"Winning Trades:        {winning_trades}")
    print(f"Losing Trades:         {losing_trades}")
    print(f"Win Rate:              {win_rate:.2f}%")
    print(f"Average PnL per Trade: ${avg_pnl:,.2f}")
    print(f"Average Percent Gain:  {avg_percent_gain:,.4f}%") 
    print("="*70)

    # NEW: Create columns specifically for the formatted datetime strings
    # The timezone information (Z for Zulu time/UTC or the offset) from yfinance is preserved here.
    trade_df['Entry_Time'] = trade_df['Entry_Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    trade_df['Exit_Time'] = trade_df['Exit_Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')

    # UPDATED: Show the complete trade log including the new formatted time columns
    print("\n--- COMPLETE TRADE LOG (Timezone included from API Data) ---")
    print(trade_df[['Entry_Time', 'Exit_Time', 'Direction', 'Entry_Price', 'Exit_Price', 'Percent_Gain', 'PnL_Dollars', 'Exit_Reason']].to_string(index=False, float_format="%.4f"))


# --- MAIN EXECUTION ---

if __name__ == "__main__":

    # 1. Load Data from API
    data_df = load_data()

    if not data_df.empty:
        # 2. Calculate Indicators
        data_df = calculate_emas(data_df)

        # 3. Run Strategy
        trade_results_df = run_backtest(data_df)

        # 4. Analyze Results
        analyze_performance(trade_results_df)

    print("\n--- END OF BACKTEST ---")
