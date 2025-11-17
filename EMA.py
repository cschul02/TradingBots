import pandas as pd
import numpy as np
import datetime
import yfinance as yf # New library for fetching real historical data

# --- CONFIGURATION ---
STARTING_EQUITY = 1000.00
TRANSACTION_COST_PER_TRADE = 5.00  # $5.00 round trip
CONTRACTS_PER_TRADE = 10

# Strategy Parameters (Fixed)
EMA_FAST = 13
EMA_MID = 48
EMA_SLOW = 200

# --- DATA FETCHING CONFIGURATION (Using yfinance) ---
# NOTE: QQQ is used as a liquid proxy for the NASDAQ 100 index (NQ futures underlying).
# NQ futures data usually requires a paid API subscription.
TICKER = 'NQ=F'  # Yahoo Finance ticker for E-mini NASDAQ 100 futures
# Using a historical period that includes your requested date for demonstration
START_DATE_API = '2024-10-01' 
END_DATE_API = '2025-01-01'
# Fetching 1-day interval data for simplicity with yfinance, 
# as high-frequency futures data is often restricted.
INTERVAL = '1d' 

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
        
        # FIX: Force all selected columns to lowercase for stable iteration with itertuples()
        # This remains in place to ensure consistency for all column access
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower).copy()
        
    except Exception as e:
        print(f"FATAL API ERROR: Could not fetch data using yfinance. Check your setup. Error: {e}")
        return pd.DataFrame()


# --- INDICATOR CALCULATION ---

def calculate_emas(df):
    """Calculates the 13, 48, and 200 EMAs and adds them to the DataFrame."""
    if df.empty:
        return df
        
    # Column names are already lowercase ('close') from load_data
    df['ema13'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema48'] = df['close'].ewm(span=EMA_MID, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    # Drop initial rows where EMAs are invalid
    df.dropna(inplace=True) 
    return df


# --- CORE BACKTEST LOGIC ---

def run_backtest(df):
    """Runs the pure EMA Crossover backtest strategy using iterrows for guaranteed dictionary access."""
    
    trades = []
    position = 0  # 0: Flat, 1: Long
    entry_price = 0
    entry_date = None
    
    # FIX: Using iterrows() to avoid the attribute naming conflict from itertuples()
    for index, row in df.iterrows():
        
        # FIX: Extract the scalar value using .item() to silence the FutureWarning 
        # (Calling float on a single element Series is deprecated)
        ema13 = row['ema13'].item()
        ema48 = row['ema48'].item()
        ema200 = row['ema200'].item()
        current_close = row['close'].item()
        
        # Check for ENTRY (Bullish Stack: 13 > 48 > 200)
        is_long_entry = (ema13 > ema48) and (ema48 > ema200)
        
        # Check for EXIT (Bearish Stack: 200 > 48 > 13)
        # Note: If trading Long Only, this is the signal to close the long position.
        is_long_exit = (ema200 > ema48) and (ema48 > ema13)
        
        
        if position == 0:
            # Try to Enter Long
            if is_long_entry:
                position = 1
                entry_price = current_close
                entry_date = index   # The index (date) is the first element from iterrows
                
        elif position == 1:
            # Try to Exit Long (Pure Crossover Exit)
            if is_long_exit:
                exit_price = current_close
                exit_date = index
                
                # Since this is daily data, duration is in days
                duration_days = (exit_date - entry_date).days
                
                pnl_per_contract = exit_price - entry_price
                pnl_gross = pnl_per_contract * CONTRACTS_PER_TRADE
                pnl_net = pnl_gross - TRANSACTION_COST_PER_TRADE
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': exit_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'PnL_Dollars': pnl_net,
                    'Duration_Days': int(duration_days)
                })
                
                # Reset position
                position = 0
                entry_price = 0
                entry_date = None
                
    # Handle final open position (force close at the last available bar)
    if position == 1:
        last_row = df.iloc[-1]
        exit_price = last_row['close'].item() # Accessing via dictionary key
        exit_date = last_row.name

        duration_days = (exit_date - entry_date).days
        pnl_per_contract = exit_price - entry_price
        pnl_gross = pnl_per_contract * CONTRACTS_PER_TRADE
        pnl_net = pnl_gross - (TRANSACTION_COST_PER_TRADE / 2) # Only half cost for open trade

        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': exit_date,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'PnL_Dollars': pnl_net,
            'Duration_Days': int(duration_days),
            'Status': 'Open'
        })

    return pd.DataFrame(trades)


# --- PERFORMANCE ANALYSIS AND REPORTING ---

def analyze_performance(trade_df):
    """Calculates and prints key performance metrics."""
    
    if trade_df.empty:
        print("\n--- PERFORMANCE SUMMARY ---")
        print("No trades executed.")
        return

    # FIX: Explicitly cast the PnL column to a numeric type to ensure comparisons work correctly
    # This prevents the "ValueError: The truth value of a Series is ambiguous" crash.
    trade_df['PnL_Dollars'] = pd.to_numeric(trade_df['PnL_Dollars'], errors='coerce')

    total_trades = len(trade_df)
    total_pnl = trade_df['PnL_Dollars'].sum()
    winning_trades = len(trade_df[trade_df['PnL_Dollars'] > 0])
    losing_trades = total_trades - winning_trades
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    
    final_balance = STARTING_EQUITY + total_pnl
    
    # Format for clean output
    print("\n" + "="*50)
    print("        TRIPLE EMA CROSSOVER BACKTEST REPORT")
    print("="*50)
    print(f"Data Source: yfinance API ({TICKER} - {INTERVAL} bars)")
    print(f"Period Tested: {START_DATE_API} to {END_DATE_API}")
    print(f"Strategy: Long only (13>48>200 Entry, 200>48>13 Exit)")
    print("-" * 50)
    
    print(f"Starting Equity: ${STARTING_EQUITY:,.2f}")
    print(f"Final Balance:   ${final_balance:,.2f} {'(Profit)' if total_pnl >= 0 else '(Loss)'}")
    print(f"Total Net PnL:   ${total_pnl:,.2f}")
    print("-" * 50)
    
    print(f"Total Trades Executed: {total_trades}")
    print(f"Winning Trades:        {winning_trades}")
    print(f"Losing Trades:         {losing_trades}")
    print(f"Win Rate:              {win_rate:.2f}%")
    print(f"Average PnL per Trade: ${avg_pnl:,.2f}")
    print("="*50)
    
    print("\n--- TRADE LOG (First 10 Trades) ---")
    print(trade_df.head(10).to_string(index=False))


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
    
