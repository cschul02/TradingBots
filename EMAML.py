import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
STARTING_EQUITY = 1000.00
TRANSACTION_COST_PER_TRADE = 5.00  # $5.00 round trip
CONTRACTS_PER_TRADE = 10

# Strategy Parameters
EMA_FAST = 10
EMA_MID = 45
EMA_SLOW = 195
LOOKAHEAD_BARS = 3 # How many future bars to look at for labeling success

# FIXED POINT RISK MANAGEMENT PARAMETERS (Points on the NQ index)
STOP_LOSS_POINTS = 20.0     # Stop Loss distance (e.g., 20 points)
TAKE_PROFIT_POINTS = 50.0   # Take Profit distance (e.g., 50 points)

# Profit Trailing Logic
MOVE_SL_PROFIT_POINTS = TAKE_PROFIT_POINTS / 2.0  # 25.0 points (Halfway to TP)
NEW_SL_PROFIT_POINTS = TAKE_PROFIT_POINTS * 0.25   # 12.5 points (Locking in 25% of TP)

# General Risk Management
MAX_HOLDING_DAYS = 1        # Maximum number of days a trade can be held
MAX_DAILY_TRADES = 2        # Maximum number of entries per calendar day

# ML Training Configuration
TRAIN_TEST_SPLIT_PERCENT = 0.80 # 80% of signals for training, 20% for testing/backtesting

# --- DATA FETCHING CONFIGURATION (Using yfinance) ---
TICKER = 'NQ=F'
START_DATE_API = '2025-09-17'
END_DATE_API = '2025-11-04'
INTERVAL = '15m'

# --- UTILITY FUNCTIONS ---

def calculate_emas(df):
    """Calculates Fast, Mid, and Slow EMAs and adds them to the DataFrame."""
    df[f'EMA_FAST_{EMA_FAST}'] = df['Close'].ewm(span=EMA_FAST, adjust=False).mean()
    df[f'EMA_MID_{EMA_MID}'] = df['Close'].ewm(span=EMA_MID, adjust=False).mean()
    df[f'EMA_SLOW_{EMA_SLOW}'] = df['Close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    """Generates Long/Short signals based on the EMA Stacked Crossover Strategy."""
    
    # Stacking Condition: Fast > Mid > Slow for Long / Fast < Mid < Slow for Short
    df['STACKED_LONG'] = (df[f'EMA_FAST_{EMA_FAST}'] > df[f'EMA_MID_{EMA_MID}']) & \
                         (df[f'EMA_MID_{EMA_MID}'] > df[f'EMA_SLOW_{EMA_SLOW}'])

    df['STACKED_SHORT'] = (df[f'EMA_FAST_{EMA_FAST}'] < df[f'EMA_MID_{EMA_MID}']) & \
                          (df[f'EMA_MID_{EMA_MID}'] < df[f'EMA_SLOW_{EMA_SLOW}'])
    
    # Crossover Condition: Previous bar was NOT stacked, current bar IS stacked.
    prev_stacked_long = df['STACKED_LONG'].shift(1).astype(bool).fillna(False)
    prev_stacked_short = df['STACKED_SHORT'].shift(1).astype(bool).fillna(False)

    df['CANDIDATE_LONG'] = df['STACKED_LONG'] & (~prev_stacked_long)
    df['CANDIDATE_SHORT'] = df['STACKED_SHORT'] & (~prev_stacked_short)
    
    # Final Signal: 1 for Long, -1 for Short, 0 for Hold/No Trade
    df['Signal'] = np.where(df['CANDIDATE_LONG'], 1, 
                   np.where(df['CANDIDATE_SHORT'], -1, 0))
    
    # Reset index to a simple RangeIndex for reliable iloc/loc positional indexing in labeling
    return df.reset_index(drop=True) 

def label_outcomes(df):
    """
    Labels historical signal bars (where Signal != 0) based on future PnL outcome.
    Requires df to have a RangeIndex (0, 1, 2, ...)
    """
    print(f"Labeling data based on {LOOKAHEAD_BARS}-bar lookahead...")
    
    # Explicitly ensure the index is integer-based RangeIndex before proceeding
    if not isinstance(df.index, pd.RangeIndex):
         df.reset_index(drop=True, inplace=True)
         
    df['Label'] = np.nan
    
    # Get indices where an EMA Crossover Signal occurred (these are integer positions)
    signal_indices = df[df['Signal'].abs() == 1].index
    
    for i in signal_indices:
        
        # Determine current trade details
        # We use .at[i, 'Col'] to reliably extract the scalar value based on the integer index i
        direction = int(df.at[i, 'Signal']) 
        entry_price = float(df.at[i, 'Close'])
        
        # Calculate fixed targets
        sl_target = entry_price - (STOP_LOSS_POINTS * direction)
        tp_target = entry_price + (TAKE_PROFIT_POINTS * direction)
        
        # Define the lookahead window using positional slicing (iloc)
        lookahead_window = df.iloc[i + 1 : i + 1 + LOOKAHEAD_BARS]
        
        # Check if window is empty (at end of data)
        if lookahead_window.empty:
            continue
            
        # Check for SL or TP hit in the lookahead window
        hit_sl = False
        hit_tp = False

        if direction == 1: # Long
            # Check if any Low <= SL Target (Hit SL)
            hit_sl = (lookahead_window['Low'] <= sl_target).any()
            # Check if any High >= TP Target (Hit TP)
            hit_tp = (lookahead_window['High'] >= tp_target).any()
            
        elif direction == -1: # Short
            # Check if any High >= SL Target (Hit SL)
            hit_sl = (lookahead_window['High'] >= sl_target).any()
            # Check if any Low <= TP Target (Hit TP - remember TP is a lower price for short)
            hit_tp = (lookahead_window['Low'] <= tp_target).any()
            
        # Assign Label using .loc with the single integer index i
        if hit_tp and not hit_sl:
            df.loc[i, 'Label'] = 1 # Success: Hit TP first
        elif hit_sl and not hit_tp:
            df.loc[i, 'Label'] = 0 # Failure: Hit SL first
        elif hit_tp and hit_sl:
            # Handle bars that cross both on the same bar (prefer SL for conservative analysis)
            df.loc[i, 'Label'] = 0 
            
    return df

def train_and_filter_signals(df_full):
    """Trains the ML model and uses it to filter the EMA signals."""
    
    # 1. Prepare Features (X) and Labels (y)
    df = label_outcomes(df_full.copy())
    
    # Features are the 3 EMA values
    features = [f'EMA_FAST_{EMA_FAST}', f'EMA_MID_{EMA_MID}', f'EMA_SLOW_{EMA_SLOW}']
    
    # Filter for bars that actually generated an EMA Crossover Signal AND have a valid Label
    df_trainable = df[(df['Signal'].abs() == 1) & (df['Label'].notna())].copy()
    
    if df_trainable.empty:
        print("Error: No signal bars with defined labels found. Cannot train model.")
        # Restore original Datetime index before returning
        return df_full.set_index('Datetime'), None

    X = df_trainable[features]
    y = df_trainable['Label'].astype(int)

    # 2. Simple Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - TRAIN_TEST_SPLIT_PERCENT), random_state=42, stratify=y
    )
    
    # 3. Train Model
    print(f"\nTraining RandomForestClassifier on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate on Test Set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Accuracy (on test set): {accuracy:.2f}")

    # 5. Predict on ALL Signal Bars
    
    # Initialize the ML_Prediction column to 0 BEFORE assignment
    df['ML_Prediction'] = 0 
    
    # Get the features for all bars that have a Signal
    df_predict = df[df['Signal'].abs() == 1].copy()
    
    if not df_predict.empty:
        X_all = df_predict[features]
        
        # --- ROBUST FIX FOR 'SCALAR INDEX' ERROR ---
        # Instead of using the Index object directly, use the underlying numpy array of indices (.values)
        # This ensures Pandas uses pure integers for assignment against the RangeIndex of df.
        prediction_indices = df_predict.index.values
        predictions = model.predict(X_all)
        
        df.loc[prediction_indices, 'ML_Prediction'] = predictions
        
    # Restore original Datetime index
    df = df.set_index('Datetime') 
    
    return df, model


def run_backtest(df_full):
    """
    Runs the backtest using the ML-filtered signals.
    """
    
    print("\n--- Running ML-Filtered Strategy Backtest ---")
    
    equity = STARTING_EQUITY
    trade_log = []
    in_trade = False
    entry_price = 0.0
    entry_date = None
    direction = 0
    daily_trade_count = {}
    
    df = df_full.copy()
    
    # --- Iterating over rows ---
    for index, current_bar in df.iterrows():
        # Convert index to a date object for daily counting
        try:
            # Safely attempt to convert DatetimeIndex to UTC date
            current_date = index.tz_convert('UTC').date()
        except TypeError:
            # If the index is already naive, just use the date
            current_date = index.date()
        
        # --- OPEN TRADE LOGIC ---
        if not in_trade:
            
            # Initialize daily trade counter
            if current_date not in daily_trade_count:
                daily_trade_count[current_date] = 0
            
            # Use .loc[index, column] on the DatetimeIndex df
            signal_value = int(current_bar['Signal'])
            ml_filter = int(current_bar['ML_Prediction'])

            # Check for signal, ML approval (filter == 1), and daily limit
            if (signal_value != 0) and (ml_filter == 1) and (daily_trade_count[current_date] < MAX_DAILY_TRADES):
                
                # EXECUTE TRADE (ML Filter is applied here!)
                in_trade = True
                # Ensure these are SCALARS
                entry_price = float(current_bar['Close'])
                entry_date = index
                direction = int(signal_value)
                daily_trade_count[current_date] += 1

                # Calculate initial SL/TP targets
                sl_target = entry_price - (STOP_LOSS_POINTS * direction)
                tp_target = entry_price + (TAKE_PROFIT_POINTS * direction)
                
                # Trailing Logic activation target
                profit_trail_target = entry_price + (MOVE_SL_PROFIT_POINTS * direction)
                new_sl_price = sl_target # SL will be updated/moved later

                
        # --- MANAGE & CLOSE TRADE LOGIC (Same as previous simple version) ---
        elif in_trade:
            # Ensure price values are scalar floats
            high = float(current_bar['High'])
            low = float(current_bar['Low'])
            close = float(current_bar['Close'])
            
            exit_reason = None
            exit_price = 0.0
            
            # 1. Check for Take Profit (TP) or Stop Loss (SL)
            if direction == 1: # Long Trade
                if high >= tp_target:
                    exit_reason = "Take Profit (TP)"
                    exit_price = tp_target
                elif low <= new_sl_price: # Use updated SL
                    exit_reason = "Stop Loss (SL)"
                    exit_price = new_sl_price
            elif direction == -1: # Short Trade
                if low <= tp_target:
                    exit_reason = "Take Profit (TP)"
                    exit_price = tp_target
                elif high >= new_sl_price: # Use updated SL
                    exit_reason = "Stop Loss (SL)"
                    exit_price = new_sl_price

            # 2. Check for Profit Trailing / Moving SL to Breakeven+
            if exit_reason is None:
                if direction == 1 and high >= profit_trail_target:
                    # Long: If profit target hit, move SL to lock-in profit
                    # Ensure the new SL is never below the old one
                    new_sl_price = max(new_sl_price, entry_price + NEW_SL_PROFIT_POINTS)
                elif direction == -1 and low <= profit_trail_target:
                    # Short: If profit target hit, move SL to lock-in profit
                    # Ensure the new SL is never above the old one
                    new_sl_price = min(new_sl_price, entry_price - NEW_SL_PROFIT_POINTS)

            # 3. Check for Max Holding Day Exit
            if exit_reason is None:
                # Handle potential NaT or timezone issues when calculating duration
                try:
                    duration = (index - entry_date).days
                except:
                    duration = 0 # Default if calculation fails
                    
                if duration >= MAX_HOLDING_DAYS:
                    exit_reason = "Max Hold Time"
                    exit_price = close # Exit at the closing price
                    
            # 4. Process Exit
            if exit_reason is not None:
                pnl_points = (exit_price - entry_price) * direction
                pnl_dollars = (pnl_points * CONTRACTS_PER_TRADE) - TRANSACTION_COST_PER_TRADE
                
                # Update equity
                equity += pnl_dollars
                
                trade_log.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': index,
                    'Direction': 'Long' if direction == 1 else 'Short',
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'PnL_Dollars': pnl_dollars,
                    'Exit_Reason': exit_reason,
                })
                
                # Reset trade state
                in_trade = False
                direction = 0

    return pd.DataFrame(trade_log), equity

def print_results(trade_df, final_balance, accuracy):
    """Prints the final backtest statistics."""
    
    total_trades = len(trade_df)
    total_pnl = trade_df['PnL_Dollars'].sum()
    
    print("\n[A] ML-Filtered Strategy Backtest Results:")
    print("=" * 70)
    print(f"Model Accuracy (on test set): {accuracy:.2f}")
    print(f"Filtering Strategy: RandomForest based on 3 EMAs, Labelled by {TAKE_PROFIT_POINTS}pt TP / {STOP_LOSS_POINTS}pt SL in {LOOKAHEAD_BARS} bars.")
    print("=" * 70)

    print(f"Starting Equity: ${STARTING_EQUITY:,.2f}")
    print(f"Final Balance:   ${final_balance:,.2f} {'(Profit)' if total_pnl >= 0 else '(Loss)'}")
    print(f"Total Net PnL:   ${total_pnl:,.2f}")
    print("-" * 70)

    if total_trades > 0:
        winning_trades = len(trade_df[trade_df['PnL_Dollars'] > 0])
        losing_trades = len(trade_df[trade_df['PnL_Dollars'] <= 0])
        win_rate = (winning_trades / total_trades) * 100
        avg_pnl = total_pnl / total_trades
        print(f"Total Trades Executed (Filtered): {total_trades}")
        print(f"Winning Trades:                 {winning_trades}")
        print(f"Losing Trades:                  {losing_trades}")
        print(f"Win Rate:                       {win_rate:.2f}%")
        print(f"Average PnL per Trade:          ${avg_pnl:,.2f}")
        
        # Display the trade log
        print("\n--- COMPLETE TRADE LOG (First 10 Trades) ---")
        # Ensure the Date fields are converted to the right string format
        trade_df['Entry_Date_Str'] = trade_df['Entry_Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        trade_df['Exit_Date_Str'] = trade_df['Exit_Date'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        print(trade_df[['Entry_Date_Str', 'Exit_Date_Str', 'Direction', 'Entry_Price', 'Exit_Price', 'PnL_Dollars', 'Exit_Reason']].head(10).to_string(index=False, float_format="%.2f"))

    else:
        print("No filtered trades executed after ML screening.")
    print("="*70)

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    try:
        # --- Data Fetching ---
        print(f"Fetching data for {TICKER} from {START_DATE_API} to {END_DATE_API} at {INTERVAL} interval...")
        # Reset index after download to enable simple iloc for lookahead in labeling
        df = yf.download(TICKER, start=START_DATE_API, end=END_DATE_API, interval=INTERVAL).reset_index()
        df.rename(columns={'index': 'Datetime'}, inplace=True)
        
        if df.empty:
            print("Error: DataFrame is empty. Check ticker, dates, and interval.")
        else:
            # --- 1. Prepare Data and Signals ---
            df = calculate_emas(df)
            df = generate_signals(df)
            
            # --- 2. Train Model and Filter Signals ---
            df_filtered, model = train_and_filter_signals(df)
            
            # The rest of the script (backtesting and reporting) continues only if training succeeds
            if model is None:
                # Handle case where training failed
                accuracy = 0.0 
                trade_df_results = pd.DataFrame()
                final_balance = STARTING_EQUITY
            else:
                # Recalculate test accuracy outside the function for reporting consistency
                trainable_df_for_accuracy = df_filtered[(df_filtered['Signal'].abs() == 1) & (df_filtered['Label'].notna())].copy()
                features = [f'EMA_FAST_{EMA_FAST}', f'EMA_MID_{EMA_MID}', f'EMA_SLOW_{EMA_SLOW}']
                
                if not trainable_df_for_accuracy.empty:
                    # We need to ensure the train/test split is identical to the one inside training
                    X_train_acc, X_test_acc, y_train_acc, y_test_acc = train_test_split(
                        trainable_df_for_accuracy[features], 
                        trainable_df_for_accuracy['Label'].astype(int), 
                        test_size=(1 - TRAIN_TEST_SPLIT_PERCENT), 
                        random_state=42, 
                        stratify=trainable_df_for_accuracy['Label'].astype(int)
                    )
                    y_pred_test = model.predict(X_test_acc)
                    accuracy = accuracy_score(y_test_acc, y_pred_test)
                else:
                    accuracy = 0.0
                
                # --- 3. Run Backtest ---
                trade_df_results, final_balance = run_backtest(df_filtered.copy())
            
            # --- 4. Print Results ---
            print_results(trade_df_results, final_balance, accuracy)
            
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("Please ensure you have all required libraries ('yfinance', 'pandas', 'numpy', 'sklearn') installed.")