import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import ta

STARTING_EQUITY = 10000.00
TRANSACTION_COST_PER_TRADE = 10.00
CONTRACTS_PER_TRADE = 20

EMA_FAST = 12
EMA_MID = 45
EMA_SLOW = 190
LOOKAHEAD_BARS = 10

STOP_LOSS_POINTS = 50.0
TAKE_PROFIT_POINTS = 100.0
MOVE_SL_PROFIT_POINTS = TAKE_PROFIT_POINTS / 2.0
NEW_SL_PROFIT_POINTS = TAKE_PROFIT_POINTS * 0.25

MAX_HOLDING_DAYS = 1
MAX_DAILY_TRADES = 5

TICKER = 'NQ=F'
END_DATE_API = datetime.now()
START_DATE_API = END_DATE_API - timedelta(days=50)
INTERVAL = '2m'

WALK_FORWARD_FOLDS = 20
SAVE_MODEL = True
MODEL_PATH = 'best_trading_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'


def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators for feature engineering."""
    print("Calculating technical indicators...")
    
    df = df.copy()
    
    df[f'EMA_FAST_{EMA_FAST}'] = df['Close'].ewm(span=EMA_FAST, adjust=False).mean()
    df[f'EMA_MID_{EMA_MID}'] = df['Close'].ewm(span=EMA_MID, adjust=False).mean()
    df[f'EMA_SLOW_{EMA_SLOW}'] = df['Close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['RSI_7'] = ta.momentum.RSIIndicator(close=df['Close'], window=7).rsi()
    
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
    df['BB_Position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
    
    df['ATR_14'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    
    if 'Volume' in df.columns:
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-10)
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    else:
        df['Volume_Ratio'] = 1.0
        df['OBV'] = 0.0
    
    df['Price_ROC_5'] = df['Close'].pct_change(5) * 100
    df['Price_ROC_10'] = df['Close'].pct_change(10) * 100
    df['Price_ROC_20'] = df['Close'].pct_change(20) * 100
    
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Open_Diff'] = (df['Close'] - df['Open']) / df['Open']
    
    df['EMA_Fast_Slope'] = df[f'EMA_FAST_{EMA_FAST}'].pct_change(3) * 100
    df['EMA_Mid_Slope'] = df[f'EMA_MID_{EMA_MID}'].pct_change(3) * 100
    df['EMA_Slow_Slope'] = df[f'EMA_SLOW_{EMA_SLOW}'].pct_change(3) * 100
    
    df['EMA_Fast_Mid_Diff'] = (df[f'EMA_FAST_{EMA_FAST}'] - df[f'EMA_MID_{EMA_MID}']) / df['Close']
    df['EMA_Mid_Slow_Diff'] = (df[f'EMA_MID_{EMA_MID}'] - df[f'EMA_SLOW_{EMA_SLOW}']) / df['Close']
    
    df['Price_vs_EMA_Fast'] = (df['Close'] - df[f'EMA_FAST_{EMA_FAST}']) / df[f'EMA_FAST_{EMA_FAST}']
    df['Price_vs_EMA_Mid'] = (df['Close'] - df[f'EMA_MID_{EMA_MID}']) / df[f'EMA_MID_{EMA_MID}']
    df['Price_vs_EMA_Slow'] = (df['Close'] - df[f'EMA_SLOW_{EMA_SLOW}']) / df[f'EMA_SLOW_{EMA_SLOW}']
    
    for period in [5, 10, 20]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'Price_vs_SMA_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']
    
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    df.dropna(inplace=True)
    
    return df


def generate_signals(df):
    """Generate Long/Short signals based on EMA Stacked Crossover Strategy."""
    df = df.copy()
    
    df['STACKED_LONG'] = (df[f'EMA_FAST_{EMA_FAST}'] > df[f'EMA_MID_{EMA_MID}']) & \
                         (df[f'EMA_MID_{EMA_MID}'] > df[f'EMA_SLOW_{EMA_SLOW}'])
    
    df['STACKED_SHORT'] = (df[f'EMA_FAST_{EMA_FAST}'] < df[f'EMA_MID_{EMA_MID}']) & \
                          (df[f'EMA_MID_{EMA_MID}'] < df[f'EMA_SLOW_{EMA_SLOW}'])
    
    prev_stacked_long = df['STACKED_LONG'].shift(1).astype(bool).fillna(False)
    prev_stacked_short = df['STACKED_SHORT'].shift(1).astype(bool).fillna(False)
    
    df['CANDIDATE_LONG'] = df['STACKED_LONG'] & (~prev_stacked_long)
    df['CANDIDATE_SHORT'] = df['STACKED_SHORT'] & (~prev_stacked_short)
    
    df['Signal'] = np.where(df['CANDIDATE_LONG'], 1, 
                   np.where(df['CANDIDATE_SHORT'], -1, 0)) ##Supposed to be -1,
    
    return df.reset_index(drop=True)


def label_outcomes_improved(df):
    """Enhanced labeling with realistic trade simulation."""
    print(f"Labeling trades with {LOOKAHEAD_BARS}-bar lookahead and realistic SL/TP logic...")
    
    if not isinstance(df.index, pd.RangeIndex):
        df.reset_index(drop=True, inplace=True)
    
    df['Label'] = np.nan
    df['Label_PnL'] = np.nan
    df['Label_Bars_Held'] = np.nan
    
    signal_indices = df[df['Signal'].abs() == 1].index
    
    for i in signal_indices:
        direction = int(df.at[i, 'Signal'])
        entry_price = float(df.at[i, 'Close'])
        
        sl_target = entry_price - (STOP_LOSS_POINTS * direction)
        tp_target = entry_price + (TAKE_PROFIT_POINTS * direction)
        
        lookahead_window = df.iloc[i + 1 : i + 1 + LOOKAHEAD_BARS]
        
        if lookahead_window.empty:
            continue
        
        exit_bar = None
        exit_price = None
        exit_reason = None
        
        for idx, (j, bar) in enumerate(lookahead_window.iterrows(), 1):
            high = float(bar['High'])
            low = float(bar['Low'])
            close = float(bar['Close'])
            
            if direction == 1:
                if low <= sl_target:
                    exit_bar = idx
                    exit_price = sl_target
                    exit_reason = 'SL'
                    break
                elif high >= tp_target:
                    exit_bar = idx
                    exit_price = tp_target
                    exit_reason = 'TP'
                    break
            elif direction == -1:
                if high >= sl_target:
                    exit_bar = idx
                    exit_price = sl_target
                    exit_reason = 'SL'
                    break
                elif low <= tp_target:
                    exit_bar = idx
                    exit_price = tp_target
                    exit_reason = 'TP'
                    break
        
        if exit_bar is None:
            last_bar = lookahead_window.iloc[-1]
            exit_price = float(last_bar['Close'])
            exit_bar = len(lookahead_window)
            exit_reason = 'TIMEOUT'
        
        pnl_points = (exit_price - entry_price) * direction
        pnl_dollars = (pnl_points * CONTRACTS_PER_TRADE) - TRANSACTION_COST_PER_TRADE
        
        df.loc[i, 'Label'] = 1 if pnl_dollars > 0 else 0
        df.loc[i, 'Label_PnL'] = pnl_dollars
        df.loc[i, 'Label_Bars_Held'] = exit_bar
    
    return df


def get_feature_columns():
    """Define which columns to use as features for ML models."""
    features = [
        f'EMA_FAST_{EMA_FAST}', f'EMA_MID_{EMA_MID}', f'EMA_SLOW_{EMA_SLOW}',
        'RSI_14', 'RSI_7',
        'MACD', 'MACD_Signal', 'MACD_Diff',
        'BB_Width', 'BB_Position',
        'ATR_14',
        'Stoch_K', 'Stoch_D',
        'ADX',
        'Volume_Ratio',
        'Price_ROC_5', 'Price_ROC_10', 'Price_ROC_20',
        'High_Low_Range', 'Close_Open_Diff',
        'EMA_Fast_Slope', 'EMA_Mid_Slope', 'EMA_Slow_Slope',
        'EMA_Fast_Mid_Diff', 'EMA_Mid_Slow_Diff',
        'Price_vs_EMA_Fast', 'Price_vs_EMA_Mid', 'Price_vs_EMA_Slow',
        'Price_vs_SMA_5', 'Price_vs_SMA_10', 'Price_vs_SMA_20',
        'Hour_Sin', 'Hour_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos'
    ]
    return features


def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple ML models."""
    print("\n" + "="*70)
    print("TRAINING MULTIPLE ML MODELS")
    print("="*70)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n[{name}]")
        print(f"Training on {len(X_train)} samples...")
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1 Score:       {f1:.4f}")
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        trained_models[name] = model
    
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "="*70)
    print(f"BEST MODEL: {best_model_name} (F1 Score: {results[best_model_name]['f1']:.4f})")
    print("="*70)
    
    return best_model, best_model_name, results


def analyze_feature_importance(model, feature_names, model_name, top_n=20):
    """Analyze and visualize feature importance."""
    print(f"\n[Feature Importance Analysis for {model_name}]")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(feature_importance_df.head(top_n).to_string(index=False))
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png', dpi=150)
        print(f"Saved feature importance plot to 'feature_importance_{model_name.replace(' ', '_')}.png'")
        plt.close()
        
        return feature_importance_df
    else:
        print("Model does not support feature importance analysis")
        return None


def walk_forward_validation(df_full, features):
    """Implement walk-forward validation to prevent overfitting."""
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION")
    print("="*70)
    
    df = label_outcomes_improved(df_full.copy())
    
    df_trainable = df[(df['Signal'].abs() == 1) & (df['Label'].notna())].copy()
    
    if df_trainable.empty:
        print("Error: No labeled signal data available for validation")
        return None
    
    X = df_trainable[features].values
    y = df_trainable['Label'].astype(int).values
    
    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_FOLDS)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{WALK_FORWARD_FOLDS}")
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train_scaled, y_train_fold)
        
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        rec = recall_score(y_test_fold, y_pred, zero_division=0)
        f1 = f1_score(y_test_fold, y_pred, zero_division=0)
        
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        fold_results.append({
            'fold': fold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    
    results_df = pd.DataFrame(fold_results)
    print("\n" + "="*70)
    print("Walk-Forward Validation Summary:")
    print("="*70)
    print(f"Average Accuracy:  {results_df['accuracy'].mean():.4f} (+/- {results_df['accuracy'].std():.4f})")
    print(f"Average Precision: {results_df['precision'].mean():.4f} (+/- {results_df['precision'].std():.4f})")
    print(f"Average Recall:    {results_df['recall'].mean():.4f} (+/- {results_df['recall'].std():.4f})")
    print(f"Average F1 Score:  {results_df['f1'].mean():.4f} (+/- {results_df['f1'].std():.4f})")
    print("="*70)
    
    return results_df


def train_final_model(df_full):
    """Train the final ML model with all improvements."""
    print("\n" + "="*70)
    print("FINAL MODEL TRAINING")
    print("="*70)
    
    df = label_outcomes_improved(df_full.copy())
    
    features = get_feature_columns()
    
    df_trainable = df[(df['Signal'].abs() == 1) & (df['Label'].notna())].copy()
    
    if df_trainable.empty:
        print("Error: No signal bars with defined labels found. Cannot train model.")
        return df_full.set_index('Datetime'), None, None, None
    
    print(f"\nTotal labeled signals: {len(df_trainable)}")
    print(f"Positive labels (profitable): {(df_trainable['Label'] == 1).sum()}")
    print(f"Negative labels (unprofitable): {(df_trainable['Label'] == 0).sum()}")
    
    X = df_trainable[features].values
    y = df_trainable['Label'].astype(int).values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    best_model, best_model_name, all_results = train_multiple_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    feature_importance_df = analyze_feature_importance(best_model, features, best_model_name)
    
    print("\n[Applying Predictions to Out-of-Sample Data Only]")
    df['ML_Prediction'] = 0
    df['In_Sample'] = True
    
    trainable_indices = df_trainable.index.values
    test_start_idx = trainable_indices[split_idx]
    
    df_signal_locs = df[df['Signal'].abs() == 1].index.values
    test_signal_mask = df_signal_locs >= test_start_idx
    test_signal_indices = df_signal_locs[test_signal_mask]
    
    df.loc[test_signal_indices, 'In_Sample'] = False
    
    if len(test_signal_indices) > 0:
        X_test_pred = df.loc[test_signal_indices, features].values
        X_test_pred_scaled = scaler.transform(X_test_pred)
        predictions = best_model.predict(X_test_pred_scaled)
        df.loc[test_signal_indices, 'ML_Prediction'] = predictions
        print(f"Predictions made on {len(test_signal_indices)} out-of-sample signals")
    else:
        print("Warning: No out-of-sample signals found for prediction")
    
    df = df.set_index('Datetime')
    
    if SAVE_MODEL:
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"\nModel saved to '{MODEL_PATH}'")
        print(f"Scaler saved to '{SCALER_PATH}'")
    
    return df, best_model, best_model_name, all_results


def calculate_advanced_metrics(trade_df, equity_curve, starting_equity):
    """Calculate advanced trading performance metrics."""
    if len(trade_df) == 0:
        return {}
    
    returns = equity_curve.pct_change().dropna()
    
    total_trades = len(trade_df)
    winning_trades = len(trade_df[trade_df['PnL_Dollars'] > 0])
    losing_trades = len(trade_df[trade_df['PnL_Dollars'] <= 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = trade_df['PnL_Dollars'].sum()
    avg_win = trade_df[trade_df['PnL_Dollars'] > 0]['PnL_Dollars'].mean() if winning_trades > 0 else 0
    avg_loss = abs(trade_df[trade_df['PnL_Dollars'] <= 0]['PnL_Dollars'].mean()) if losing_trades > 0 else 0
    
    profit_factor = (trade_df[trade_df['PnL_Dollars'] > 0]['PnL_Dollars'].sum() / 
                    abs(trade_df[trade_df['PnL_Dollars'] <= 0]['PnL_Dollars'].sum())) if losing_trades > 0 else float('inf')
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    if len(returns) > 0 and returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24)
    else:
        sharpe_ratio = 0
    
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else starting_equity
    total_return = ((final_equity - starting_equity) / starting_equity) * 100
    
    metrics = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'final_equity': final_equity
    }
    
    return metrics


def run_backtest_enhanced(df_full):
    """Enhanced backtest with detailed tracking (out-of-sample only)."""
    print("\n" + "="*70)
    print("ENHANCED ML-FILTERED STRATEGY BACKTEST (OUT-OF-SAMPLE ONLY)")
    print("="*70)
    
    equity = STARTING_EQUITY
    trade_log = []
    equity_curve = [STARTING_EQUITY]
    equity_dates = []
    in_trade = False
    entry_price = 0.0
    entry_date = None
    direction = 0
    daily_trade_count = {}
    
    df = df_full.copy()
    
    oos_count = len(df[df.get('In_Sample', True) == False])
    print(f"Processing {oos_count} out-of-sample bars to prevent data leakage...")
    
    for index, current_bar in df.iterrows():
        try:
            current_date = index.tz_convert('UTC').date() if hasattr(index, 'tz_convert') else index.date()
        except:
            current_date = index.date()
        
        if not in_trade:
            if current_date not in daily_trade_count:
                daily_trade_count[current_date] = 0
            
            signal_value = int(current_bar['Signal'])
            ml_filter = int(current_bar['ML_Prediction'])
            is_oos = not bool(current_bar.get('In_Sample', True))
            
            if (signal_value != 0) and (ml_filter == 1) and is_oos and (daily_trade_count[current_date] < MAX_DAILY_TRADES):
                in_trade = True
                entry_price = float(current_bar['Close'])
                entry_date = index
                direction = int(signal_value)
                daily_trade_count[current_date] += 1
                
                sl_target = entry_price - (STOP_LOSS_POINTS * direction)
                tp_target = entry_price + (TAKE_PROFIT_POINTS * direction)
                profit_trail_target = entry_price + (MOVE_SL_PROFIT_POINTS * direction)
                new_sl_price = sl_target
        
        elif in_trade:
            high = float(current_bar['High'])
            low = float(current_bar['Low'])
            close = float(current_bar['Close'])
            
            exit_reason = None
            exit_price = 0.0
            
            if direction == 1:
                if high >= tp_target:
                    exit_reason = "Take Profit (TP)"
                    exit_price = tp_target
                elif low <= new_sl_price:
                    exit_reason = "Stop Loss (SL)"
                    exit_price = new_sl_price
            elif direction == -1:
                if low <= tp_target:
                    exit_reason = "Take Profit (TP)"
                    exit_price = tp_target
                elif high >= new_sl_price:
                    exit_reason = "Stop Loss (SL)"
                    exit_price = new_sl_price
            
            if exit_reason is None:
                if direction == 1 and high >= profit_trail_target:
                    new_sl_price = max(new_sl_price, entry_price + NEW_SL_PROFIT_POINTS)
                elif direction == -1 and low <= profit_trail_target:
                    new_sl_price = min(new_sl_price, entry_price - NEW_SL_PROFIT_POINTS)
            
            if exit_reason is None:
                try:
                    duration = (index - entry_date).days
                except:
                    duration = 0
                
                if duration >= MAX_HOLDING_DAYS:
                    exit_reason = "Max Hold Time"
                    exit_price = close
            
            if exit_reason is not None:
                pnl_points = (exit_price - entry_price) * direction
                pnl_dollars = (pnl_points * CONTRACTS_PER_TRADE) - TRANSACTION_COST_PER_TRADE
                
                equity += pnl_dollars
                
                trade_log.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': index,
                    'Direction': 'Long' if direction == 1 else 'Short',
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'PnL_Points': pnl_points,
                    'PnL_Dollars': pnl_dollars,
                    'Exit_Reason': exit_reason,
                })
                
                equity_curve.append(equity)
                equity_dates.append(index)
                
                in_trade = False
                direction = 0
    
    trade_df = pd.DataFrame(trade_log)
    equity_series = pd.Series(equity_curve, index=[df.index[0]] + equity_dates if equity_dates else [df.index[0]])
    
    return trade_df, equity, equity_series


def visualize_equity_curve(equity_curve, trade_df):
    """Visualize the equity curve and drawdown."""
    print("\nGenerating equity curve visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    ax1.plot(equity_curve.index, equity_curve.values, linewidth=2, color='blue', label='Equity')
    ax1.axhline(y=STARTING_EQUITY, color='red', linestyle='--', alpha=0.5, label='Starting Equity')
    ax1.set_ylabel('Equity ($)')
    ax1.set_title('Equity Curve - ML-Enhanced Trading Strategy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if len(trade_df) > 0:
        winning_trades = trade_df[trade_df['PnL_Dollars'] > 0]
        losing_trades = trade_df[trade_df['PnL_Dollars'] <= 0]
        
        if len(winning_trades) > 0:
            ax1.scatter(winning_trades['Exit_Date'], 
                       [equity_curve.loc[d] for d in winning_trades['Exit_Date']], 
                       color='green', marker='^', s=100, alpha=0.6, label='Wins', zorder=5)
        
        if len(losing_trades) > 0:
            ax1.scatter(losing_trades['Exit_Date'], 
                       [equity_curve.loc[d] for d in losing_trades['Exit_Date']], 
                       color='red', marker='v', s=100, alpha=0.6, label='Losses', zorder=5)
    
    returns = equity_curve.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max * 100
    
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=2)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=150)
    print("Saved equity curve to 'equity_curve.png'")
    plt.close()


def print_results_enhanced(trade_df, metrics, model_name, all_results):
    """Print comprehensive backtest results."""
    print("\n" + "="*70)
    print("ML-ENHANCED TRADING STRATEGY - FINAL RESULTS")
    print("="*70)
    
    print(f"\nBest Model: {model_name}")
    print(f"Starting Equity: ${STARTING_EQUITY:,.2f}")
    print(f"Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Total PnL: ${metrics['total_pnl']:,.2f}")
    
    print("\n" + "-"*70)
    print("TRADING STATISTICS")
    print("-"*70)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    print("\n" + "-"*70)
    print("RISK METRICS")
    print("-"*70)
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    if len(trade_df) > 0:
        print("\n" + "-"*70)
        print("SAMPLE TRADES (First 10)")
        print("-"*70)
        display_df = trade_df.head(10).copy()
        display_df['Entry_Date'] = display_df['Entry_Date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['Exit_Date'] = display_df['Exit_Date'].dt.strftime('%Y-%m-%d %H:%M')
        print(display_df[['Entry_Date', 'Exit_Date', 'Direction', 'Entry_Price', 'Exit_Price', 'PnL_Dollars', 'Exit_Reason']].to_string(index=False))
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Test Accuracy: {result['test_acc']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1 Score: {result['f1']:.4f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        print("="*70)
        print("ENHANCED ML TRADING STRATEGY")
        print("="*70)
        print(f"Ticker: {TICKER}")
        print(f"Date Range: {START_DATE_API} to {END_DATE_API}")
        print(f"Interval: {INTERVAL}")
        print(f"Starting Equity: ${STARTING_EQUITY:,.2f}")
        print("="*70)
        
        print(f"\nFetching data for {TICKER}...")
        df = yf.download(TICKER, start=START_DATE_API, end=END_DATE_API, interval=INTERVAL)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df = df.reset_index()
        df.rename(columns={'index': 'Datetime'}, inplace=True)
        
        if df.empty:
            print("Error: DataFrame is empty. Check ticker, dates, and interval.")
        else:
            print(f"Downloaded {len(df)} bars of data")
            
            df = calculate_technical_indicators(df)
            print(f"After technical indicators: {len(df)} bars")
            
            df = generate_signals(df)
            print(f"Generated {(df['Signal'].abs() == 1).sum()} EMA crossover signals")
            
            features = get_feature_columns()
            print(f"\nTotal features for ML: {len(features)}")
            
            validation_results = walk_forward_validation(df.copy(), features)
            
            df_filtered, model, model_name, all_results = train_final_model(df)
            
            if model is None:
                print("\nTraining failed. Exiting.")
            else:
                trade_df, final_equity, equity_curve = run_backtest_enhanced(df_filtered.copy())
                
                metrics = calculate_advanced_metrics(trade_df, equity_curve, STARTING_EQUITY)
                
                if len(trade_df) > 0:
                    visualize_equity_curve(equity_curve, trade_df)
                
                print_results_enhanced(trade_df, metrics, model_name, all_results)
                
                print("\n" + "="*70)
                print("COMPLETED SUCCESSFULLY!")
                print("="*70)
                print("\nGenerated Files:")
                print("  - best_trading_model.pkl (Trained model)")
                print("  - feature_scaler.pkl (Feature scaler)")
                print("  - equity_curve.png (Equity curve visualization)")
                print(f"  - feature_importance_{model_name.replace(' ', '_')}.png (Feature importance)")
                print("="*70)
                
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
