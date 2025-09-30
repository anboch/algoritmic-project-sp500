import ta
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# --- BASIC INDICATORS ---

def add_rsi(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.DataFrame:
    # Adds the Relative Strength Index (RSI) to the DataFrame.
    df_copy = df.copy()
    rsi_indicator = ta.momentum.RSIIndicator(close=df_copy[column], window=period)
    df_copy[f'rsi_{period}'] = rsi_indicator.rsi()
    return df_copy

def add_sma(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    # Adds a Simple Moving Average (SMA) to the DataFrame.
    df_copy = df.copy()
    df_copy[f'{column}_sma_{period}'] = df_copy[column].rolling(window=period).mean()
    return df_copy

def add_kama(df: pd.DataFrame, column: str = 'close', period: int = 10) -> pd.DataFrame:
    # Adds Kaufman's Adaptive Moving Average (KAMA) to the DataFrame.
    df_copy = df.copy()
    kama_indicator = ta.momentum.KAMAIndicator(df_copy[column], period)
    df_copy[f"kama_{period}"] = kama_indicator.kama()
    return df_copy

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    # Adds the Average True Range (ATR) for volatility measurement.
    df_copy = df.copy()
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df_copy['high'],
        low=df_copy['low'],
        close=df_copy['close'],
        window=period
    )
    df_copy[f'atr_{period}'] = atr_indicator.average_true_range()
    return df_copy

# --- RETURN & VOLATILITY FEATURES ---

def add_log_return(df: pd.DataFrame, column: str = 'close', period: int = 1) -> pd.DataFrame:
    # Adds the n-period logarithmic return.
    df_copy = df.copy()
    df_copy[f"log_return_{period}"] = np.log(df_copy[column]).diff(period)
    return df_copy

def add_auto_corr(df: pd.DataFrame, column: str, period: int, lag: int) -> pd.DataFrame:
    # Adds rolling autocorrelation. Useful on returns, not prices.
    df_copy = df.copy()
    col_name = f'autocorr_{column}_n{period}_lag{lag}'
    df_copy[col_name] = df_copy[column].rolling(window=period).corr(df_copy[column].shift(lag))
    return df_copy

def add_yang_zhang_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    # Adds Yang-Zhang volatility, which accounts for gaps and intraday prices.
    # Fills initial NaNs by back-filling from the first valid calculation.
    df_copy = df.copy()
    log_ho = np.log(df_copy['high'] / df_copy['open'])
    log_lo = np.log(df_copy['low'] / df_copy['open'])
    log_co = np.log(df_copy['close'] / df_copy['open'])
    
    log_oc = np.log(df_copy['open'] / df_copy['close'].shift(1))
    log_cc = np.log(df_copy['close'] / df_copy['close'].shift(1))

    var_o = log_oc.rolling(window=window).var()
    var_c = log_cc.rolling(window=window).var()
    
    rs_component = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    var_rs = rs_component.rolling(window=window).mean()
    
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    variance_yz = var_o + k * var_c + (1 - k) * var_rs
    volatility = np.sqrt(variance_yz)
    
    df_copy[f'volatility_yz_{window}'] = volatility.bfill()
    return df_copy

# --- CANDLE ANATOMY FEATURES ---

def add_candle_info(df: pd.DataFrame) -> pd.DataFrame:
    # Adds basic candle info: direction, body-to-range ratio, and amplitude.
    df_copy = df.copy()
    
    # 1 for bullish, -1 for bearish
    df_copy["candle_way"] = np.sign(df_copy["close"] - df_copy["open"]).replace(0, -1)
    
    range_hl = df_copy["high"] - df_copy["low"]
    body_size = np.abs(df_copy["close"] - df_copy["open"])
    
    # What % of the candle's total range is the body. 1.0 = marubozu, 0.0 = doji
    df_copy["filling_ratio"] = np.where(range_hl > 0, body_size / range_hl, 0)
    
    return df_copy

def add_derivatives(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # Calculates the 'velocity' and 'acceleration' of a column.
    df_copy = df.copy()
    velocity = df_copy[column].diff()
    acceleration = velocity.diff()
    
    df_copy[f"{column}_velocity"] = velocity.fillna(0)
    df_copy[f"{column}_acceleration"] = acceleration.fillna(0)
    
    return df_copy

def add_spread_info(df: pd.DataFrame) -> pd.DataFrame:
    # Adds features related to the candle's spread (high-low range).
    df_copy = df.copy()
    spread = df_copy["high"] - df_copy["low"]
    df_copy["spread"] = spread
    
    # Compares current spread to the 20-period average. >1 is expansion, <1 is contraction.
    avg_spread = spread.rolling(window=20).mean()
    df_copy['spread_contraction'] = np.where(avg_spread > 0, spread / avg_spread, 1).fillna(1)
    return df_copy

# --- MARKET STRUCTURE FEATURES ---

def add_fair_value_gap(df: pd.DataFrame) -> pd.DataFrame:
    # Identifies Fair Value Gaps (FVG) and tracks them until they are mitigated.
    # Returns 0 if no FVG is active.
    df_copy = df.copy()
    high_prev2 = df_copy['high'].shift(2)
    low_prev2 = df_copy['low'].shift(2)

    # 1. Detect where an FVG is created
    is_bull_fvg = df_copy['low'] > high_prev2
    is_bear_fvg = df_copy['high'] < low_prev2

    # 2. Propagate the FVG boundaries forward in time
    bull_bottom = pd.Series(np.where(is_bull_fvg, high_prev2, np.nan), index=df_copy.index).ffill()
    bull_top = pd.Series(np.where(is_bull_fvg, df_copy['low'], np.nan), index=df_copy.index).ffill()
    bear_bottom = pd.Series(np.where(is_bear_fvg, df_copy['high'], np.nan), index=df_copy.index).ffill()
    bear_top = pd.Series(np.where(is_bear_fvg, low_prev2, np.nan), index=df_copy.index).ffill()

    # 3. Detect when an FVG is mitigated (invalidated)
    # We need a unique ID for each FVG to track its specific mitigation
    bull_fvg_id = is_bull_fvg.cumsum()
    bear_fvg_id = is_bear_fvg.cumsum()
    
    bull_mitigated = (df_copy['low'] <= bull_bottom).groupby(bull_fvg_id).cumsum() > 0
    bear_mitigated = (df_copy['high'] >= bear_top).groupby(bear_fvg_id).cumsum() > 0

    # 4. Nullify the FVG levels once they have been mitigated
    df_copy['bull_fvg_bottom'] = bull_bottom.where(~bull_mitigated, 0).fillna(0)
    df_copy['bull_fvg_top'] = bull_top.where(~bull_mitigated, 0).fillna(0)
    df_copy['bear_fvg_bottom'] = bear_bottom.where(~bear_mitigated, 0).fillna(0)
    df_copy['bear_fvg_top'] = bear_top.where(~bear_mitigated, 0).fillna(0)
    
    return df_copy

# --- MARKET REGIME FEATURES ---

def add_kama_regime(df: pd.DataFrame, fast_period: int = 10, slow_period: int = 50) -> pd.DataFrame:
    # Defines a market regime using two KAMAs.
    # Assumes KAMA columns already exist. Call add_kama() for both periods first.
    df_copy = df.copy()
    fast_kama_col = f"kama_{fast_period}"
    slow_kama_col = f"kama_{slow_period}"
    
    # Check if required columns exist to prevent silent errors
    if fast_kama_col not in df_copy.columns or slow_kama_col not in df_copy.columns:
        raise ValueError(
            f"Required KAMA columns ('{fast_kama_col}', '{slow_kama_col}') not found. "
            "Please call add_kama() for both periods before this function."
        )
            
    kama_diff = df_copy[fast_kama_col] - df_copy[slow_kama_col]
    
    # 1 if fast KAMA > slow KAMA (uptrend), 0 otherwise.
    df_copy[f'kama_regime_{fast_period}_{slow_period}'] = np.where(kama_diff > 0, 1, 0)
    return df_copy

def add_rolling_adf_pvalue(df: pd.DataFrame, column: str = 'close', window: int = 100, step: int = 10) -> pd.DataFrame:
    # Calculates the p-value of the Augmented Dickey-Fuller test on a rolling window.
    # A low p-value (<0.05) suggests the series is stationary (mean-reverting).
    # A high p-value suggests a trend (non-stationary).
    # Calculated sparsely (using a step) for performance reasons.
    df_copy = df.copy()
    p_values = pd.Series(index=df_copy.index, dtype=float)
    
    for i in range(window, len(df_copy), step):
        window_data = df_copy[column].iloc[i - window : i]
        # Avoid errors with null or constant data
        if window_data.isnull().any() or window_data.nunique() < 2:
            p_value = 1.0 # Assume non-stationary if data is bad
        else:
            try:
                p_value = adfuller(window_data, autolag='AIC')[1]
            except Exception:
                p_value = 1.0 # Be conservative if the test fails
        p_values.iloc[i] = p_value
            
    p_values.ffill(inplace=True)
    df_copy[f'adf_pvalue_{window}_{step}'] = p_values.fillna(1.0)
    return df_copy