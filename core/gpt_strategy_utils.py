# core/gpt_strategy_utils.py

import pandas as pd
import numpy as np

from core.options_directional_utils import fetch_price_data


def _rsi(series: pd.Series, period: int = 5) -> pd.Series:
    """Standard RSI calculation, default period=5 for momentum bursts."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Heikin-Ashi candles and add:
      HA_Open, HA_Close, HA_High, HA_Low, HA_Bullish (True=green, False=red)
    """
    ha = df.copy()

    # Heikin-Ashi Close = (O+H+L+C)/4
    ha_close = (ha["Open"] + ha["High"] + ha["Low"] + ha["Close"]) / 4

    # Heikin-Ashi Open
    ha_open = ha["Open"].copy()
    ha_open.iloc[0] = (ha["Open"].iloc[0] + ha["Close"].iloc[0]) / 2

    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    ha_high = pd.concat([ha["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([ha["Low"], ha_open, ha_close], axis=1).min(axis=1)

    df["HA_Open"] = ha_open
    df["HA_Close"] = ha_close
    df["HA_High"] = ha_high
    df["HA_Low"] = ha_low
    df["HA_Bullish"] = df["HA_Close"] > df["HA_Open"]  # True = green

    return df


def compute_gpt_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicators for GPTStrategy:
      - EMA20, EMA40
      - RSI5
      - Heikin-Ashi candles + HA_Bullish
    """
    df = df.copy()

    # Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # EMAs
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_40"] = df["Close"].ewm(span=40, adjust=False).mean()

    # RSI 5
    df["RSI_5"] = _rsi(df["Close"], period=5)

    # Heikin-Ashi
    df = _heikin_ashi(df)

    df.dropna(inplace=True)
    return df


def evaluate_gpt_signal(latest_row_or_df):
    """
    Evaluate GPTStrategy signal from a single row or DataFrame.

    Rules:
      - Trend: EMA20 vs EMA40
      - Momentum: RSI5 zone
      - Impulse: Heikin-Ashi color

    Returns:
      overall_signal: "BUY_CALL" | "BUY_PUT" | "NO_TRADE"
      indicators: dict(name -> {bullish: bool, bearish: bool})
    """
    if isinstance(latest_row_or_df, pd.DataFrame):
        latest = latest_row_or_df.iloc[-1]
    else:
        latest = latest_row_or_df

    ema20 = float(latest["EMA_20"])
    ema40 = float(latest["EMA_40"])
    rsi5 = float(latest["RSI_5"])
    ha_bull = bool(latest["HA_Bullish"])

    indicators = {}

    # Trend
    indicators["Trend (EMA20 vs EMA40)"] = {
        "bullish": ema20 > ema40,
        "bearish": ema20 < ema40,
    }

    # RSI5 momentum band (buffer zone 45–55 is neutral)
    indicators["Momentum (RSI 5)"] = {
        "bullish": rsi5 >= 55,
        "bearish": rsi5 <= 45,
    }

    # Heikin-Ashi color
    indicators["Impulse (Heikin-Ashi)"] = {
        "bullish": ha_bull,
        "bearish": not ha_bull,
    }

    bull_score = sum(1 for v in indicators.values() if v["bullish"])
    bear_score = sum(1 for v in indicators.values() if v["bearish"])

    if bull_score >= 3 and bull_score >= bear_score + 1:
        overall = "BUY_CALL"
    elif bear_score >= 3 and bear_score >= bull_score + 1:
        overall = "BUY_PUT"
    else:
        overall = "NO_TRADE"

    return overall, indicators