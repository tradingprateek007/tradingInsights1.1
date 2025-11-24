import yfinance as yf
import pandas as pd


def fetch_price_data(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    """
    Downloads OHLCV data for a ticker using yfinance.
    Handles cases where yfinance returns MultiIndex columns.
    """
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}.")

    # --- FIX: Flatten MultiIndex Columns ---
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.dropna(inplace=True)
    return data


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes EMA21, EMA50, RSI14, MACD, MACD signal, MACD histogram,
    and 20-day average relative volume.
    """
    df = df.copy()

    # --- Extra safety: flatten MultiIndex columns again ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure required columns exist
    if "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError(f"Data missing expected columns. Got: {df.columns}")

    # --- Volume must be Series ---
    volume = df["Volume"].astype(float)

    # EMAs
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Relative Volume
    df["Vol_20d_avg"] = volume.rolling(window=20).mean()
    df["Vol_rel"] = volume / df["Vol_20d_avg"]

    df.dropna(inplace=True)
    return df


def evaluate_signal(latest_row_or_df):
    """
    Evaluates bullish/bearish signals based on:
      - EMA trend
      - RSI momentum
      - MACD
      - Relative volume

    Accepts:
      • A single Series row
      • A DataFrame (it will use last row)
    """

    # Normalize input to a single row
    if isinstance(latest_row_or_df, pd.DataFrame):
        latest = latest_row_or_df.iloc[-1]
    else:
        latest = latest_row_or_df

    # Extract fields as floats
    price = float(latest["Close"])
    ema21 = float(latest["EMA_21"])
    ema50 = float(latest["EMA_50"])
    rsi = float(latest["RSI_14"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_signal"])
    macd_hist = float(latest["MACD_hist"])
    vol_rel = float(latest["Vol_rel"])

    indicators = {}

    # Trend rules
    indicators["Trend (Price vs EMA)"] = {
        "bullish": (price > ema21) and (ema21 > ema50),
        "bearish": (price < ema21) and (ema21 < ema50)
    }

    # RSI momentum
    indicators["Momentum (RSI 14)"] = {
        "bullish": 50 <= rsi <= 70,
        "bearish": 30 <= rsi < 50
    }

    # MACD
    indicators["MACD"] = {
        "bullish": (macd > macd_signal) and (macd_hist > 0),
        "bearish": (macd < macd_signal) and (macd_hist < 0),
    }

    # Volume confirmation
    indicators["Relative Volume"] = {
        "bullish": (vol_rel >= 1.2) and (price > ema21),
        "bearish": (vol_rel >= 1.2) and (price < ema21),
    }

    # Score signals
    bull_score = sum(1 for x in indicators.values() if x["bullish"])
    bear_score = sum(1 for x in indicators.values() if x["bearish"])

    # Decision logic
    if bull_score >= bear_score + 2 and bull_score >= 3:
        overall = "BUY_CALL"
    elif bear_score >= bull_score + 2 and bear_score >= 3:
        overall = "BUY_PUT"
    else:
        overall = "NO_TRADE"

    return overall, indicators