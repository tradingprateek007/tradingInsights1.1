from core.data import fetch_history, fetch_option_data, fetch_option_chain_summary
from scipy.stats import zscore
import numpy as np


def compute_zscore(series, window=20):
    return zscore(series[-window:])[-1]


def calculate_indicators(options_data):
    results = {}
    results['avg_iv'] = options_data['impliedVolatility'].mean()
    results['high_oi'] = options_data['openInterest'].max()
    results['high_vol'] = options_data['volume'].max()
    put_vol = options_data[options_data['type'] == 'put']['volume'].sum()
    call_vol = options_data[options_data['type'] == 'call']['volume'].sum()
    results['put_call_ratio'] = put_vol / call_vol if call_vol else 0
    otm_calls = options_data[(options_data['type'] == 'call') &
                             (options_data['strike'] > options_data['strike'].median())]
    otm_puts = options_data[(options_data['type'] == 'put') &
                            (options_data['strike'] < options_data['strike'].median())]
    results['vol_skew'] = otm_puts['impliedVolatility'].mean() - otm_calls['impliedVolatility'].mean()
    return results


def generate_options_signal(indicators):
    signals = []
    if indicators['avg_iv'] < 0.3:
        signals.append("Buy (IV is low)")
    else:
        signals.append("Sell (IV is high)")

    if indicators['put_call_ratio'] > 1:
        signals.append("Contrarian Buy (High put/call)")
    else:
        signals.append("Cautious (Call-heavy)")

    if indicators['vol_skew'] > 0:
        signals.append("Bearish Skew (Put IV > Call IV)")
    else:
        signals.append("Bullish Skew (Call IV > Put IV)")
    return signals


def generate_combined_trade_signal(ticker):
    prices = fetch_history(ticker)
    z = compute_zscore(prices.pct_change().dropna())

    options_data = fetch_option_data(ticker)
    indicators = calculate_indicators(options_data)
    option_signals = generate_options_signal(indicators)

    oc = fetch_option_chain_summary(ticker)
    skew = round(oc["put_iv"] - oc["call_iv"], 3) if oc else None

    if z > 1:
        price_signal = "Buy Call"
    elif z < -1:
        price_signal = "Buy Put"
    else:
        price_signal = "Hold"

    return {
        "Ticker": ticker,
        "Z-Score": round(z, 2),
        "Price-Based Signal": price_signal,
        "IV Skew": skew,
        "Indicators": indicators,
        "Option Sentiment": option_signals
    }