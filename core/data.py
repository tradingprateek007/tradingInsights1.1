import yfinance as yf
import pandas as pd


def fetch_history(ticker, period="5y"):
    return yf.Ticker(ticker).history(period=period)["Close"]


def fetch_option_data(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        raise ValueError("No options data available.")

    data = []
    for exp in expirations[:1]:
        options = stock.option_chain(exp)
        calls = options.calls
        puts = options.puts
        calls["type"] = "call"
        puts["type"] = "put"
        merged = pd.concat([calls, puts])
        merged["expirationDate"] = exp
        data.append(merged)
    return pd.concat(data)


def fetch_option_chain_summary(ticker):
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        return None
    opt = tk.option_chain(expirations[0])
    calls, puts = opt.calls, opt.puts
    last_price = tk.history(period="1d")['Close'].iloc[-1]
    atm = round(last_price, 0)
    nearest_call = calls.iloc[(calls['strike'] - atm).abs().argsort()[:1]]
    nearest_put = puts.iloc[(puts['strike'] - atm).abs().argsort()[:1]]
    return {
        "atm_strike": atm,
        "call_price": nearest_call['lastPrice'].values[0],
        "put_price": nearest_put['lastPrice'].values[0],
        "call_iv": nearest_call['impliedVolatility'].values[0],
        "put_iv": nearest_put['impliedVolatility'].values[0],
        "expiry": expirations[0]
    }