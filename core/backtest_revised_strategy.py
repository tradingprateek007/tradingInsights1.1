import pandas as pd
from typing import Literal, Tuple

import streamlit as st

from core.options_directional_utils import (
    fetch_price_data,
    compute_indicators,
    evaluate_signal,
)

SignalType = Literal["BUY_CALL", "BUY_PUT", "NO_TRADE"]


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a price+indicator DataFrame, compute the signal for each row.
    Uses your existing evaluate_signal() logic from options_directional_utils.
    """
    signals = []
    indicator_snapshots = []

    for _, row in df.iterrows():
        overall, indicators = evaluate_signal(row)
        signals.append(overall)
        indicator_snapshots.append(indicators)

    df = df.copy()
    df["signal"] = signals
    df["indicators_snapshot"] = indicator_snapshots  # optional for debugging
    return df


def backtest_directional_strategy(
    ticker: str = "NVDA",
    period: str = "5y",
    max_hold_bars: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtests your revised directional strategy on the underlying (NVDA by default).

    - Uses your indicator-based signal (BUY_CALL / BUY_PUT / NO_TRADE)
    - Enters at next day's open
    - Exits when:
        a) opposite signal appears, or
        b) max_hold_bars is reached

    Returns:
        df_signals: full OHLCV+indicators+signal DataFrame
        trades: trades DataFrame with PnL stats
    """

    # 1. Fetch price data & indicators
    df = fetch_price_data(ticker, period=period, interval="1d")
    df = compute_indicators(df)

    if df.empty:
        raise ValueError("No data after indicator computation.")

    # 2. Generate signals row by row
    df = generate_signals(df)

    # Ensure we have an 'Open' column for entry price
    if "Open" not in df.columns:
        raise ValueError("Expected 'Open' column in price data.")

    # 3. Build trades
    trades = []
    position_side: Literal["LONG", "SHORT", "FLAT"] = "FLAT"
    entry_price = None
    entry_index = None
    entry_date = None
    entry_signal: SignalType = "NO_TRADE"

    df_iter = df.reset_index()  # so Date is a column
    if "Date" not in df_iter.columns:
        # When downloaded via yfinance, index is usually DatetimeIndex named 'Date'
        df_iter = df_iter.rename(columns={df_iter.columns[0]: "Date"})

    for i in range(len(df_iter) - 1):
        row = df_iter.iloc[i]
        next_row = df_iter.iloc[i + 1]

        signal: SignalType = row["signal"]

        # If no position open, check for entry
        if position_side == "FLAT":
            if signal == "BUY_CALL":
                position_side = "LONG"
                entry_price = float(next_row["Open"])  # enter at next open
                entry_index = i + 1
                entry_date = next_row["Date"]
                entry_signal = signal
            elif signal == "BUY_PUT":
                position_side = "SHORT"
                entry_price = float(next_row["Open"])
                entry_index = i + 1
                entry_date = next_row["Date"]
                entry_signal = signal

        else:
            # We have an open position; check exit conditions
            bars_held = (i - entry_index) + 1  # number of bars since entry
            exit = False
            reason = ""

            # Condition 1: opposite signal
            if position_side == "LONG" and signal == "BUY_PUT":
                exit = True
                reason = "Opposite signal (PUT)"
            elif position_side == "SHORT" and signal == "BUY_CALL":
                exit = True
                reason = "Opposite signal (CALL)"

            # Condition 2: time stop
            if not exit and bars_held >= max_hold_bars:
                exit = True
                reason = f"Time stop ({max_hold_bars} bars)"

            if exit:
                exit_price = float(next_row["Open"])  # exit at next open
                exit_date = next_row["Date"]

                if position_side == "LONG":
                    pnl = exit_price - entry_price
                    rtn = pnl / entry_price
                else:  # SHORT
                    pnl = entry_price - exit_price
                    rtn = pnl / entry_price

                trades.append(
                    {
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "side": position_side,
                        "reason": reason,
                        "signal_at_entry": entry_signal,
                        "bars_held": bars_held,
                        "pnl": pnl,
                        "return_pct": rtn * 100.0,
                    }
                )

                # Reset position
                position_side = "FLAT"
                entry_price = None
                entry_index = None
                entry_date = None
                entry_signal = "NO_TRADE"

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
        trades_df["cum_return_pct"] = trades_df["return_pct"].cumsum()

    return df, trades_df


def summarize_trades_stats(trades: pd.DataFrame) -> dict:
    """
    Compute stats from trades df and return as dict.
    """
    if trades.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown": 0.0,
        }

    total_trades = len(trades)
    wins = (trades["pnl"] > 0).sum()
    losses = (trades["pnl"] <= 0).sum()
    win_rate = wins / total_trades * 100.0

    avg_win = trades.loc[trades["pnl"] > 0, "return_pct"].mean() if wins > 0 else 0.0
    avg_loss = trades.loc[trades["pnl"] <= 0, "return_pct"].mean() if losses > 0 else 0.0
    expectancy = 0.0
    if total_trades > 0:
        expectancy = (wins / total_trades) * avg_win + (losses / total_trades) * avg_loss

    total_return_pct = trades["return_pct"].sum()
    max_drawdown = 0.0
    if "cum_return_pct" in trades.columns:
        equity = trades["cum_return_pct"]
        dd = equity.cummax() - equity
        max_drawdown = dd.max()

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "total_return_pct": total_return_pct,
        "max_drawdown": max_drawdown,
    }


# ---------------- Streamlit Tab Renderer ---------------- #


def render_revised_strategy_backtest_tab():
    st.header("📈 Backtest – Revised Directional Strategy")

    st.markdown(
        """
This backtest simulates your **revised options strategy** on the underlying stock:

- **BUY CALL** signal → go **long** the stock  
- **BUY PUT** signal → go **short** the stock  
- **Entry:** next day's open after a signal  
- **Exit:** opposite signal, or after N bars (time stop)  

This shows whether your **directional edge** is positive over time.
"""
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value="NVDA", key="rev_bt_ticker").strip().upper()
    with col2:
        period = st.selectbox(
            "Lookback period",
            ["1y", "3y", "5y", "max"],
            index=2,
            key="rev_bt_period",
        )
    with col3:
        max_hold_bars = st.slider(
            "Max holding bars (days)",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            key="rev_bt_max_hold",
        )

    if st.button("Run Backtest", key="rev_bt_run"):
        if not ticker:
            st.warning("Please enter a ticker.")
            return

        try:
            df_signals, trades = backtest_directional_strategy(
                ticker=ticker,
                period=period,
                max_hold_bars=max_hold_bars,
            )
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            return

        if trades.empty:
            st.info("No trades were generated by the strategy for this configuration.")
            return

        stats = summarize_trades_stats(trades)

        # --- Summary Metrics ---
        st.subheader("Summary Stats")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Trades", stats["total_trades"])
            st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        with m2:
            st.metric("Avg Win", f"{stats['avg_win']:.2f}%")
            st.metric("Avg Loss", f"{stats['avg_loss']:.2f}%")
        with m3:
            st.metric("Expectancy / Trade", f"{stats['expectancy']:.2f}%")
            st.metric("Total Return", f"{stats['total_return_pct']:.2f}%")

        st.metric("Max Drawdown (trade-to-trade, %)", f"{stats['max_drawdown']:.2f}%")

        # --- Equity Curve ---
        st.subheader("Equity Curve (Cumulative Return %)")

        eq_df = trades[["exit_date", "cum_return_pct"]].set_index("exit_date")
        st.line_chart(eq_df)

        # --- Trades Table ---
        st.subheader("Trades")
        st.dataframe(
            trades[
                [
                    "entry_date",
                    "exit_date",
                    "side",
                    "reason",
                    "entry_price",
                    "exit_price",
                    "bars_held",
                    "return_pct",
                    "cum_return_pct",
                ]
            ],
            use_container_width=True,
        )


if __name__ == "__main__":
    # CLI test: run 5y NVDA backtest
    df_sig, trades_df = backtest_directional_strategy("NVDA", period="5y", max_hold_bars=10)
    print(summarize_trades_stats(trades_df))
    print(trades_df.tail())