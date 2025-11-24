# core/gpt_strategy_backtest.py

import pandas as pd
import streamlit as st
from typing import Literal, Tuple

from core.gpt_strategy_utils import (
    fetch_price_data,
    compute_gpt_indicators,
    evaluate_gpt_signal,
)

SignalType = Literal["BUY_CALL", "BUY_PUT", "NO_TRADE"]


def _generate_gpt_signals(df: pd.DataFrame) -> pd.DataFrame:
    signals = []
    for _, row in df.iterrows():
        overall, _ = evaluate_gpt_signal(row)
        signals.append(overall)

    df = df.copy()
    df["signal"] = signals
    return df


def backtest_gpt_strategy(
    ticker: str = "AMD",
    period: str = "1y",
    max_hold_bars: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest GPTStrategy on the underlying (stock).

    - BUY_CALL -> long stock
    - BUY_PUT  -> short stock
    - Entry: next day's open
    - Exit: opposite signal OR max_hold_bars
    """

    df = fetch_price_data(ticker, period=period, interval="1d")
    df = compute_gpt_indicators(df)

    if df.empty:
        raise ValueError("No data after GPT indicator computation.")

    df = _generate_gpt_signals(df)

    if "Open" not in df.columns:
        raise ValueError("Expected 'Open' column in price data.")

    trades = []
    position_side: Literal["LONG", "SHORT", "FLAT"] = "FLAT"
    entry_price = None
    entry_index = None
    entry_date = None
    entry_signal: SignalType = "NO_TRADE"

    df_iter = df.reset_index()
    if "Date" not in df_iter.columns:
        df_iter = df_iter.rename(columns={df_iter.columns[0]: "Date"})

    for i in range(len(df_iter) - 1):
        row = df_iter.iloc[i]
        next_row = df_iter.iloc[i + 1]
        signal: SignalType = row["signal"]

        if position_side == "FLAT":
            if signal == "BUY_CALL":
                position_side = "LONG"
                entry_price = float(next_row["Open"])
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
            bars_held = (i - entry_index) + 1
            exit = False
            reason = ""

            if position_side == "LONG" and signal == "BUY_PUT":
                exit = True
                reason = "Opposite signal (PUT)"
            elif position_side == "SHORT" and signal == "BUY_CALL":
                exit = True
                reason = "Opposite signal (CALL)"

            if not exit and bars_held >= max_hold_bars:
                exit = True
                reason = f"Time stop ({max_hold_bars} bars)"

            if exit:
                exit_price = float(next_row["Open"])
                exit_date = next_row["Date"]

                if position_side == "LONG":
                    pnl = exit_price - entry_price
                    rtn = pnl / entry_price
                else:
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


def _summarize_trades(trades: pd.DataFrame) -> dict:
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
        eq = trades["cum_return_pct"]
        dd = eq.cummax() - eq
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


def render_gpt_strategy_backtest_tab():
    st.header("🤖 GPT Strategy Backtest")

    st.markdown(
        """
This backtest evaluates the **GPTStrategy** on the underlying stock:

- **BUY CALL** → long the stock  
- **BUY PUT** → short the stock  
- **Entry:** next day's open after signal  
- **Exit:** opposite signal or time stop (max bars)  
"""
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value="AMD", key="gpt_bt_ticker").strip().upper()
    with col2:
        period = st.selectbox(
            "Lookback period",
            ["1y", "3y", "5y", "max"],
            index=0,
            key="gpt_bt_period",
        )
    with col3:
        max_hold_bars = st.slider(
            "Max holding bars (days)",
            min_value=3,
            max_value=15,
            value=6,
            step=1,
            key="gpt_bt_max_hold",
        )

    if st.button("Run GPTStrategy Backtest", key="gpt_bt_run"):
        if not ticker:
            st.warning("Please enter a ticker.")
            return

        try:
            df_sig, trades = backtest_gpt_strategy(
                ticker=ticker,
                period=period,
                max_hold_bars=max_hold_bars,
            )
        except Exception as e:
            st.error(f"Error running GPTStrategy backtest: {e}")
            return

        if trades.empty:
            st.info("No trades were generated by GPTStrategy.")
            return

        stats = _summarize_trades(trades)

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

        st.metric(
            "Max Drawdown (trade-to-trade, %)",
            f"{stats['max_drawdown']:.2f}%",
        )

        st.subheader("Equity Curve (Cumulative Return %)")

        eq_df = trades[["exit_date", "cum_return_pct"]].set_index("exit_date")
        st.line_chart(eq_df)

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