import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import zscore
import plotly.graph_objects as go
from core.data import fetch_history


def detect_momentum_bursts(prices, window=5, threshold=2.5):
    returns = prices.pct_change().dropna()
    z_scores = returns.rolling(window).apply(
        lambda x: zscore(x)[-1] if len(x) == window else np.nan)
    burst_mask = z_scores.abs() > threshold
    bursts = pd.DataFrame({
        "Date": returns.index[burst_mask],
        "Z-Score": z_scores[burst_mask],
        "Direction": np.where(z_scores[burst_mask] > 0, "Bullish", "Bearish")
    })
    return bursts, z_scores


def momentum_burst_tab():
    st.header("Momentum Bursts")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    if not ticker:
        return

    prices = fetch_history(ticker)
    bursts, z_scores = detect_momentum_bursts(prices)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z_scores.index, y=z_scores, mode="lines", name="Z-Score"))
    if not bursts.empty:
        fig.add_trace(go.Scatter(
            x=bursts["Date"],
            y=bursts["Z-Score"],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Bursts"
        ))
    fig.update_layout(title="Z-Score Momentum Bursts", xaxis_title="Date", yaxis_title="Z-Score")
    st.plotly_chart(fig, use_container_width=True)

    if not bursts.empty:
        st.subheader("Detected Momentum Bursts")
        st.dataframe(bursts)

        st.markdown("### 🧠 Strategy Commentary")
        st.markdown(
            "**Momentum bursts indicate short-term overreactions. If bullish burst detected, consider short-dated Long Calls. If bearish, Long Puts. Consider using tight stops or spreads to manage risk.**")
    else:
        st.info("No significant momentum bursts detected.")