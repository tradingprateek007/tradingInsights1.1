import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


def fetch_weekly_insider_sentiment(ticker: str):
    """
    Dummy function simulating 4 weeks of insider sentiment data.
    Replace this with real API or database access logic.
    """
    today = datetime.today()
    data = []
    for i in range(4):
        week_ending = today - timedelta(weeks=3 - i)
        data.append({
            "week": f"Week Ending {week_ending.strftime('%Y-%m-%d')}",
            "shares_bought": 100_000 + i * 5_000,
            "shares_sold": 80_000 + i * 3_000,
        })
    return pd.DataFrame(data)


def render_insider_sentiment_tab():
    st.header("🕵️ Insider Sentiment (Last 4 Weeks)")

    ticker = st.text_input("Enter Ticker Symbol", value="AAPL", key="insider_ticker").strip().upper()

    if st.button("Fetch Insider Sentiment", key="insider_button"):
        if not ticker:
            st.warning("⚠️ Please enter a valid ticker.")
            return

        df = fetch_weekly_insider_sentiment(ticker)
        st.write(f"📊 Insider Activity for: **{ticker}**")

        st.dataframe(df)

        st.bar_chart(df.set_index("week")[["shares_bought", "shares_sold"]])