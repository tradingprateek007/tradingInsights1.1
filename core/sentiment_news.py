import requests
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob
import plotly.express as px
import streamlit as st


def fetch_finviz_news(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    res = requests.get(url, headers=headers, timeout=10)
    if res.status_code != 200:
        raise ValueError("Could not fetch data from Finviz (status code {})".format(res.status_code))

    soup = BeautifulSoup(res.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")

    if news_table is None:
        raise ValueError("News table not found on Finviz. Page structure might have changed.")

    rows = news_table.find_all("tr")
    if not rows:
        raise ValueError("No news rows found for this ticker.")

    headlines = []
    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 2:
            continue  # skip malformed row
        timestamp = tds[0].text.strip()
        title = tds[1].text.strip()
        headlines.append({
            "datetime": timestamp,
            "headline": title
        })

    return pd.DataFrame(headlines)


def analyze_sentiment(df):
    df["polarity"] = df["headline"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment"] = df["polarity"].apply(
        lambda x: "Bullish" if x > 0.1 else "Bearish" if x < -0.1 else "Neutral"
    )
    return df


def render_sentiment_tab():
    st.title("📈 Market News Sentiment")
    ticker = st.text_input("Enter ticker symbol for sentiment", "AAPL").strip().upper()

    if not ticker:
        return

    try:
        df = fetch_finviz_news(ticker)
        df = analyze_sentiment(df)

        st.subheader("Recent Headlines & Sentiment")
        st.dataframe(df[["datetime", "headline", "sentiment"]])

        # Pie chart
        sentiment_counts = df["sentiment"].value_counts()
        fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Distribution",
            color_discrete_sequence=["green", "red", "gray"]
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to fetch sentiment: {e}")