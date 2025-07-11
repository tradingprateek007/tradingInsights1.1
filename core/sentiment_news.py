import streamlit as st
from textblob import TextBlob
import yfinance as yf
from bs4 import BeautifulSoup
import requests
import pandas as pd
import datetime
import plotly.graph_objects as go

def fetch_finviz_news(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(page.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")
    if news_table is None:
        return []
    rows = news_table.find_all("tr")
    news_items = []
    for row in rows:
        tds = row.find_all("td")
        if len(tds) != 2:
            continue
        dt_text = tds[0].text.strip()
        link = tds[1].a["href"]
        headline = tds[1].a.text.strip()
        # Parse date
        try:
            if ':' in dt_text:  # time only (same day)
                date = datetime.datetime.today().date()
            else:
                date = datetime.datetime.strptime(dt_text, "%b-%d-%y").date()
        except:
            date = datetime.datetime.today().date()
        news_items.append({"date": date, "headline": headline, "link": link})
    return news_items


def analyze_sentiment(news_items):
    sentiment_data = []
    for item in news_items:
        blob = TextBlob(item["headline"])
        polarity = blob.sentiment.polarity
        sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
        sentiment_data.append({**item, "polarity": polarity, "sentiment": sentiment})
    return pd.DataFrame(sentiment_data)


def render_sentiment_tab():
    st.title("📰 Market Sentiment Analysis")

    ticker = st.text_input("Enter ticker for sentiment analysis", "AAPL").strip().upper()
    if not ticker:
        return

    news_items = fetch_finviz_news(ticker)
    if not news_items:
        st.warning("No news headlines found.")
        return

    sentiment_df = analyze_sentiment(news_items)

    st.subheader("Latest Headlines Sentiment")
    st.dataframe(sentiment_df[["date", "headline", "sentiment", "polarity"]])

    # Filter last 7 days
    last_week = datetime.datetime.today().date() - datetime.timedelta(days=7)
    week_df = sentiment_df[sentiment_df["date"] >= last_week]

    if week_df.empty:
        st.info("No news in the past week.")
        return

    # Aggregate counts
    sentiment_counts = week_df["sentiment"].value_counts().to_dict()
    pos_count = sentiment_counts.get("positive", 0)
    neg_count = sentiment_counts.get("negative", 0)

    st.subheader("Sentiment in the Last Week")

    # Bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Positive"],
        y=[pos_count],
        marker_color='green',
        name="Positive"
    ))
    fig.add_trace(go.Bar(
        x=["Negative"],
        y=[neg_count],
        marker_color='red',
        name="Negative"
    ))
    fig.update_layout(
        title="Positive vs Negative Sentiment (Last 7 Days)",
        yaxis_title="Number of Headlines",
        xaxis_title="Sentiment",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)