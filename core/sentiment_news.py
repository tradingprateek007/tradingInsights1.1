import streamlit as st
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import plotly.graph_objects as go


def fetch_newsapi_news(ticker, api_key):
    from_date = (datetime.datetime.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = datetime.datetime.today().strftime("%Y-%m-%d")
    query = f"{ticker} stock"

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}&language=en"
    )

    resp = requests.get(url)
    if resp.status_code != 200:
        st.warning(f"NewsAPI error: {resp.status_code} - {resp.text}")
        return None

    articles = resp.json().get("articles", [])
    news_items = []

    for art in articles:
        pub_date = datetime.datetime.strptime(art["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").date()
        news_items.append({
            "date": pub_date,
            "headline": art["title"],
            "link": art["url"]
        })
    return news_items


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

    current_date = datetime.datetime.today().date()

    for row in rows:
        tds = row.find_all("td")
        if len(tds) != 2:
            continue

        dt_text = tds[0].text.strip()
        a_tag = tds[1].a
        if a_tag is None:
            continue

        link = a_tag.get("href", "")
        headline = a_tag.text.strip()

        try:
            if ':' in dt_text:
                date = current_date
            else:
                current_date = datetime.datetime.strptime(dt_text, "%b-%d-%y").date()
                date = current_date
        except:
            date = current_date

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


def plot_sentiment_pie(df, title):
    counts = df["sentiment"].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=["green", "red", "grey"]
    )])
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_tab():
    st.title("📰 Market Sentiment Analysis")

    ticker = st.text_input("Enter ticker for sentiment analysis", "AAPL").strip().upper()
    if not ticker:
        return

    api_key = st.secrets.get("newsapi_key", None)

    newsapi_df = pd.DataFrame()
    finviz_df = pd.DataFrame()

    if api_key:
        newsapi_items = fetch_newsapi_news(ticker, api_key)
        if newsapi_items is not None:
            newsapi_df = analyze_sentiment(newsapi_items)
        else:
            st.info("Falling back to Finviz headlines.")
            finviz_items = fetch_finviz_news(ticker)
            finviz_df = analyze_sentiment(finviz_items)
    else:
        st.info("No NewsAPI key found — using Finviz only.")
        finviz_items = fetch_finviz_news(ticker)
        finviz_df = analyze_sentiment(finviz_items)

    # 📊 Display dataframes
    if not newsapi_df.empty:
        st.subheader("NewsAPI Headlines Sentiment")
        st.dataframe(newsapi_df[["date", "headline", "sentiment", "polarity"]])
        plot_sentiment_pie(newsapi_df, "NewsAPI Sentiment Breakdown")

    if not finviz_df.empty:
        st.subheader("Finviz Headlines Sentiment")
        st.dataframe(finviz_df[["date", "headline", "sentiment", "polarity"]])
        plot_sentiment_pie(finviz_df, "Finviz Sentiment Breakdown")

    # 📊 Bar chart — take whichever we have data for
    df = newsapi_df if not newsapi_df.empty else finviz_df

    last_week = datetime.datetime.today().date() - datetime.timedelta(days=7)
    week_df = df[df["date"] >= last_week]

    if week_df.empty:
        st.info("No news in the past week.")
        return

    st.subheader("Daily Sentiment Counts (Last 7 Days)")

    daily_counts = (
        week_df.groupby(["date", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("date")
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_counts["date"],
        y=daily_counts.get("positive", [0] * len(daily_counts)),
        marker_color='green',
        name="Positive"
    ))
    fig.add_trace(go.Bar(
        x=daily_counts["date"],
        y=daily_counts.get("negative", [0] * len(daily_counts)),
        marker_color='red',
        name="Negative"
    ))

    fig.update_layout(
        title="Positive vs Negative Sentiment Per Day (Last 7 Days)",
        yaxis_title="Number of Headlines",
        xaxis_title="Date",
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)