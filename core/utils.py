import streamlit as st
from alpaca_trade_api.rest import REST

API_KEY = st.secrets.get("ALPACA_API_KEY")
SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY")
BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = REST(API_KEY, SECRET_KEY, BASE_URL)