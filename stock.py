import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from yahooquery import search
from newsapi import NewsApiClient
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Stock Market Prediction",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Smart Market")

# Cache for trending stock overview
@st.cache_data(ttl=3600)
def get_trending_data(symbols):
    data = []
    for symbol in symbols:
        try:
            ticker_obj = yf.Ticker(symbol)
            todays_data = ticker_obj.history(period="1d")
            last_close = todays_data["Close"].iloc[-1]
            previous_close = ticker_obj.info.get("previousClose", None)
            change = None
            if previous_close:
                change = ((last_close - previous_close) / previous_close) * 100
            data.append({
                "Symbol": symbol,
                "Last Close ($)": f"{last_close:.2f}",
                "Change (%)": f"{change:.2f}" if change is not None else "N/A",
            })
        except Exception:
            data.append({
                "Symbol": symbol,
                "Last Close ($)": "Error",
                "Change (%)": "Error",
            })
    return pd.DataFrame(data)

trending_tickers = ["AAPL", "TSLA", "AMZN", "MSFT", "GOOGL", "NVDA"]
st.subheader("Current Market Overview (Trending Stocks)")
st.table(get_trending_data(trending_tickers))

# Sidebar Search
st.sidebar.header("Select Stock")
query = st.sidebar.text_input("Search company", value="Apple")

@st.cache_data(ttl=3600)
def search_ticker(query):
    results = search(query)
    symbols = []
    if "quotes" in results:
        for item in results["quotes"]:
            symbol = item.get("symbol", "")
            longname = item.get("longname", "")
            if symbol and longname:
                symbols.append(f"{symbol} - {longname}")
    return symbols

ticker = ""
if query:
    symbols = search_ticker(query)
    if symbols:
        selected = st.sidebar.selectbox("Select Symbol", symbols)
        ticker = selected.split(" - ")[0].strip()
    else:
        st.sidebar.warning("No results found. Try another keyword.")
else:
    st.sidebar.info("Please enter a search term above.")

start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())

short_window = st.sidebar.number_input("Short MA Window", min_value=1, value=20)
long_window = st.sidebar.number_input("Long MA Window", min_value=1, value=50)

@st.cache_data(show_spinner=False)
def get_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

@st.cache_data(ttl=3600)
def get_news(query):
    newsapi = NewsApiClient(api_key='deaec211f36641aa875ed068d4bc8c03')
    return newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=5)

if st.sidebar.button("Predict"):
    if not ticker:
        st.warning("Please enter a valid stock ticker.")
    else:
        with st.spinner("Loading and predicting..."):
            try:
                data = get_stock_data(ticker, start_date, end_date)
                if data.empty:
                    st.error("No data found. Please check the ticker and date range.")
                else:
                    st.subheader(f"Historical Data for {ticker}")
                    st.dataframe(data.tail())

                    if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
                        data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
                        data["Long_MA"] = data["Close"].rolling(window=long_window).mean()

                        st.subheader("Candlestick Chart with Moving Averages")
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                                                     low=data["Low"], close=data["Close"],
                                                     name="Candlestick", increasing_line_color="green",
                                                     decreasing_line_color="red"))
                        fig.add_trace(go.Scatter(x=data.index, y=data["Short_MA"], mode="lines",
                                                 name=f"{short_window}-Day MA",
                                                 line=dict(color="blue", width=1.5)))
                        fig.add_trace(go.Scatter(x=data.index, y=data["Long_MA"], mode="lines",
                                                 name=f"{long_window}-Day MA",
                                                 line=dict(color="orange", width=1.5)))

                        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)",
                                          xaxis_rangeslider_visible=True, template="plotly_white",
                                          height=600, legend=dict(orientation="h", yanchor="bottom",
                                          y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Trading Signal")
                        if len(data) >= 2 and pd.notnull(data["Short_MA"].iloc[-1]) and pd.notnull(data["Long_MA"].iloc[-1]):
                            prev_short = data["Short_MA"].iloc[-2]
                            prev_long = data["Long_MA"].iloc[-2]
                            curr_short = data["Short_MA"].iloc[-1]
                            curr_long = data["Long_MA"].iloc[-1]
                            signal = "Hold"
                            if prev_short < prev_long and curr_short > curr_long:
                                signal = "Buy Signal"
                            elif prev_short > prev_long and curr_short < curr_long:
                                signal = "Sell Signal"
                            st.success(f"**{signal}**")
                        else:
                            st.info("Not enough data to generate a trading signal.")

                    data["Return"] = data["Close"].pct_change()
                    data["MA10"] = data["Close"].rolling(10).mean()
                    data["MA20"] = data["Close"].rolling(20).mean()
                    data = data.dropna()

                    features = data[["Open", "High", "Low", "Volume", "Return", "MA10", "MA20"]]
                    target_close = data["Close"]
                    target_high = data["High"]

                    X_train = features[:-1]
                    y_train_close = target_close[:-1]
                    y_train_high = target_high[:-1]
                    X_predict = features.iloc[-1].values.reshape(1, -1)
                    current_price = float(target_close.iloc[-1])

                    rf_close = RandomForestRegressor(n_estimators=200, random_state=42)
                    rf_close.fit(X_train, y_train_close)
                    pred_close_rf = rf_close.predict(X_predict)[0]

                    rf_high = RandomForestRegressor(n_estimators=200, random_state=42)
                    rf_high.fit(X_train, y_train_high)
                    pred_high_rf = rf_high.predict(X_predict)[0]

                    xgb_close = XGBRegressor(n_estimators=200, random_state=42)
                    xgb_close.fit(X_train, y_train_close)
                    pred_close_xgb = xgb_close.predict(X_predict)[0]

                    xgb_high = XGBRegressor(n_estimators=200, random_state=42)
                    xgb_high.fit(X_train, y_train_high)
                    pred_high_xgb = xgb_high.predict(X_predict)[0]

                    gb_close = GradientBoostingRegressor(n_estimators=200, random_state=42)
                    gb_close.fit(X_train, y_train_close)
                    pred_close_gb = gb_close.predict(X_predict)[0]

                    gb_high = GradientBoostingRegressor(n_estimators=200, random_state=42)
                    gb_high.fit(X_train, y_train_high)
                    pred_high_gb = gb_high.predict(X_predict)[0]

                    pred_close_avg = np.mean([pred_close_rf, pred_close_xgb, pred_close_gb])
                    pred_high_avg = np.mean([pred_high_rf, pred_high_xgb, pred_high_gb])

                    st.subheader("Predictions")
                    currency_symbol = "₹" if ticker.upper().endswith(".NS") else "$"
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
                    col2.metric("Predicted High Price", f"{currency_symbol}{pred_high_avg:.2f}")
                    col3.metric("Predicted Close Price", f"{currency_symbol}{pred_close_avg:.2f}")

                    last_open = float(data["Open"].iloc[-1])
                    last_close = float(data["Close"].iloc[-1])
                    sentiment = "**Bullish**" if last_close > last_open else "**Bearish**" if last_close < last_open else "**Neutral**"
                    st.subheader("Market Sentiment")
                    st.info(f"Based on the last candlestick: {sentiment}")
                    st.success("Prediction complete!")

                    st.subheader("Recent News (NewsAPI)")
                    articles = get_news(query)
                    if articles and articles.get('articles'):
                        for article in articles['articles']:
                            title = article.get('title', 'No Title')
                            source = article.get('source', {}).get('name', 'Unknown Source')
                            url = article.get('url', '#')
                            published_at = article.get('publishedAt', 'Unknown Time')
                            st.markdown(f"**[{title}]({url})**  \n*{source} — {published_at}*")
                            st.write("---")
                    else:
                        st.info("No news articles found.")
            except Exception as e:
                st.error(f"Error: {e}")

# Hide Streamlit footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
