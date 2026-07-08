"""
Market Data Collection Script

This script downloads historical financial data using Yahoo Finance and
combines multiple market indicators into a single dataset.

Data sources:
- Stock Price
- Volatility Index (^VIX)
- 10-Year Treasury Yield (^TNX)

The final dataset is saved as a CSV for downstream analysis or modeling.
"""

import yfinance as yf
import pandas as pd


def fetch_market_data(ticker, start_date, end_date):
    """
    Downloads price data for the given ticker plus VIX and TNX,
    merges them into a single dataframe.
    """
    df_price = yf.download(ticker, start=start_date, end=end_date)
    df_vix = yf.download("^VIX", start=start_date, end=end_date)
    df_tnx = yf.download("^TNX", start=start_date, end=end_date)

    final_df = df_price.copy()
    final_df["VIX_close"] = df_vix["Close"]
    final_df["TNX_close"] = df_tnx["Close"]

    final_df = final_df.dropna()
    final_df = final_df.droplevel(1, axis=1)
    final_df = final_df.reset_index()

    return final_df


if __name__ == "__main__":
    TICKER = input("Enter ticker symbol (e.g. AAPL, ^GSPC, TSLA): ")
    START_DATE = "2015-01-01"
    END_DATE = "2024-01-01"

    final_df = fetch_market_data(TICKER, START_DATE, END_DATE)
    final_df.to_csv("data/raw/rawData.csv", index=True)