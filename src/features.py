"""
This script takes raw market data and engineers features for modeling.
It adds:
- Time-based features (year, month, day, weekday)
- Return-based features (daily returns and lagged returns)
- Technical indicator (RSI-14)
"""

import pandas as pd
import numpy as np


def split_date(df):
    """
    Extracts useful time-based features from the Date column.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["Day_of_Week"] = df.Date.dt.dayofweek
    return df


def build_features(df):
    """
    Takes a raw dataframe and returns it with engineered
    features added (returns, lagged returns, RSI-14, volume/price range).
    """
    df = df.copy()
    df = split_date(df)
    df = df.drop(columns="Date")

    # Daily returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Lagged return features
    df["Lag_1"] = df["Daily_Return"].shift(1)
    df["Lag_2"] = df["Daily_Return"].shift(2)
    df["Lag_3"] = df["Daily_Return"].shift(3)

    # Compute RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Volume change captures unusual trading activity which often precedes price moves
    df["Volume_Change"] = df["Volume"].pct_change()

    # Price range measures intraday volatility relative to closing price
    df["Price_Range"] = (df["High"] - df["Low"]) / df["Close"]

    # Replace any infinity values with NaN, then drop remaining NaN rows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


if __name__ == "__main__":
    df_raw = pd.read_csv("data/raw/rawData.csv")
    df_processed = build_features(df_raw)

    df_processed["Future_Return"] = df_processed["Close"].pct_change().shift(-1)
    df_processed["Target"] = (df_processed["Future_Return"] > 0).astype(int)
    df_processed = df_processed.dropna()
    df_processed = df_processed.drop(columns=["Future_Return"])

    df_processed.to_csv("data/processed/processedData.csv", index=False)