"""
Data Processing Script

This script takes raw market data and engineers features for modeling.
It adds:
- Time-based features (year, month, day, weekday)
- Return-based features (daily returns and lagged returns)
- Technical indicator (RSI-14)

Output is saved as a cleaned dataset ready for analysis or ML models.
"""

import pandas as pd
import numpy as np

# Load raw dataset
df_raw = pd.read_csv("data/raw/rawData.csv")


# Extract date features
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


# Apply date feature engineering
df_processed = split_date(df_raw)

# Drop original Date column after extraction
df_processed = df_processed.drop(columns="Date")


# Create daily returns
df_processed["Daily_Return"] = df_processed["Close"].pct_change()

# Create lagged return features
df_processed["Lag_1"] = df_processed["Daily_Return"].shift(1)
df_processed["Lag_2"] = df_processed["Daily_Return"].shift(2)
df_processed["Lag_3"] = df_processed["Daily_Return"].shift(3)


# Compute RSI (14-day)
delta = df_processed["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

# Average gains and losses over 14 periods
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

# Relative Strength Index
rs = avg_gain / avg_loss
df_processed["RSI_14"] = 100 - (100 / (1 + rs))

# Adding a target column
df_processed["Future_Return"] = df_processed["Close"].pct_change().shift(-1)
df_processed["Target"] = (df_processed["Future_Return"] > 0).astype(int)

# Volume change captures unusual trading activity which often precedes price moves
df_processed['Volume_Change'] = df_processed['Volume'].pct_change()

# Price range measures intraday volatility relative to closing price
df_processed['Price_Range'] = (df_processed['High'] - df_processed['Low']) / df_processed['Close']

# Replace any infinity values with NaN
df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

# Remove NaN rows from dataset
df_processed = df_processed.dropna()
df_processed = df_processed.drop(columns=['Future_Return'])

# Save processed dataset
df_processed.to_csv("data/processed/processedData.csv", index=False)