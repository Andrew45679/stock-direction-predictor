"""
Market Data Collection Script

This script downloads historical financial data using Yahoo Finance and
combines multiple market indicators into a single dataset.

Data sources:
- S&P 500 (^GSPC)
- Volatility Index (^VIX)
- 10-Year Treasury Yield (^TNX)

The final dataset is saved as a CSV for downstream analysis or modeling.
"""

import yfinance as yf
import pandas as pd

# Define time range for data collection
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"

# Download market index data for the S&P 500
df_price = yf.download("^GSPC", start=START_DATE, end=END_DATE)

# Download volatility index 
df_vix = yf.download("^VIX", start=START_DATE, end=END_DATE)

# Download 10-year treasury yield 
df_tnx = yf.download("^TNX", start=START_DATE, end=END_DATE)

# Start building final dataset using S&P 500 as the base
final_df = df_price.copy()

# Adding more columns to the base from the volatility index and 10-year treasury yield
final_df["VIX_close"] = df_vix["Close"]
final_df["TNX_close"] = df_tnx["Close"]

# Remove rows with missing values
final_df = final_df.dropna()

# Flatten multi-index columns by getting rid of the Tickers
final_df = final_df.droplevel(1, axis=1)

# Save cleaned dataset in data/raw
final_df.to_csv("data/raw/rawData.csv", index=True)

