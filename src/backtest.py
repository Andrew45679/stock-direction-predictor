"""
Runs a backtest on the trained XGBoost model's predictions, computes
strategy metrics (Sharpe, max drawdown, win rate, returns), and plots
the strategy equity curve against a buy-and-hold benchmark.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

TRANSACTION_COST = 0

# Load trained model and scaler
model_XGB = joblib.load("models/model_XGB.joblib")
scaler = joblib.load("models/scaler.joblib")

# Load processed data and build a date column
df = pd.read_csv("data/processed/processedData.csv")
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

# Make the test split
test_df = df[df["Year"] >= 2023].drop(columns="Year").set_index("Date")

# Split into inputs and target
test_inputs = test_df.drop(columns=["Target"])
test_target = test_df["Target"]

# Scale the inputs
test_inputs_scaled = scaler.transform(test_inputs)

# Generate predictions
predictions = pd.Series(model_XGB.predict(test_inputs_scaled), index=test_df.index)

# Shift position forward one day so we trade using yesterday's prediction
test_df["position"] = predictions
test_df["position"] = test_df["position"].shift(1).fillna(0)

# Cost charged whenever the position changes
test_df["trade_cost"] = test_df["position"].diff().abs().fillna(0) * TRANSACTION_COST

# Daily return earned by the strategy
test_df["strategy_return"] = (test_df["position"] * test_df["Daily_Return"]) - test_df["trade_cost"]

# Running equity curves (starting at 1.0)
test_df["strategy_equity"] = (1 + test_df["strategy_return"]).cumprod()
test_df["buy_hold"] = (1 + test_df["Daily_Return"]).cumprod()

# Performance metrics
sharpe = (test_df["strategy_return"].mean() / test_df["strategy_return"].std()) * np.sqrt(252)
max_dd = (test_df["strategy_equity"] / test_df["strategy_equity"].cummax() - 1).min()
active_days = test_df[test_df["position"] != 0]
win_rate = (active_days["strategy_return"] > 0).mean() if len(active_days) > 0 else float("nan")

# Print results
print("Sharpe Ratio:      {:.2f}".format(sharpe))
print("Max Drawdown:      {:.2%}".format(max_dd))
print("Win Rate:          {:.2%}".format(win_rate))
print("Cumulative Return: {:.2%}".format(test_df["strategy_equity"].iloc[-1] - 1))
print("Buy & Hold Return: {:.2%}".format(test_df["buy_hold"].iloc[-1] - 1))

# Plot the graph of Strategy vs Buy-and-Hold Equity Curve
plt.figure(figsize=(10, 6))
plt.plot(test_df["strategy_equity"], label="Strategy")
plt.plot(test_df["buy_hold"], label="Buy & Hold")
plt.title("Strategy vs Buy-and-Hold Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.savefig("outputs/figures/equity_curve.png", dpi=300, bbox_inches="tight")
plt.show()