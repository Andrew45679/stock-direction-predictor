"""
Streamlit app for predicting next-day stock price direction.

Fetches recent market data for a user-supplied ticker, builds features,
and uses a pretrained XGBoost model to predict whether the stock will
move up or down. Displays the prediction confidence, a SHAP waterfall
chart explaining the key drivers behind the prediction, and a recent
price chart.
"""
import streamlit as st
import joblib
import shap

from features import build_features
from data_pull import fetch_market_data
from datetime import date, timedelta
import matplotlib.pyplot as plt


# Load model/scaler once
model = joblib.load("models/model_XGB.joblib")
scaler = joblib.load("models/scaler.joblib")

MIN_ROWS_NEEDED = 80

# Page setup
st.title("Stock Direction Predictor")
ticker = st.text_input("Enter a ticker:")

# Getting the date from today and 180 days in the past
today = date.today()
start = today - timedelta(days=180)

# The button for predicting 
if st.button("Predict"):
    ticker = ticker.strip().upper()

    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        try:
            data = fetch_market_data(ticker, start_date=start, end_date=today)
        except Exception as e:
            st.error(f"Couldn't fetch data for '{ticker}': {e}")
            st.stop()

        if data.empty:
            st.error(f"No data found for '{ticker}'. Check the ticker symbol and try again.")
        elif len(data) < MIN_ROWS_NEEDED:
            st.error(f"Not enough historical data for '{ticker}' to generate a prediction.")
        else:
            st.success(f"Fetched {len(data)} rows of data for {ticker}.")

            features = build_features(data)
            features = features.drop(columns="Year")

            # Use only the most recent day for prediction
            latest_features = features.iloc[[-1]]

            # Scale and predict
            scaled = scaler.transform(latest_features)
            prediction = model.predict(scaled)[0]
            confidence = model.predict_proba(scaled)[0].max()

            # Display prediction
            direction = "UP 📈" if prediction == 1 else "DOWN 📉"
            st.metric("Predicted Direction:", direction)
            st.write(f"Confidence: {confidence:.1%}")

            # SHAP explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled)

            # Displaying a waterfall chart about all the features effecting the model
            st.write("What's driving this prediction:")
            fig, ax = plt.subplots()
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=latest_features.iloc[0],
                    feature_names=latest_features.columns
                ),
                show=False
            )
            st.pyplot(fig)

            # Recent price chart
            st.write("The recent Price chart over 180 days:")
            st.line_chart(data["Close"])