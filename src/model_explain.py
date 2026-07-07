"""
Uses SHAP to interpret the trained XGBoost model's predictions, 
showing which features drive the model's buy/sell signals. Then 
saves a summary plot to outputs/figures/
"""
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and data
model_XGB = joblib.load("models/model_XGB.joblib")
scaler = joblib.load("models/scaler.joblib")

df = pd.read_csv("data/processed/processedData.csv")
test_df = df[df["Year"] >= 2023].drop(columns="Year")
test_inputs = test_df.drop(columns=["Target"])
test_inputs_scaled = scaler.transform(test_inputs)

# SHAP explainer
explainer = shap.TreeExplainer(model_XGB)
shap_values = explainer.shap_values(test_inputs_scaled)

# Summary plot
shap.summary_plot(shap_values, test_inputs_scaled, feature_names=test_inputs.columns, show=False)
plt.savefig("outputs/figures/shap_summary.png", dpi=300, bbox_inches="tight")